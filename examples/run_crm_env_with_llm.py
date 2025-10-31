"""Run CrmEnv episodes with a live LLM agent and capture telemetry."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.crm_env import CrmEnv
from src.harness import ClaudeAgent, MockAgent, OpenAIAgent, build_prompt


PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4.1",
    "anthropic": "claude-sonnet-4-5-20250929",
}


def _maybe_load_dotenv() -> None:
    if load_dotenv is None:
        print("python-dotenv not installed; skipping automatic .env loading.")
        return
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        print("No .env file found; proceeding with current environment variables.")


def _create_agent(args: argparse.Namespace):
    provider = args.provider
    model_name = args.model or PROVIDER_DEFAULT_MODELS.get(provider)

    if provider == "mock":
        return MockAgent()

    if provider == "openai":
        return OpenAIAgent(
            model_name=model_name or PROVIDER_DEFAULT_MODELS["openai"],
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )

    if provider == "anthropic":
        return ClaudeAgent(
            model_name=model_name or PROVIDER_DEFAULT_MODELS["anthropic"],
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )

    raise ValueError(f"Unsupported provider '{provider}'.")


def _format_tool_hints(hints: Mapping[str, str]) -> str:
    lines = ["Available CRM tools:"]
    for tool, signature in sorted(hints.items()):
        lines.append(f"- {tool}: {signature}")
    return "\n".join(lines)


def run_episode(
    env: CrmEnv,
    agent,
    *,
    episode_index: int,
    reset_seed: Optional[int] = None,
    reset_options: Optional[Dict[str, Any]] = None,
    log_steps: bool = True,
    include_hints_in_prompt: bool = True,
) -> Dict[str, Any]:
    observation, info = env.reset(seed=reset_seed, options=reset_options)
    case = env.active_case
    context = env.active_context or {}
    if case is None:
        raise RuntimeError("Environment did not produce an active case after reset().")

    tool_hints: Mapping[str, str] = info.get("tool_hints", {}) or {}
    prompt = build_prompt(case, context)
    if include_hints_in_prompt and tool_hints:
        prompt = f"{prompt}\n\n{_format_tool_hints(tool_hints)}"

    cumulative_reward = 0.0
    episode_steps: List[Dict[str, Any]] = []
    terminated = False
    truncated = False
    final_validator_success = False
    last_reward_breakdown: Dict[str, Any] = {}
    last_learning_signals: Dict[str, Any] = {}
    last_validator_metadata: Dict[str, Any] = {}

    while not (terminated or truncated):
        tool_call = agent.tool_call(case, prompt)
        action = {"tool_name": tool_call.tool_name, "arguments": tool_call.arguments}
        observation, reward, terminated, truncated, step_info = env.step(action)

        cumulative_reward += reward
        final_validator_success = bool(step_info.get("validator_success", False))
        last_reward_breakdown = step_info.get("reward_breakdown", {})
        last_learning_signals = step_info.get("learning_signals", {})
        last_validator_metadata = step_info.get("validator_metadata", {})

        step_record = {
            "step": len(episode_steps) + 1,
            "tool_name": tool_call.tool_name,
            "arguments": tool_call.arguments,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "validator_message": step_info.get("validator_message"),
            "validator_success": step_info.get("validator_success"),
            "reward_breakdown": last_reward_breakdown,
            "learning_signals": last_learning_signals,
            "validator_metadata": last_validator_metadata,
            "verifier": {
                "name": step_info.get("verifier_name"),
                "score": step_info.get("verifier_score"),
                "weight": step_info.get("verifier_weight"),
                "rationale": step_info.get("verifier_rationale"),
                "metadata": step_info.get("verifier_metadata"),
            },
        }

        episode_steps.append(step_record)

        if log_steps:
            print(json.dumps(step_record, indent=2))

        # Update prompt only if the environment provides new tool hints or history.
        if include_hints_in_prompt and tool_hints:
            prompt = f"{build_prompt(case, context)}\n\n{_format_tool_hints(tool_hints)}"

    episode_summary = {
        "episode_index": episode_index,
        "case_id": info.get("case_id"),
        "task": info.get("task"),
        "utterance": info.get("utterance"),
        "description": info.get("description"),
        "expected_success": info.get("expected_success"),
        "cumulative_reward": cumulative_reward,
        "terminated": terminated,
        "truncated": truncated,
        "validator_success": final_validator_success,
        "steps": episode_steps,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reward_breakdown": last_reward_breakdown,
        "learning_signals": last_learning_signals,
        "validator_metadata": last_validator_metadata,
    }

    return episode_summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll CrmEnv with a live LLM agent.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "mock"], default="openai")
    parser.add_argument("--model", help="Model identifier for the selected provider.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=2, help="Per-episode step limit passed to CrmEnv.")
    parser.add_argument("--max-output-tokens", type=int, default=1024, help="Maximum tokens returned by the model.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the model.")
    parser.add_argument("--include-negative-cases", action="store_true", help="Allow negative golden cases in sampling.")
    parser.add_argument("--log-json", type=Path, help="Optional JSONL file to append episode telemetry.")
    parser.add_argument("--seed", type=int, help="Base seed for environment resets.")
    parser.add_argument("--case-id", help="Force a specific golden-case identifier.")
    parser.add_argument("--task", help="Force sampling from a specific task group.")
    parser.add_argument("--no-dotenv", action="store_true", help="Disable automatic loading of .env files.")
    parser.add_argument("--no-step-logs", action="store_true", help="Suppress per-step JSON logs on stdout.")
    parser.add_argument("--no-tool-hints", action="store_true", help="Omit tool signature hints from prompts.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if not args.no_dotenv:
        _maybe_load_dotenv()

    try:
        agent = _create_agent(args)
    except Exception as exc:  # pragma: no cover - runtime configuration failure
        raise SystemExit(f"Failed to initialize agent: {exc}") from exc

    env = CrmEnv(
        max_steps=args.max_steps,
        include_negative_cases=args.include_negative_cases,
        expose_reference=False,
        include_tool_hints=not args.no_tool_hints,
        seed=args.seed,
    )

    if args.log_json:
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        log_file = args.log_json.open("a", encoding="utf-8")
    else:
        log_file = None

    successes = 0
    episodes: List[Dict[str, Any]] = []

    try:
        for episode_idx in range(args.episodes):
            reset_seed = args.seed + episode_idx if args.seed is not None else None
            options = {}
            if args.case_id:
                options["case_id"] = args.case_id
            if args.task:
                options["task"] = args.task
            if not options:
                options = None

            summary = run_episode(
                env,
                agent,
                episode_index=episode_idx,
                reset_seed=reset_seed,
                reset_options=options,
                log_steps=not args.no_step_logs,
                include_hints_in_prompt=not args.no_tool_hints,
            )

            successes += int(summary["validator_success"])
            episodes.append(summary)
            print(
                f"Episode {episode_idx}: success={summary['validator_success']} reward={summary['cumulative_reward']:.2f}"
            )

            if log_file:
                log_file.write(json.dumps(summary) + "\n")

    finally:
        env.close()
        if log_file:
            log_file.close()

    print(
        f"Completed {len(episodes)} episodes with success rate {successes}/{len(episodes)}"
    )


if __name__ == "__main__":
    main()
