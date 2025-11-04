"""CLI for running baseline evaluations on synthetic CRM scenarios or multi-turn conversations.

Usage (scenarios):
    python -m src.run_baseline \\
        --mode scenario \\
        --scenarios artifacts/generated_scenarios/scenarios.jsonl \\
        --agent claude \\
        --backend postgres \\
        --sample 10 \\
        --seed 42 \\
        --output artifacts/baseline_claude.jsonl

Usage (conversations):
    python -m src.run_baseline \\
        --mode conversation \\
        --conversations artifacts/conversations/conversations.jsonl \\
        --agent claude \\
        --backend postgres \\
        --sample 10 \\
        --seed 42 \\
        --output artifacts/baseline_claude_conversations.jsonl
"""

import argparse
import random
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv

from .crm_backend import DatabaseConfig
from .harness import ClaudeAgent, OpenAIAgent
from .scenario_generator import Scenario
from .scenario_harness import ScenarioBaselineHarness, ScenarioMockAgent, load_scenarios_from_jsonl

# Load environment variables from .env file
load_dotenv()


def sample_scenarios(
    scenarios: List[Scenario],
    count: int,
    seed: int | None = None,
    stratified: bool = True,
) -> List[Scenario]:
    """Sample scenarios, optionally maintaining success/failure ratio.

    Args:
        scenarios: Full list of scenarios
        count: Number to sample
        seed: Random seed for reproducibility
        stratified: If True, maintain success/failure ratio in sample

    Returns:
        Sampled scenarios
    """
    if seed is not None:
        random.seed(seed)

    if count >= len(scenarios):
        return scenarios

    if stratified:
        # Maintain success/failure ratio
        success_scenarios = [s for s in scenarios if s.expect_success]
        failure_scenarios = [s for s in scenarios if not s.expect_success]

        # Calculate ratio
        total = len(scenarios)
        success_ratio = len(success_scenarios) / total if total > 0 else 0.6

        # Sample proportionally
        success_count = round(count * success_ratio)
        failure_count = count - success_count

        sampled_success = random.sample(success_scenarios, min(success_count, len(success_scenarios)))
        sampled_failure = random.sample(failure_scenarios, min(failure_count, len(failure_scenarios)))

        sampled = sampled_success + sampled_failure
        random.shuffle(sampled)
        return sampled
    else:
        return random.sample(scenarios, count)


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation on synthetic CRM scenarios or multi-turn conversations"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["scenario", "conversation"],
        default="scenario",
        help="Execution mode: scenario (single-turn) or conversation (multi-turn)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        help="Path to scenarios JSONL file (required if mode=scenario)",
    )
    parser.add_argument(
        "--conversations",
        type=str,
        help="Path to conversations JSONL file (required if mode=conversation)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["claude", "gpt4.1", "gpt4", "mock"],
        help="Agent to use (claude=Claude 4.5 Sonnet, gpt4.1=GPT-4.1, gpt4=GPT-4, mock=ground truth)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mock",
        choices=["mock", "postgres"],
        help="Backend to use (mock=in-memory, postgres=real DB)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Number of scenarios to sample (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for JSONL log file",
    )
    parser.add_argument(
        "--enable-verifier",
        action="store_true",
        help="Enable Teacher/Student verification (Claude 4.5 grades responses)",
    )
    parser.add_argument(
        "--verifier-weight",
        type=float,
        default=0.3,
        help="Weight for verifier contribution to reward (default: 0.3)",
    )

    args = parser.parse_args()

    # Validate mode-specific arguments
    if args.mode == "conversation":
        if not args.conversations:
            print("Error: --conversations required when mode=conversation")
            return 1
        
        from .conversation_harness import ConversationHarness, ConversationMockAgent, load_conversations_from_jsonl
        
        conversations_path = Path(args.conversations)
        if not conversations_path.exists():
            print(f"Error: Conversations file not found: {conversations_path}")
            return 1

        print(f"Loading conversations from {conversations_path}...")
        conversations = load_conversations_from_jsonl(conversations_path)
        print(f"Loaded {len(conversations)} conversations")

        # Initialize agent
        print(f"Initializing {args.agent} agent...")
        if args.agent == "claude":
            agent = ClaudeAgent(model_name="claude-sonnet-4-5-20250929")
        elif args.agent == "gpt4.1":
            agent = OpenAIAgent(model_name="gpt-4.1")
        elif args.agent == "gpt4":
            agent = OpenAIAgent(model_name="gpt-4.1-mini")
        elif args.agent == "mock":
            agent = ConversationMockAgent()
        else:
            print(f"Error: Unknown agent: {args.agent}")
            return 1

        # Initialize harness
        print(f"Initializing conversation harness with {args.backend} backend...")
        db_config = DatabaseConfig.from_env() if args.backend == "postgres" else None

        harness = ConversationHarness(
            conversations=conversations,
            agent=agent,
            log_path=args.output,
            backend=args.backend,
            db_config=db_config,
            reset_database_each_case=True,
            enable_verifier=args.enable_verifier,
            verifier_name="structured",
            verifier_reward_weight=args.verifier_weight,
        )

        # Run evaluation
        print(f"\nRunning evaluation...")
        print(f"  Conversations: {len(conversations)}")
        print(f"  Agent: {agent.provider_name} / {agent.model_name}")
        print(f"  Backend: {args.backend}")
        print(f"  Verifier: {'enabled' if args.enable_verifier else 'disabled'}")
        print(f"  Output: {args.output}")
        print()

        mode = "mock" if args.agent == "mock" else "agent"
        result = harness.run(mode=mode)

        # Print results
        print("\n" + "="*80)
        print("BASELINE EVALUATION RESULTS")
        print("="*80)
        print(f"\nTotal conversations: {result['total']}")
        print(f"Success: {result['success_count']} ({result['success_count']/result['total']:.1%})")
        print(f"Failure: {result['failure_count']} ({result['failure_count']/result['total']:.1%})")
        print(f"\nLog file: {args.output}")

    else:  # scenario mode (existing)
        if not args.scenarios:
            print("Error: --scenarios required when mode=scenario")
            return 1

        scenarios_path = Path(args.scenarios)
        if not scenarios_path.exists():
            print(f"Error: Scenarios file not found: {scenarios_path}")
            return 1

        print(f"Loading scenarios from {scenarios_path}...")
        scenarios = load_scenarios_from_jsonl(scenarios_path)
        print(f"Loaded {len(scenarios)} scenarios")

        # Sample if requested
        if args.sample and args.sample < len(scenarios):
            print(f"Sampling {args.sample} scenarios with seed {args.seed}...")
            scenarios = sample_scenarios(scenarios, args.sample, args.seed, stratified=True)
            print(f"Sampled {len(scenarios)} scenarios (stratified by success/failure ratio)")

        # Initialize agent
        print(f"Initializing {args.agent} agent...")
        if args.agent == "claude":
            agent = ClaudeAgent(model_name="claude-sonnet-4-5-20250929")
        elif args.agent == "gpt4.1":
            agent = OpenAIAgent(model_name="gpt-4.1")
        elif args.agent == "gpt4":
            agent = OpenAIAgent(model_name="gpt-4.1-mini")
        elif args.agent == "mock":
            agent = ScenarioMockAgent()
        else:
            print(f"Error: Unknown agent: {args.agent}")
            return 1

        # Initialize harness
        print(f"Initializing harness with {args.backend} backend...")
        db_config = DatabaseConfig.from_env() if args.backend == "postgres" else None

        harness = ScenarioBaselineHarness(
            scenarios=scenarios,
            agent=agent,
            log_path=args.output,
            backend=args.backend,
            db_config=db_config,
            reset_database_each_case=True,
            enable_verifier=args.enable_verifier,
            verifier_name="structured",
            verifier_reward_weight=args.verifier_weight,
        )

        # Run evaluation
        print(f"\nRunning evaluation...")
        print(f"  Scenarios: {len(scenarios)}")
        print(f"  Agent: {agent.provider_name} / {agent.model_name}")
        print(f"  Backend: {args.backend}")
        print(f"  Verifier: {'enabled' if args.enable_verifier else 'disabled'}")
        print(f"  Output: {args.output}")
        print()

        mode = "mock" if args.agent == "mock" else "agent"
        result = harness.run(mode=mode)

        # Print results
        print("\n" + "="*80)
        print("BASELINE EVALUATION RESULTS")
        print("="*80)
        print(f"\nTotal scenarios: {result['total']}")
        print(f"Success: {result['success_count']} ({result['success_count']/result['total']:.1%})")
        print(f"Failure: {result['failure_count']} ({result['failure_count']/result['total']:.1%})")
        print(f"\nLog file: {args.output}")

        # Show sample failures
        failed_episodes = [ep for ep in result['episodes'] if not ep.success]
        if failed_episodes:
            print(f"\nSample failures ({min(5, len(failed_episodes))} of {len(failed_episodes)}):")
            for ep in failed_episodes[:5]:
                print(f"  - {ep.case_id} ({ep.task}): {ep.message[:80]}...")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
