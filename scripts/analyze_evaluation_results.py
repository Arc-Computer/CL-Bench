#!/usr/bin/env python3
"""Analyze evaluation results from baseline and Atlas runs.

Generates comprehensive comparison report in three formats:
1. Console summary (stdout)
2. Markdown report (detailed tables and analysis)
3. JSON data (machine-readable summary)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    results = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def calculate_baseline_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate metrics for baseline results."""
    total = len(results)
    if total == 0:
        return {
            "total_conversations": 0,
            "successful_conversations": 0,
            "task_success_rate": 0.0,
            "total_turns": 0,
            "successful_turns": 0,
            "turn_success_rate": 0.0,
            "judge_usage": {
                "judge_evaluated_count": 0,
                "judge_approved_count": 0,
                "judge_approval_rate": 0.0,
            },
            "token_usage": {},
        }

    successful = sum(1 for r in results if r.get("overall_success", False))
    task_success_rate = (successful / total * 100.0) if total > 0 else 0.0

    # Calculate turn-level metrics
    total_turns = 0
    successful_turns = 0
    judge_evaluated = 0
    judge_approved = 0
    token_totals = defaultdict(int)

    for result in results:
        per_turn = result.get("per_turn_results", [])
        total_turns += len(per_turn)
        for turn in per_turn:
            if turn.get("tool_success", False):
                successful_turns += 1
            if turn.get("judge_used", False):
                judge_evaluated += 1
                if turn.get("judge_pass", False):
                    judge_approved += 1

        # Aggregate token usage
        metadata = result.get("metadata", {})
        agent_meta = metadata.get("agent", {})
        token_usage = agent_meta.get("token_usage", {})
        for key, value in token_usage.items():
            if isinstance(value, (int, float)):
                token_totals[key] += value

    turn_success_rate = (successful_turns / total_turns * 100.0) if total_turns > 0 else 0.0
    judge_approval_rate = (judge_approved / judge_evaluated * 100.0) if judge_evaluated > 0 else 0.0

    return {
        "total_conversations": total,
        "successful_conversations": successful,
        "task_success_rate": task_success_rate,
        "total_turns": total_turns,
        "successful_turns": successful_turns,
        "turn_success_rate": turn_success_rate,
        "judge_usage": {
            "judge_evaluated_count": judge_evaluated,
            "judge_approved_count": judge_approved,
            "judge_approval_rate": judge_approval_rate,
        },
        "token_usage": dict(token_totals),
    }


def calculate_atlas_metrics(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate metrics for Atlas sessions."""
    total = len(sessions)
    if total == 0:
        return {
            "total_sessions": 0,
            "successful_sessions": 0,
            "task_success_rate": 0.0,
            "learning_growth": {},
            "reward_trends": {},
        }

    successful = 0
    reward_scores = []
    learning_lengths = []

    for session in sessions:
        # Parse conversation result if available
        final_answer = session.get("final_answer", "")
        if isinstance(final_answer, str):
            try:
                answer_dict = json.loads(final_answer)
                conv_result = answer_dict.get("conversation_result", {})
                if conv_result.get("overall_success", False):
                    successful += 1
                reward = conv_result.get("reward_signal", 0.0)
                if reward:
                    reward_scores.append(reward)
            except (json.JSONDecodeError, TypeError):
                pass

        # Extract learning state if available
        metadata = session.get("metadata", {})
        learning_state = metadata.get("learning_state", {})
        student_learning = learning_state.get("student_learning", "")
        if student_learning:
            learning_lengths.append(len(student_learning))

    task_success_rate = (successful / total * 100.0) if total > 0 else 0.0

    reward_trends = {}
    if reward_scores:
        reward_trends = {
            "mean": sum(reward_scores) / len(reward_scores),
            "min": min(reward_scores),
            "max": max(reward_scores),
            "count": len(reward_scores),
        }

    learning_growth = {}
    if learning_lengths:
        learning_growth = {
            "initial_length": learning_lengths[0] if learning_lengths else 0,
            "final_length": learning_lengths[-1] if learning_lengths else 0,
            "growth": learning_lengths[-1] - learning_lengths[0] if len(learning_lengths) > 1 else 0,
            "max_length": max(learning_lengths) if learning_lengths else 0,
        }

    return {
        "total_sessions": total,
        "successful_sessions": successful,
        "task_success_rate": task_success_rate,
        "learning_growth": learning_growth,
        "reward_trends": reward_trends,
    }


def estimate_cost(token_usage: Dict[str, int], model: str) -> Dict[str, float]:
    """Estimate cost based on token usage."""
    # Rough pricing estimates (update with actual rates)
    pricing = {
        "claude-sonnet-4-5-20250929": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
        "gpt-4.1": {"input": 10.0 / 1_000_000, "output": 30.0 / 1_000_000},
        "gpt-4.1-mini": {"input": 0.15 / 1_000_000, "output": 0.6 / 1_000_000},
    }

    model_pricing = pricing.get(model, {"input": 0.0, "output": 0.0})
    input_tokens = token_usage.get("input_tokens", 0)
    output_tokens = token_usage.get("output_tokens", 0)

    input_cost = input_tokens * model_pricing["input"]
    output_cost = output_tokens * model_pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    }


def print_console_summary(
    baseline_metrics: Dict[str, Dict[str, Any]], atlas_metrics: Dict[str, Any]
) -> None:
    """Print formatted summary to console."""
    print("=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print()

    print("BASELINE RESULTS")
    print("-" * 80)
    for agent_name, metrics in baseline_metrics.items():
        print(f"\n{agent_name.upper()}:")
        print(f"  Conversations: {metrics['successful_conversations']}/{metrics['total_conversations']}")
        print(f"  Task Success Rate: {metrics['task_success_rate']:.2f}%")
        print(f"  Turn Success Rate: {metrics['turn_success_rate']:.2f}%")
        print(f"  Judge Evaluated: {metrics['judge_usage']['judge_evaluated_count']}")
        print(f"  Judge Approval Rate: {metrics['judge_usage']['judge_approval_rate']:.2f}%")

        if metrics["token_usage"]:
            model_name = agent_name.replace("_", "-")
            cost = estimate_cost(metrics["token_usage"], model_name)
            print(f"  Total Tokens: {cost['total_tokens']:,}")
            print(f"  Estimated Cost: ${cost['total_cost_usd']:.2f}")

    print("\n" + "-" * 80)
    print("ATLAS RESULTS")
    print("-" * 80)
    print(f"\nSessions: {atlas_metrics['successful_sessions']}/{atlas_metrics['total_sessions']}")
    print(f"Task Success Rate: {atlas_metrics['task_success_rate']:.2f}%")

    if atlas_metrics["learning_growth"]:
        lg = atlas_metrics["learning_growth"]
        print(f"\nLearning Growth:")
        print(f"  Initial Length: {lg['initial_length']} chars")
        print(f"  Final Length: {lg['final_length']} chars")
        print(f"  Growth: +{lg['growth']} chars")

    if atlas_metrics["reward_trends"]:
        rt = atlas_metrics["reward_trends"]
        print(f"\nReward Trends:")
        print(f"  Mean: {rt['mean']:.3f}")
        print(f"  Range: {rt['min']:.3f} - {rt['max']:.3f}")

    print("\n" + "=" * 80)


def generate_markdown_report(
    baseline_metrics: Dict[str, Dict[str, Any]], atlas_metrics: Dict[str, Any]
) -> str:
    """Generate detailed markdown report."""
    lines = []
    lines.append("# Evaluation Results Report")
    lines.append("")
    lines.append("## Baseline Results")
    lines.append("")
    lines.append("| Agent | Conversations | Task Success Rate | Turn Success Rate | Judge Evaluated | Judge Approval Rate |")
    lines.append("|-------|--------------|-------------------|-------------------|-----------------|---------------------|")

    for agent_name, metrics in baseline_metrics.items():
        display_name = agent_name.replace("_", " ").title()
        lines.append(
            f"| {display_name} | "
            f"{metrics['successful_conversations']}/{metrics['total_conversations']} | "
            f"{metrics['task_success_rate']:.2f}% | "
            f"{metrics['turn_success_rate']:.2f}% | "
            f"{metrics['judge_usage']['judge_evaluated_count']} | "
            f"{metrics['judge_usage']['judge_approval_rate']:.2f}% |"
        )

    lines.append("")
    lines.append("### Token Usage and Cost Estimates")
    lines.append("")
    lines.append("| Agent | Input Tokens | Output Tokens | Total Tokens | Estimated Cost (USD) |")
    lines.append("|-------|--------------|----------------|--------------|----------------------|")

    for agent_name, metrics in baseline_metrics.items():
        if metrics["token_usage"]:
            model_name = agent_name.replace("_", "-")
            cost = estimate_cost(metrics["token_usage"], model_name)
            display_name = agent_name.replace("_", " ").title()
            lines.append(
                f"| {display_name} | "
                f"{cost['input_tokens']:,} | "
                f"{cost['output_tokens']:,} | "
                f"{cost['total_tokens']:,} | "
                f"${cost['total_cost_usd']:.2f} |"
            )

    lines.append("")
    lines.append("## Atlas Results")
    lines.append("")
    lines.append(f"**Total Sessions:** {atlas_metrics['total_sessions']}")
    lines.append(f"**Successful Sessions:** {atlas_metrics['successful_sessions']}")
    lines.append(f"**Task Success Rate:** {atlas_metrics['task_success_rate']:.2f}%")
    lines.append("")

    if atlas_metrics["learning_growth"]:
        lg = atlas_metrics["learning_growth"]
        lines.append("### Learning Growth")
        lines.append("")
        lines.append(f"- **Initial Learning Length:** {lg['initial_length']} characters")
        lines.append(f"- **Final Learning Length:** {lg['final_length']} characters")
        lines.append(f"- **Total Growth:** +{lg['growth']} characters")
        lines.append(f"- **Maximum Length:** {lg['max_length']} characters")
        lines.append("")

    if atlas_metrics["reward_trends"]:
        rt = atlas_metrics["reward_trends"]
        lines.append("### Reward Trends")
        lines.append("")
        lines.append(f"- **Mean Reward:** {rt['mean']:.3f}")
        lines.append(f"- **Min Reward:** {rt['min']:.3f}")
        lines.append(f"- **Max Reward:** {rt['max']:.3f}")
        lines.append(f"- **Sessions with Rewards:** {rt['count']}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("--baseline-claude", type=Path, required=True, help="Claude baseline results JSONL")
    parser.add_argument("--baseline-gpt4", type=Path, required=True, help="GPT-4.1 baseline results JSONL")
    parser.add_argument("--baseline-gpt4mini", type=Path, required=True, help="GPT-4.1 mini baseline results JSONL")
    parser.add_argument("--atlas-sessions", type=Path, required=True, help="Atlas sessions JSONL")
    parser.add_argument("--output-report", type=Path, required=True, help="Output markdown report path")
    parser.add_argument("--output-json", type=Path, required=True, help="Output JSON summary path")

    args = parser.parse_args()

    # Load results
    print("Loading baseline results...")
    claude_results = load_jsonl(args.baseline_claude)
    gpt4_results = load_jsonl(args.baseline_gpt4)
    gpt4mini_results = load_jsonl(args.baseline_gpt4mini)

    print("Loading Atlas sessions...")
    atlas_sessions = load_jsonl(args.atlas_sessions)

    # Calculate metrics
    print("Calculating metrics...")
    baseline_metrics = {
        "claude_sonnet_4_5": calculate_baseline_metrics(claude_results),
        "gpt4_1": calculate_baseline_metrics(gpt4_results),
        "gpt4_1_mini": calculate_baseline_metrics(gpt4mini_results),
    }
    atlas_metrics = calculate_atlas_metrics(atlas_sessions)

    # Print console summary
    print_console_summary(baseline_metrics, atlas_metrics)

    # Generate markdown report
    markdown_report = generate_markdown_report(baseline_metrics, atlas_metrics)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_report.open("w", encoding="utf-8") as f:
        f.write(markdown_report)
    print(f"\n✓ Markdown report written to {args.output_report}")

    # Generate JSON summary
    json_summary = {
        "baseline_metrics": baseline_metrics,
        "atlas_metrics": atlas_metrics,
        "timestamp": json.dumps({}, default=str),  # Add timestamp if needed
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, default=str)
    print(f"✓ JSON summary written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

