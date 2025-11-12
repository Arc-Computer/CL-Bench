#!/usr/bin/env python3
"""Quick analysis of alternative metrics from existing baseline data.

Computes multiple evaluation perspectives:
1. Strict (current): All turns must succeed (tool + response)
2. Tool-only: Tool execution success (ignoring response quality)
3. Goal achievement: Final turn success (did they complete the task?)
4. Partial success: ≥80% of turns succeed
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    results = []
    if not file_path.exists():
        return results
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def calculate_alternative_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate multiple metric perspectives from the same data."""
    total = len(results)
    if total == 0:
        return {
            "total_conversations": 0,
            "strict_success_rate": 0.0,
            "tool_only_success_rate": 0.0,
            "goal_achievement_rate": 0.0,
            "partial_success_rate": 0.0,
            "turn_level_metrics": {},
        }

    # Current strict metric (all turns must succeed)
    strict_successful = sum(1 for r in results if r.get("overall_success", False))
    strict_success_rate = (strict_successful / total * 100.0) if total > 0 else 0.0

    # Tool-only success (all tools execute successfully, ignoring response quality)
    tool_only_successful = 0
    goal_achievement_successful = 0
    partial_successful = 0
    
    # Turn-level metrics
    total_turns = 0
    tool_successful_turns = 0
    response_successful_turns = 0
    final_turn_tool_success = 0
    final_turn_response_success = 0
    
    for result in results:
        per_turn = result.get("per_turn_results", [])
        if not per_turn:
            continue
            
        total_turns += len(per_turn)
        
        # Check if all tools succeeded
        all_tools_succeeded = all(turn.get("tool_success", False) for turn in per_turn)
        if all_tools_succeeded:
            tool_only_successful += 1
        
        # Check final turn (goal achievement)
        final_turn = per_turn[-1]
        if final_turn.get("tool_success", False):
            final_turn_tool_success += 1
        if final_turn.get("response_success", False):
            final_turn_response_success += 1
        if final_turn.get("success", False):  # Both tool and response
            goal_achievement_successful += 1
        
        # Count successful turns
        for turn in per_turn:
            if turn.get("tool_success", False):
                tool_successful_turns += 1
            if turn.get("response_success", False):
                response_successful_turns += 1
        
        # Partial success: ≥80% of turns succeed
        success_path_turns = [t for t in per_turn if t.get("expect_success", True)]
        if success_path_turns:
            successful_turns = sum(1 for t in success_path_turns if t.get("success", False))
            success_rate = successful_turns / len(success_path_turns)
            if success_rate >= 0.8:
                partial_successful += 1
    
    tool_only_success_rate = (tool_only_successful / total * 100.0) if total > 0 else 0.0
    goal_achievement_rate = (goal_achievement_successful / total * 100.0) if total > 0 else 0.0
    partial_success_rate = (partial_successful / total * 100.0) if total > 0 else 0.0
    
    turn_tool_success_rate = (tool_successful_turns / total_turns * 100.0) if total_turns > 0 else 0.0
    turn_response_success_rate = (response_successful_turns / total_turns * 100.0) if total_turns > 0 else 0.0
    final_turn_tool_rate = (final_turn_tool_success / total * 100.0) if total > 0 else 0.0
    final_turn_response_rate = (final_turn_response_success / total * 100.0) if total > 0 else 0.0
    
    return {
        "total_conversations": total,
        "strict_success_rate": strict_success_rate,
        "strict_successful": strict_successful,
        "tool_only_success_rate": tool_only_success_rate,
        "tool_only_successful": tool_only_successful,
        "goal_achievement_rate": goal_achievement_rate,
        "goal_achievement_successful": goal_achievement_successful,
        "partial_success_rate": partial_success_rate,
        "partial_successful": partial_successful,
        "turn_level_metrics": {
            "total_turns": total_turns,
            "tool_success_rate": turn_tool_success_rate,
            "response_success_rate": turn_response_success_rate,
            "final_turn_tool_rate": final_turn_tool_rate,
            "final_turn_response_rate": final_turn_response_rate,
        },
    }


def print_metrics_table(metrics_dict: Dict[str, Dict[str, Any]]) -> None:
    """Print formatted metrics table."""
    print("\n" + "=" * 100)
    print("ALTERNATIVE METRICS ANALYSIS")
    print("=" * 100)
    print("\nMultiple evaluation perspectives from the same baseline data:")
    print()
    
    # Header
    print(f"{'Metric':<40} {'Claude 4.5':<20} {'GPT-4.1':<20} {'GPT-4.1 Mini':<20}")
    print("-" * 100)
    
    # Strict (current)
    print(f"{'1. Strict (All turns succeed)':<40} "
          f"{metrics_dict['claude']['strict_success_rate']:>6.2f}% ({metrics_dict['claude']['strict_successful']:>4}/{metrics_dict['claude']['total_conversations']:<4}) "
          f"{metrics_dict['gpt4']['strict_success_rate']:>6.2f}% ({metrics_dict['gpt4']['strict_successful']:>4}/{metrics_dict['gpt4']['total_conversations']:<4}) "
          f"{metrics_dict['gpt4mini']['strict_success_rate']:>6.2f}% ({metrics_dict['gpt4mini']['strict_successful']:>4}/{metrics_dict['gpt4mini']['total_conversations']:<4})")
    
    # Tool-only
    print(f"{'2. Tool-only (All tools execute)':<40} "
          f"{metrics_dict['claude']['tool_only_success_rate']:>6.2f}% ({metrics_dict['claude']['tool_only_successful']:>4}/{metrics_dict['claude']['total_conversations']:<4}) "
          f"{metrics_dict['gpt4']['tool_only_success_rate']:>6.2f}% ({metrics_dict['gpt4']['tool_only_successful']:>4}/{metrics_dict['gpt4']['total_conversations']:<4}) "
          f"{metrics_dict['gpt4mini']['tool_only_success_rate']:>6.2f}% ({metrics_dict['gpt4mini']['tool_only_successful']:>4}/{metrics_dict['gpt4mini']['total_conversations']:<4})")
    
    # Goal achievement
    print(f"{'3. Goal achievement (Final turn)':<40} "
          f"{metrics_dict['claude']['goal_achievement_rate']:>6.2f}% ({metrics_dict['claude']['goal_achievement_successful']:>4}/{metrics_dict['claude']['total_conversations']:<4}) "
          f"{metrics_dict['gpt4']['goal_achievement_rate']:>6.2f}% ({metrics_dict['gpt4']['goal_achievement_successful']:>4}/{metrics_dict['gpt4']['total_conversations']:<4}) "
          f"{metrics_dict['gpt4mini']['goal_achievement_rate']:>6.2f}% ({metrics_dict['gpt4mini']['goal_achievement_successful']:>4}/{metrics_dict['gpt4mini']['total_conversations']:<4})")
    
    # Partial success
    print(f"{'4. Partial success (≥80% turns)':<40} "
          f"{metrics_dict['claude']['partial_success_rate']:>6.2f}% ({metrics_dict['claude']['partial_successful']:>4}/{metrics_dict['claude']['total_conversations']:<4}) "
          f"{metrics_dict['gpt4']['partial_success_rate']:>6.2f}% ({metrics_dict['gpt4']['partial_successful']:>4}/{metrics_dict['gpt4']['total_conversations']:<4}) "
          f"{metrics_dict['gpt4mini']['partial_success_rate']:>6.2f}% ({metrics_dict['gpt4mini']['partial_successful']:>4}/{metrics_dict['gpt4mini']['total_conversations']:<4})")
    
    print("\n" + "-" * 100)
    print("Turn-Level Metrics:")
    print("-" * 100)
    
    for model_name, metrics in metrics_dict.items():
        display_name = model_name.replace("gpt4", "GPT-4.1").replace("claude", "Claude 4.5").replace("mini", "Mini").title()
        turn_metrics = metrics["turn_level_metrics"]
        print(f"\n{display_name}:")
        print(f"  Total Turns: {turn_metrics['total_turns']}")
        print(f"  Tool Success Rate: {turn_metrics['tool_success_rate']:.2f}%")
        print(f"  Response Success Rate: {turn_metrics['response_success_rate']:.2f}%")
        print(f"  Final Turn Tool Rate: {turn_metrics['final_turn_tool_rate']:.2f}%")
        print(f"  Final Turn Response Rate: {turn_metrics['final_turn_response_rate']:.2f}%")
    
    print("\n" + "=" * 100)
    print("\nINTERPRETATION:")
    print("-" * 100)
    print("""
1. STRICT (Current): Tests raw capability - all turns must succeed (tool + response).
   This is intentionally rigorous to isolate model performance gaps.

2. TOOL-ONLY: Focuses on tool execution success, ignoring response quality.
   Shows models can execute tools, but response quality needs work.

3. GOAL ACHIEVEMENT: Final turn success - did they complete the task?
   More aligned with production metrics (did the user get what they needed?).

4. PARTIAL SUCCESS: ≥80% of turns succeed - allows for minor errors.
   More forgiving metric that reflects real-world tolerance.

The gap between tool-only and strict metrics shows where response quality
is the bottleneck. The gap between goal achievement and strict shows the
impact of all-or-nothing evaluation.
    """)


def main() -> int:
    eval_dir = Path(_REPO_ROOT) / "artifacts" / "evaluation"
    
    claude_file = eval_dir / "baseline_claude_sonnet_4_5.jsonl"
    gpt4_file = eval_dir / "baseline_gpt4_1.jsonl"
    gpt4mini_file = eval_dir / "baseline_gpt4_1_mini.jsonl"
    
    print("Loading baseline results...")
    claude_results = load_jsonl(claude_file)
    gpt4_results = load_jsonl(gpt4_file)
    gpt4mini_results = load_jsonl(gpt4mini_file)
    
    print(f"Claude: {len(claude_results)} conversations")
    print(f"GPT-4.1: {len(gpt4_results)} conversations")
    print(f"GPT-4.1 Mini: {len(gpt4mini_results)} conversations")
    
    print("\nCalculating alternative metrics...")
    metrics = {
        "claude": calculate_alternative_metrics(claude_results),
        "gpt4": calculate_alternative_metrics(gpt4_results),
        "gpt4mini": calculate_alternative_metrics(gpt4mini_results),
    }
    
    print_metrics_table(metrics)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

