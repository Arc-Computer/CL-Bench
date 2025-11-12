#!/usr/bin/env python3
"""Estimate Atlas token costs from sample sessions.

Extracts token usage from a sample of sessions and extrapolates to total cost.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List


def extract_reward_tokens(session: Dict[str, Any]) -> Dict[str, int]:
    """Extract reward token usage from session_reward_audit."""
    tokens = defaultdict(int)
    atlas_meta = session.get("atlas_metadata", {})
    session_meta = atlas_meta.get("session_metadata", {})
    reward_audit = session_meta.get("reward_audit", [])
    
    for entry in reward_audit:
        raw_response = entry.get("raw_response", {})
        usage = raw_response.get("usage", {})
        if isinstance(usage, dict):
            def safe_int(value):
                try:
                    return int(value) if value is not None else 0
                except (TypeError, ValueError):
                    return 0
            
            tokens["prompt_tokens"] += safe_int(usage.get("prompt_tokens"))
            tokens["completion_tokens"] += safe_int(usage.get("completion_tokens"))
            tokens["total_tokens"] += safe_int(usage.get("total_tokens"))
    
    return dict(tokens)


def calculate_cost(tokens: Dict[str, int], model: str) -> Dict[str, float]:
    """Calculate cost based on token usage and model pricing."""
    pricing = {
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "claude-4.5-sonnet": {"input": 3.00, "output": 15.00},
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    }
    
    model_pricing = pricing.get(model.lower(), {"input": 0.0, "output": 0.0})
    prompt_tokens = tokens.get("prompt_tokens", 0)
    completion_tokens = tokens.get("completion_tokens", 0)
    
    input_cost = (prompt_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * model_pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": tokens.get("total_tokens", prompt_tokens + completion_tokens),
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    }


def estimate_costs(
    sessions_path: Path,
    sample_size: int = 20,
    total_conversations: int = 200,
) -> Dict[str, Any]:
    """Estimate token costs from sample sessions."""
    sessions: List[Dict[str, Any]] = []
    
    with open(sessions_path) as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            if line.strip():
                try:
                    session = json.loads(line)
                    sessions.append(session)
                except json.JSONDecodeError:
                    continue
    
    if not sessions:
        return {
            "error": f"No valid sessions found in {sessions_path}",
            "sample_size": 0,
        }
    
    # Extract reward tokens
    reward_tokens = defaultdict(int)
    sessions_with_reward = 0
    
    for session in sessions:
        tokens = extract_reward_tokens(session)
        if any(tokens.values()):
            sessions_with_reward += 1
            reward_tokens["prompt_tokens"] += tokens.get("prompt_tokens", 0)
            reward_tokens["completion_tokens"] += tokens.get("completion_tokens", 0)
            reward_tokens["total_tokens"] += tokens.get("total_tokens", 0)
    
    # Calculate averages
    by_component: Dict[str, Any] = {}
    
    if sessions_with_reward > 0:
        avg_reward = {
            "prompt_tokens": reward_tokens["prompt_tokens"] / sessions_with_reward,
            "completion_tokens": reward_tokens["completion_tokens"] / sessions_with_reward,
            "total_tokens": reward_tokens["total_tokens"] / sessions_with_reward,
        }
        
        # Extrapolate to total conversations
        total_reward = {
            "prompt_tokens": int(avg_reward["prompt_tokens"] * total_conversations),
            "completion_tokens": int(avg_reward["completion_tokens"] * total_conversations),
            "total_tokens": int(avg_reward["total_tokens"] * total_conversations),
        }
        
        # Calculate cost (reward uses GPT-4.1 mini)
        cost = calculate_cost(total_reward, "gpt-4.1-mini")
        
        by_component["reward"] = {
            "model": "gpt-4.1-mini",
            "sessions_with_data": sessions_with_reward,
            "avg_tokens_per_conv": avg_reward,
            "total_tokens_estimated": total_reward,
            "cost_usd": cost,
        }
    
    # Note missing components
    by_component["student"] = {
        "model": "gpt-4.1-mini",
        "status": "not_tracked",
        "note": "Student token usage not available in existing sessions",
    }
    
    by_component["judge"] = {
        "model": "gpt-4.1",
        "status": "not_tracked",
        "note": "Judge token usage not available in existing sessions",
    }
    
    by_component["learning"] = {
        "model": "gemini-2.5-flash",
        "status": "not_tracked",
        "note": "Learning synthesis token usage not available in existing sessions",
    }
    
    # Calculate total cost (only reward available)
    total_cost_usd = by_component.get("reward", {}).get("cost_usd", {}).get("total_cost_usd", 0.0)
    cost_per_conversation = total_cost_usd / total_conversations if total_conversations > 0 else 0.0
    
    return {
        "sample_size": len(sessions),
        "sessions_with_reward_data": sessions_with_reward,
        "total_conversations": total_conversations,
        "by_component": by_component,
        "total_cost_usd": total_cost_usd,
        "cost_per_conversation": cost_per_conversation,
        "methodology": "Sample-based estimation: Extracted token usage from first N sessions, calculated averages, and extrapolated to total conversations.",
        "limitations": [
            "Only reward evaluation tokens are available in existing sessions",
            "Student, judge, and learning synthesis tokens were not tracked",
            "Cost estimate is partial and represents only reward evaluation overhead",
            "Future evaluations will include complete token tracking for all components",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate Atlas token costs from sample sessions"
    )
    parser.add_argument(
        "sessions_path",
        type=Path,
        help="Path to sessions.jsonl file",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of sessions to sample (default: 20)",
    )
    parser.add_argument(
        "--total-conversations",
        type=int,
        default=200,
        help="Total number of conversations to extrapolate to (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file path (default: print to stdout)",
    )
    
    args = parser.parse_args()
    
    if not args.sessions_path.exists():
        print(f"Error: Sessions file not found: {args.sessions_path}", file=sys.stderr)
        return 1
    
    result = estimate_costs(
        args.sessions_path,
        sample_size=args.sample_size,
        total_conversations=args.total_conversations,
    )
    
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        return 1
    
    output_json = json.dumps(result, indent=2)
    
    if args.output:
        args.output.write_text(output_json)
        print(f"✅ Cost estimates written to {args.output}")
    else:
        print(output_json)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

