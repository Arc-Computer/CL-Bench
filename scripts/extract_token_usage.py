#!/usr/bin/env python3
"""Extract and aggregate token usage from Atlas evaluation sessions."""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List


def extract_token_usage_from_reward_audit(session: Dict[str, Any]) -> Dict[str, int]:
    """Extract token usage from reward audit (teacher LLM calls)."""
    tokens = defaultdict(int)
    
    # Check session_reward_audit
    reward_audit = session.get("atlas_metadata", {}).get("session_reward_audit", [])
    if not reward_audit:
        # Also check session_metadata.reward_audit
        reward_audit = (
            session.get("atlas_metadata", {})
            .get("session_metadata", {})
            .get("reward_audit", [])
        )
    
    for audit_entry in reward_audit:
        if isinstance(audit_entry, dict):
            raw_response = audit_entry.get("raw_response", {})
            usage = raw_response.get("usage", {})
            if isinstance(usage, dict):
                # Handle both int and string values
                def safe_int(value):
                    try:
                        return int(value) if value is not None else 0
                    except (TypeError, ValueError):
                        return 0
                
                tokens["prompt_tokens"] += safe_int(usage.get("prompt_tokens"))
                tokens["completion_tokens"] += safe_int(usage.get("completion_tokens"))
                tokens["total_tokens"] += safe_int(usage.get("total_tokens"))
    
    return dict(tokens)


def extract_token_usage_from_conversation_result(session: Dict[str, Any]) -> Dict[str, int]:
    """Extract token usage from conversation_result metadata (student agent calls)."""
    tokens = defaultdict(int)
    
    # First check if token_usage is already in atlas_metadata (new format)
    atlas_metadata = session.get("atlas_metadata", {})
    token_usage_metadata = atlas_metadata.get("token_usage", {})
    if isinstance(token_usage_metadata, dict):
        student_usage = token_usage_metadata.get("student", {})
        if isinstance(student_usage, dict):
            tokens["prompt_tokens"] += student_usage.get("prompt_tokens", 0)
            tokens["completion_tokens"] += student_usage.get("completion_tokens", 0)
            tokens["total_tokens"] += student_usage.get("total_tokens", 0)
            return dict(tokens)  # Return early if found in new format
    
    # Fallback to old format: check conversation_result.metadata.agent.token_usage
    conv_result = session.get("conversation_result", {})
    if not conv_result:
        return dict(tokens)
    
    # Check metadata.agent.token_usage (aggregated from per-turn)
    metadata = conv_result.get("metadata", {})
    agent_metadata = metadata.get("agent", {})
    agent_token_usage = agent_metadata.get("token_usage", {})
    
    if isinstance(agent_token_usage, dict):
        tokens["prompt_tokens"] += agent_token_usage.get("prompt_tokens", 0)
        tokens["completion_tokens"] += agent_token_usage.get("completion_tokens", 0)
        tokens["total_tokens"] += agent_token_usage.get("total_tokens", 0)
    
    # Also check per-turn token_usage
    per_turn = conv_result.get("per_turn_results", [])
    for turn in per_turn:
        turn_tokens = turn.get("token_usage", {})
        if isinstance(turn_tokens, dict):
            # Handle nested structure (e.g., {"judge": {...}, "judge_response": {...}})
            for key, value in turn_tokens.items():
                if isinstance(value, dict):
                    tokens["prompt_tokens"] += value.get("prompt_tokens", 0)
                    tokens["completion_tokens"] += value.get("completion_tokens", 0)
                    tokens["total_tokens"] += value.get("total_tokens", 0)
                elif key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    tokens[key] += int(value) if isinstance(value, (int, float)) else 0
    
    return dict(tokens)


def extract_token_usage_from_learning(session: Dict[str, Any]) -> Dict[str, int]:
    """Extract token usage from learning synthesis (learning LLM calls)."""
    tokens = defaultdict(int)
    
    # First check if token_usage is in atlas_metadata.token_usage.learning (new format)
    atlas_metadata = session.get("atlas_metadata", {})
    token_usage_metadata = atlas_metadata.get("token_usage", {})
    if isinstance(token_usage_metadata, dict):
        learning_usage = token_usage_metadata.get("learning", {})
        if isinstance(learning_usage, dict):
            tokens["prompt_tokens"] += learning_usage.get("prompt_tokens", 0)
            tokens["completion_tokens"] += learning_usage.get("completion_tokens", 0)
            tokens["total_tokens"] += learning_usage.get("total_tokens", 0)
            return dict(tokens)  # Return early if found in new format
    
    # Fallback to old format: check learning_usage.session.token_usage
    learning_usage = (
        session.get("atlas_metadata", {})
        .get("learning_usage", {})
        .get("session", {})
        .get("token_usage", {})
    )
    
    if isinstance(learning_usage, dict):
        tokens["prompt_tokens"] += learning_usage.get("prompt_tokens", 0)
        tokens["completion_tokens"] += learning_usage.get("completion_tokens", 0)
        tokens["total_tokens"] += learning_usage.get("total_tokens", 0)
    
    return dict(tokens)


def extract_all_token_usage(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract and aggregate token usage from all sessions."""
    totals = defaultdict(int)
    by_role = {
        "student": defaultdict(int),
        "teacher": defaultdict(int),
        "learning": defaultdict(int),
        "judge": defaultdict(int),
    }
    
    sessions_with_data = {
        "student": 0,
        "teacher": 0,
        "learning": 0,
        "judge": 0,
    }
    
    for session in sessions:
        # Student agent token usage (from conversation_result)
        student_tokens = extract_token_usage_from_conversation_result(session)
        if any(student_tokens.values()):
            sessions_with_data["student"] += 1
            for key, value in student_tokens.items():
                by_role["student"][key] += value
                totals[key] += value
        
        # Teacher/reward token usage (from reward audit)
        teacher_tokens = extract_token_usage_from_reward_audit(session)
        if any(teacher_tokens.values()):
            sessions_with_data["teacher"] += 1
            for key, value in teacher_tokens.items():
                by_role["teacher"][key] += value
                totals[key] += value
        
        # Learning synthesis token usage
        learning_tokens = extract_token_usage_from_learning(session)
        if any(learning_tokens.values()):
            sessions_with_data["learning"] += 1
            for key, value in learning_tokens.items():
                by_role["learning"][key] += value
                totals[key] += value
        
        # Judge token usage (from atlas_metadata.token_usage.judge or per-turn records)
        atlas_metadata = session.get("atlas_metadata", {})
        token_usage_metadata = atlas_metadata.get("token_usage", {})
        judge_tokens = defaultdict(int)
        
        # Check new format first
        if isinstance(token_usage_metadata, dict):
            judge_usage = token_usage_metadata.get("judge", {})
            if isinstance(judge_usage, dict):
                judge_tokens["prompt_tokens"] += judge_usage.get("prompt_tokens", 0)
                judge_tokens["completion_tokens"] += judge_usage.get("completion_tokens", 0)
                judge_tokens["total_tokens"] += judge_usage.get("total_tokens", 0)
        
        # Fallback to per-turn extraction
        if not any(judge_tokens.values()):
            conv_result = session.get("conversation_result")
            if conv_result is None:
                conv_result = {}
            per_turn = conv_result.get("per_turn_results", [])
            for turn in per_turn:
                turn_tokens = turn.get("token_usage", {})
                if isinstance(turn_tokens, dict):
                    # Extract judge and judge_response token usage
                    for judge_key in ("judge", "judge_response"):
                        judge_usage = turn_tokens.get(judge_key, {})
                        if isinstance(judge_usage, dict):
                            judge_tokens["prompt_tokens"] += judge_usage.get("prompt_tokens", 0)
                            judge_tokens["completion_tokens"] += judge_usage.get("completion_tokens", 0)
                            judge_tokens["total_tokens"] += judge_usage.get("total_tokens", 0)
        
        if any(judge_tokens.values()):
            sessions_with_data["judge"] += 1
            for key, value in judge_tokens.items():
                by_role["judge"][key] += value
                totals[key] += value
    
    return {
        "totals": dict(totals),
        "by_role": {role: dict(tokens) for role, tokens in by_role.items()},
        "sessions_with_data": sessions_with_data,
    }


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: extract_token_usage.py <sessions.jsonl>")
        sys.exit(1)
    
    sessions_path = Path(sys.argv[1])
    if not sessions_path.exists():
        print(f"Error: File not found: {sessions_path}")
        sys.exit(1)
    
    # Load sessions
    sessions = []
    with open(sessions_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    sessions.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}", file=sys.stderr)
    
    if not sessions:
        print("No sessions found in file.")
        sys.exit(1)
    
    # Extract token usage
    usage_data = extract_all_token_usage(sessions)
    
    # Print results
    print("=" * 70)
    print("TOKEN USAGE EXTRACTION")
    print("=" * 70)
    print()
    print(f"Total Sessions: {len(sessions)}")
    print()
    
    print("Sessions with Token Data:")
    for role, count in usage_data["sessions_with_data"].items():
        pct = (count / len(sessions) * 100) if sessions else 0
        print(f"   {role.title()}: {count}/{len(sessions)} ({pct:.1f}%)")
    print()
    
    print("Total Token Usage:")
    totals = usage_data["totals"]
    if totals:
        for key in sorted(totals.keys()):
            print(f"   {key}: {totals[key]:,}")
        print(f"   Grand Total: {sum(totals.values()):,} tokens")
    else:
        print("   ⚠️  No token usage data found")
    print()
    
    print("Token Usage by Role:")
    for role, tokens in usage_data["by_role"].items():
        if any(tokens.values()):
            print(f"   {role.title()}:")
            for key in sorted(tokens.keys()):
                print(f"      {key}: {tokens[key]:,}")
            role_total = sum(tokens.values())
            print(f"      Total: {role_total:,} tokens")
    
    # Output JSON
    output_path = sessions_path.parent / "token_usage.json"
    with open(output_path, "w") as f:
        json.dump(usage_data, f, indent=2)
    print()
    print(f"✅ Token usage data written to: {output_path}")


if __name__ == "__main__":
    main()

