#!/usr/bin/env python3
"""Monitor evaluation progress and extract token costs when 20 new sessions are captured."""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict


def count_sessions_with_tokens(sessions_path: Path, start_count: int = 0) -> tuple[int, int]:
    """Count total sessions and sessions with token tracking."""
    if not sessions_path.exists():
        return 0, 0
    
    total = 0
    with_tokens = 0
    
    with open(sessions_path) as f:
        for line in f:
            if line.strip():
                total += 1
                if total > start_count:  # Only count new sessions
                    try:
                        session = json.loads(line)
                        atlas_meta = session.get("atlas_metadata", {})
                        token_usage = atlas_meta.get("token_usage", {})
                        
                        # Check if student or judge tokens are present
                        if token_usage:
                            student = token_usage.get("student", {})
                            judge = token_usage.get("judge", {})
                            if (student and any(student.values())) or (judge and any(judge.values())):
                                with_tokens += 1
                    except Exception:
                        pass
    
    return total, with_tokens


def extract_token_costs_from_new_sessions(sessions_path: Path, start_count: int, sample_size: int = 20) -> dict:
    """Extract token costs from new sessions."""
    sessions = []
    
    with open(sessions_path) as f:
        for i, line in enumerate(f):
            if i < start_count:  # Skip old sessions
                continue
            if line.strip():
                try:
                    session = json.loads(line)
                    sessions.append(session)
                    if len(sessions) >= sample_size:
                        break
                except Exception:
                    pass
    
    if not sessions:
        return {"error": "No new sessions found"}
    
    # Extract tokens
    student_tokens = defaultdict(int)
    judge_tokens = defaultdict(int)
    reward_tokens = defaultdict(int)
    learning_tokens = defaultdict(int)
    
    sessions_with_student = 0
    sessions_with_judge = 0
    sessions_with_reward = 0
    sessions_with_learning = 0
    
    for session in sessions:
        atlas_meta = session.get("atlas_metadata", {})
        token_usage = atlas_meta.get("token_usage", {})
        
        # Student tokens
        student = token_usage.get("student", {})
        if student and any(student.values()):
            sessions_with_student += 1
            student_tokens["prompt_tokens"] += student.get("prompt_tokens", 0)
            student_tokens["completion_tokens"] += student.get("completion_tokens", 0)
            student_tokens["total_tokens"] += student.get("total_tokens", 0)
        
        # Judge tokens
        judge = token_usage.get("judge", {})
        if judge and any(judge.values()):
            sessions_with_judge += 1
            judge_tokens["prompt_tokens"] += judge.get("prompt_tokens", 0)
            judge_tokens["completion_tokens"] += judge.get("completion_tokens", 0)
            judge_tokens["total_tokens"] += judge.get("total_tokens", 0)
        
        # Learning tokens
        learning = token_usage.get("learning", {})
        if learning and any(learning.values()):
            sessions_with_learning += 1
            learning_tokens["prompt_tokens"] += learning.get("prompt_tokens", 0)
            learning_tokens["completion_tokens"] += learning.get("completion_tokens", 0)
            learning_tokens["total_tokens"] += learning.get("total_tokens", 0)
        
        # Reward tokens (from reward_audit)
        session_meta = atlas_meta.get("session_metadata", {})
        reward_audit = session_meta.get("reward_audit", [])
        for entry in reward_audit:
            raw_response = entry.get("raw_response", {})
            usage = raw_response.get("usage", {})
            if isinstance(usage, dict):
                sessions_with_reward += 1
                reward_tokens["prompt_tokens"] += usage.get("prompt_tokens", 0)
                reward_tokens["completion_tokens"] += usage.get("completion_tokens", 0)
                reward_tokens["total_tokens"] += usage.get("total_tokens", 0)
                break  # Count once per session
    
    # Calculate averages and costs
    pricing = {
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    }
    
    def calculate_cost(tokens: dict, model: str) -> dict:
        model_pricing = pricing.get(model.lower(), {"input": 0.0, "output": 0.0})
        prompt = tokens.get("prompt_tokens", 0)
        completion = tokens.get("completion_tokens", 0)
        input_cost = (prompt / 1_000_000) * model_pricing["input"]
        output_cost = (completion / 1_000_000) * model_pricing["output"]
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": tokens.get("total_tokens", prompt + completion),
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": input_cost + output_cost,
        }
    
    result = {
        "sample_size": len(sessions),
        "sessions_with_data": {
            "student": sessions_with_student,
            "judge": sessions_with_judge,
            "reward": sessions_with_reward,
            "learning": sessions_with_learning,
        },
        "by_component": {},
    }
    
    if sessions_with_student > 0:
        avg = {
            "prompt_tokens": student_tokens["prompt_tokens"] / sessions_with_student,
            "completion_tokens": student_tokens["completion_tokens"] / sessions_with_student,
            "total_tokens": student_tokens["total_tokens"] / sessions_with_student,
        }
        result["by_component"]["student"] = {
            "model": "gpt-4.1-mini",
            "avg_tokens_per_conv": avg,
            "total_tokens": dict(student_tokens),
            "cost_usd": calculate_cost(student_tokens, "gpt-4.1-mini"),
        }
    
    if sessions_with_judge > 0:
        avg = {
            "prompt_tokens": judge_tokens["prompt_tokens"] / sessions_with_judge,
            "completion_tokens": judge_tokens["completion_tokens"] / sessions_with_judge,
            "total_tokens": judge_tokens["total_tokens"] / sessions_with_judge,
        }
        result["by_component"]["judge"] = {
            "model": "gpt-4.1",
            "avg_tokens_per_conv": avg,
            "total_tokens": dict(judge_tokens),
            "cost_usd": calculate_cost(judge_tokens, "gpt-4.1"),
        }
    
    if sessions_with_reward > 0:
        avg = {
            "prompt_tokens": reward_tokens["prompt_tokens"] / sessions_with_reward,
            "completion_tokens": reward_tokens["completion_tokens"] / sessions_with_reward,
            "total_tokens": reward_tokens["total_tokens"] / sessions_with_reward,
        }
        result["by_component"]["reward"] = {
            "model": "gpt-4.1-mini",
            "avg_tokens_per_conv": avg,
            "total_tokens": dict(reward_tokens),
            "cost_usd": calculate_cost(reward_tokens, "gpt-4.1-mini"),
        }
    
    if sessions_with_learning > 0:
        avg = {
            "prompt_tokens": learning_tokens["prompt_tokens"] / sessions_with_learning,
            "completion_tokens": learning_tokens["completion_tokens"] / sessions_with_learning,
            "total_tokens": learning_tokens["total_tokens"] / sessions_with_learning,
        }
        result["by_component"]["learning"] = {
            "model": "gemini-2.5-flash",
            "avg_tokens_per_conv": avg,
            "total_tokens": dict(learning_tokens),
            "cost_usd": calculate_cost(learning_tokens, "gemini-2.5-flash"),
        }
    
    # Calculate total cost
    total_cost = sum(
        comp.get("cost_usd", {}).get("total_cost_usd", 0)
        for comp in result["by_component"].values()
    )
    result["total_cost_usd"] = total_cost
    result["cost_per_conversation"] = total_cost / len(sessions) if sessions else 0.0
    
    return result


def main():
    sessions_path = Path("artifacts/evaluation/atlas_400/atlas/sessions.jsonl")
    start_count = 25  # Sessions already completed before restart
    
    print("=" * 70)
    print("MONITORING TOKEN CAPTURE")
    print("=" * 70)
    print(f"Monitoring: {sessions_path}")
    print(f"Starting from session: {start_count}")
    print(f"Target: 20 new sessions with token tracking")
    print()
    
    check_count = 0
    while True:
        check_count += 1
        total, with_tokens = count_sessions_with_tokens(sessions_path, start_count)
        new_sessions = total - start_count
        
        print(f"[Check {check_count}] Total sessions: {total} | New sessions: {new_sessions} | With tokens: {with_tokens}")
        
        if with_tokens >= 20:
            print()
            print("=" * 70)
            print(f"✅ Found {with_tokens} new sessions with token tracking!")
            print("=" * 70)
            print()
            
            # Extract costs
            costs = extract_token_costs_from_new_sessions(sessions_path, start_count, sample_size=20)
            
            # Save results
            output_path = Path("artifacts/evaluation/atlas_400/token_cost_estimate_new.json")
            output_path.write_text(json.dumps(costs, indent=2))
            
            print("Token Cost Analysis:")
            print(json.dumps(costs, indent=2))
            print()
            print(f"✅ Results saved to: {output_path}")
            break
        
        if new_sessions >= 30:  # Safety limit
            print(f"⚠️  Reached {new_sessions} new sessions but only {with_tokens} have tokens")
            print("   This suggests token tracking may not be working correctly")
            break
        
        time.sleep(30)  # Check every 30 seconds


if __name__ == "__main__":
    main()

