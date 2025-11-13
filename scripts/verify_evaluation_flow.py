#!/usr/bin/env python3
"""Verify the evaluation flow is working correctly."""

import json
import sys
from pathlib import Path
from collections import Counter

_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))


def verify_evaluation_flow():
    """Verify evaluation flow logic."""
    sessions_file = Path("artifacts/evaluation/atlas_400_results/atlas/sessions.jsonl")
    
    if not sessions_file.exists():
        print("⚠️  No sessions file found")
        return
    
    sessions = []
    with sessions_file.open() as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))
    
    if not sessions:
        print("⚠️  No sessions found")
        return
    
    print("=" * 80)
    print("EVALUATION FLOW VERIFICATION")
    print(f"Sessions Analyzed: {len(sessions)}")
    print("=" * 80)
    print()
    
    # Analyze evaluation flow
    verification_methods = Counter()
    judge_usage = Counter()
    exact_match_count = 0
    judge_called_count = 0
    judge_evaluated_count = 0
    
    issues = []
    
    for session in sessions:
        result = session.get("final_payload", {}).get("conversation_result", {})
        per_turn = result.get("per_turn_results", [])
        
        for turn in per_turn:
            verification = turn.get("verification", "unknown")
            verification_methods[verification] += 1
            
            # Check exact match logic
            matches_tool = turn.get("matches_expected_tool")
            matches_args = turn.get("matches_expected_arguments")
            execution_success = turn.get("result") is not None
            tool_success = turn.get("tool_success")
            
            # Expected exact match
            expected_exact_match = (
                matches_tool is True 
                and matches_args is True 
                and execution_success is True
            )
            
            # Check if verification matches expected
            if expected_exact_match:
                exact_match_count += 1
                if verification != "exact_match":
                    issues.append({
                        "turn": turn.get("turn_id"),
                        "issue": f"Expected 'exact_match' but got '{verification}'",
                        "matches_tool": matches_tool,
                        "matches_args": matches_args,
                        "execution_success": execution_success,
                    })
            
            # Check judge usage
            judge_used = turn.get("judge_used")
            judge_usage[judge_used] += 1
            
            if verification == "llm_judge":
                judge_called_count += 1
                if judge_used is True:
                    judge_evaluated_count += 1
                elif judge_used is False or judge_used is None:
                    issues.append({
                        "turn": turn.get("turn_id"),
                        "issue": f"Verification is 'llm_judge' but judge_used={judge_used}",
                        "judge_score": turn.get("judge_score"),
                        "judge_pass": turn.get("judge_pass"),
                        "judge_error": turn.get("judge_error"),
                    })
            
            # Check if judge should have been called
            if verification == "failed" and not execution_success:
                # Judge should be called if tool matches or args match
                if matches_tool is True or matches_args is True:
                    if judge_used is not True:
                        issues.append({
                            "turn": turn.get("turn_id"),
                            "issue": "Judge should have been called (tool/args match but execution failed)",
                            "matches_tool": matches_tool,
                            "matches_args": matches_args,
                            "judge_used": judge_used,
                        })
    
    print("1. VERIFICATION METHODS")
    print("-" * 80)
    for method, count in verification_methods.most_common():
        pct = count / sum(verification_methods.values()) * 100
        print(f"  {method}: {count} ({pct:.1f}%)")
    print()
    
    print("2. JUDGE USAGE")
    print("-" * 80)
    for used, count in judge_usage.most_common():
        pct = count / sum(judge_usage.values()) * 100
        print(f"  judge_used={used}: {count} ({pct:.1f}%)")
    print()
    
    print("3. EVALUATION FLOW STATS")
    print("-" * 80)
    print(f"  Exact Matches: {exact_match_count}")
    print(f"  Judge Called (verification='llm_judge'): {judge_called_count}")
    print(f"  Judge Evaluated (judge_used=True): {judge_evaluated_count}")
    print()
    
    print("4. ISSUES FOUND")
    print("-" * 80)
    if issues:
        print(f"  ⚠️  Found {len(issues)} issues:")
        for i, issue in enumerate(issues[:10], 1):  # Show first 10
            print(f"  {i}. Turn {issue.get('turn')}: {issue.get('issue')}")
            for key, value in issue.items():
                if key not in ('turn', 'issue'):
                    print(f"     {key}: {value}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
    else:
        print("  ✅ No issues found - evaluation flow is correct!")
    print()
    
    # Show sample turn for verification
    if sessions:
        print("5. SAMPLE TURN ANALYSIS")
        print("-" * 80)
        result = sessions[0].get("final_payload", {}).get("conversation_result", {})
        turn = result.get("per_turn_results", [])[0] if result.get("per_turn_results") else {}
        
        print(f"  Turn ID: {turn.get('turn_id')}")
        print(f"  Tool: {turn.get('tool_name')}")
        print(f"  Expected Tool: {turn.get('expected_tool')}")
        print(f"  Matches Tool: {turn.get('matches_expected_tool')}")
        print(f"  Matches Args: {turn.get('matches_expected_arguments')}")
        print(f"  Execution Success: {turn.get('result') is not None}")
        print(f"  Verification: {turn.get('verification')}")
        print(f"  Tool Success: {turn.get('tool_success')}")
        print(f"  Judge Used: {turn.get('judge_used')}")
        print(f"  Judge Pass: {turn.get('judge_pass')}")
        print()


if __name__ == "__main__":
    verify_evaluation_flow()

