#!/usr/bin/env python3
"""Monitor Atlas evaluation to verify placeholder resolution and judge fixes."""

import json
import sys
from pathlib import Path
from collections import Counter

_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))


def monitor_fixes():
    """Monitor placeholder resolution and judge evaluation."""
    sessions_file = Path("artifacts/evaluation/atlas_400_results/atlas/sessions.jsonl")
    
    if not sessions_file.exists():
        print("⚠️  No sessions file found yet")
        return
    
    sessions = []
    with sessions_file.open() as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))
    
    if not sessions:
        print("⚠️  No sessions completed yet")
        return
    
    print("=" * 80)
    print("ATLAS FIX VERIFICATION MONITOR")
    print(f"Sessions Analyzed: {len(sessions)}")
    print("=" * 80)
    print()
    
    # Analyze placeholder resolution
    print("1. PLACEHOLDER RESOLUTION ANALYSIS")
    print("-" * 80)
    
    placeholder_issues = 0
    resolved_correctly = 0
    verification_methods = Counter()
    judge_usage = Counter()
    
    for session in sessions:
        result = session.get("final_payload", {}).get("conversation_result", {})
        per_turn = result.get("per_turn_results", [])
        
        for turn in per_turn:
            verification = turn.get("verification", "unknown")
            verification_methods[verification] += 1
            
            judge_used = turn.get("judge_used", False)
            judge_usage[judge_used] += 1
            
            # Check for placeholder issues
            original_args = turn.get("original_agent_arguments", {})
            resolved_args = turn.get("arguments", {})
            
            if original_args:
                original_str = json.dumps(original_args)
                resolved_str = json.dumps(resolved_args)
                
                # Check if placeholders were resolved
                if "{{" in original_str or "{" in original_str:
                    if "{{" not in resolved_str and "{" not in resolved_str:
                        resolved_correctly += 1
                    else:
                        placeholder_issues += 1
    
    total_turns = sum(len(s.get("final_payload", {}).get("conversation_result", {}).get("per_turn_results", [])) 
                     for s in sessions)
    
    print(f"Total Turns Analyzed: {total_turns}")
    print(f"Placeholders Resolved Correctly: {resolved_correctly}")
    print(f"Placeholder Issues Remaining: {placeholder_issues}")
    print()
    
    # Verification methods
    print("Verification Methods:")
    for method, count in verification_methods.most_common():
        pct = count / total_turns * 100 if total_turns > 0 else 0
        print(f"  {method}: {count} ({pct:.1f}%)")
    print()
    
    # Judge usage
    print("Judge Usage:")
    judge_count = judge_usage.get(True, 0)
    no_judge_count = judge_usage.get(False, 0)
    print(f"  Judge Used: {judge_count} ({judge_count/total_turns*100:.1f}%)")
    print(f"  Judge Not Used: {no_judge_count} ({no_judge_count/total_turns*100:.1f}%)")
    print()
    
    # Success rates
    print("2. SUCCESS RATES")
    print("-" * 80)
    
    tool_success = 0
    response_success = 0
    overall_success = 0
    
    for session in sessions:
        result = session.get("final_payload", {}).get("conversation_result", {})
        per_turn = result.get("per_turn_results", [])
        
        has_tool_success = any(t.get("tool_success", False) for t in per_turn)
        has_response_success = any(t.get("response_success", False) for t in per_turn)
        
        if has_tool_success:
            tool_success += 1
        if has_response_success:
            response_success += 1
        if result.get("overall_success", False):
            overall_success += 1
    
    print(f"Conversations w/ ≥1 Tool Success: {tool_success}/{len(sessions)} ({tool_success/len(sessions)*100:.1f}%)")
    print(f"Conversations w/ ≥1 Response Success: {response_success}/{len(sessions)} ({response_success/len(sessions)*100:.1f}%)")
    print(f"Overall Success (All Turns): {overall_success}/{len(sessions)} ({overall_success/len(sessions)*100:.1f}%)")
    print()
    
    # Turn-level success
    turn_tool_success = sum(sum(1 for t in s.get("final_payload", {}).get("conversation_result", {}).get("per_turn_results", []) 
                                if t.get("tool_success", False)) 
                           for s in sessions)
    turn_response_success = sum(sum(1 for t in s.get("final_payload", {}).get("conversation_result", {}).get("per_turn_results", []) 
                                   if t.get("response_success", False)) 
                              for s in sessions)
    
    print(f"Turn-Level Tool Success: {turn_tool_success}/{total_turns} ({turn_tool_success/total_turns*100:.1f}%)")
    print(f"Turn-Level Response Success: {turn_response_success}/{total_turns} ({turn_response_success/total_turns*100:.1f}%)")
    print()
    
    # Recent session details
    print("3. RECENT SESSION DETAILS")
    print("-" * 80)
    
    for i, session in enumerate(sessions[-3:], start=len(sessions)-2):
        conv_id = session.get("conversation_id", "N/A")
        result = session.get("final_payload", {}).get("conversation_result", {})
        per_turn = result.get("per_turn_results", [])
        
        print(f"Session {i}: {conv_id[:50]}...")
        if per_turn:
            turn = per_turn[0]
            print(f"  Turn 1: {turn.get('tool_name', 'N/A')}")
            print(f"    Tool Success: {turn.get('tool_success', False)}")
            print(f"    Verification: {turn.get('verification', 'N/A')}")
            print(f"    Judge Used: {turn.get('judge_used', False)}")
            if turn.get('error'):
                print(f"    Error: {turn.get('error')[:80]}...")
        print()


if __name__ == "__main__":
    monitor_fixes()

