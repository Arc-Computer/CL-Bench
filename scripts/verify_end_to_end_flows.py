#!/usr/bin/env python3
"""Comprehensive end-to-end verification of Atlas learning loop and evaluation flow."""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Optional

_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))


async def verify_atlas_learning_loop() -> Dict[str, Any]:
    """Verify Atlas learning loop: Student > Teacher > Reward > Learning Synthesizer > Re-injection."""
    print("=" * 80)
    print("ATLAS LEARNING LOOP VERIFICATION")
    print("=" * 80)
    print()
    
    results = {
        "student_execution": False,
        "teacher_supervision": False,
        "reward_system": False,
        "learning_synthesis": False,
        "learning_reinjection": False,
        "issues": [],
        "details": {}
    }
    
    # Check database for learning state
    try:
        import asyncpg
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        db_url = os.getenv("STORAGE__DATABASE_URL")
        
        if not db_url:
            results["issues"].append("STORAGE__DATABASE_URL not set")
            return results
        
        # Parse database URL
        # Format: postgresql://user:pass@host:port/dbname
        import re
        match = re.match(r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', db_url)
        if not match:
            results["issues"].append(f"Invalid database URL format: {db_url[:50]}...")
            return results
        
        user, password, host, port, dbname = match.groups()
        
        # Connect to database
        conn = await asyncpg.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            database=dbname
        )
        
        try:
            # 1. Check learning registry (learning synthesis)
            registry_query = """
                SELECT learning_key, 
                       LENGTH(student_learning) as student_len,
                       LENGTH(teacher_learning) as teacher_len,
                       updated_at
                FROM learning_registry
                ORDER BY updated_at DESC
                LIMIT 5
            """
            registry_rows = await conn.fetch(registry_query)
            
            if registry_rows:
                results["learning_synthesis"] = True
                latest_registry = registry_rows[0]
                results["details"]["learning_registry"] = {
                    "learning_key": latest_registry["learning_key"][:20] + "...",
                    "student_learning_chars": latest_registry["student_len"] or 0,
                    "teacher_learning_chars": latest_registry["teacher_len"] or 0,
                    "updated_at": str(latest_registry["updated_at"])
                }
                print("✅ Learning Synthesis: WORKING")
                print(f"   Learning Key: {latest_registry['learning_key'][:30]}...")
                print(f"   Student Learning: {latest_registry['student_len'] or 0} chars")
                print(f"   Teacher Learning: {latest_registry['teacher_len'] or 0} chars")
            else:
                results["issues"].append("No learning registry entries found")
                print("❌ Learning Synthesis: NOT WORKING (no registry entries)")
            
            print()
            
            # 2. Check sessions for learning state (re-injection)
            sessions_query = """
                SELECT id, created_at, 
                       metadata->'learning_state' as learning_state,
                       metadata->'applied_student_learning' as applied_student,
                       metadata->'applied_teacher_learning' as applied_teacher,
                       metadata->>'learning_key' as learning_key
                FROM sessions
                WHERE metadata->>'source' = 'crm-benchmark'
                ORDER BY created_at DESC
                LIMIT 10
            """
            session_rows = await conn.fetch(sessions_query)
            
            if session_rows:
                sessions_with_learning = sum(1 for s in session_rows if s["learning_state"] is not None)
                sessions_with_applied = sum(1 for s in session_rows if s["applied_student"] is not None)
                
                if sessions_with_learning > 0:
                    results["learning_reinjection"] = True
                    print(f"✅ Learning Re-injection: WORKING ({sessions_with_learning}/{len(session_rows)} sessions)")
                    
                    # Check applied learning
                    if sessions_with_applied > 0:
                        print(f"   Applied Student Learning: {sessions_with_applied}/{len(session_rows)} sessions")
                    else:
                        results["issues"].append(f"Applied student learning not found in recent sessions")
                else:
                    results["issues"].append("No sessions with learning_state found")
                    print("❌ Learning Re-injection: NOT WORKING (no learning_state in sessions)")
                
                results["details"]["sessions"] = {
                    "total_checked": len(session_rows),
                    "with_learning_state": sessions_with_learning,
                    "with_applied_student": sessions_with_applied
                }
            else:
                results["issues"].append("No sessions found in database")
                print("❌ Learning Re-injection: NOT WORKING (no sessions found)")
            
            print()
            
            # 3. Check rewards (reward system)
            # Reward is JSONB with structure: {"score": 0.45, "raw": {...}, "judges": [...]}
            rewards_query = """
                SELECT id, created_at, reward
                FROM sessions
                WHERE metadata->>'source' = 'crm-benchmark'
                AND reward IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 20
            """
            reward_rows = await conn.fetch(rewards_query)
            
            if reward_rows:
                reward_scores = []
                for row in reward_rows:
                    reward_data = row["reward"]
                    # asyncpg returns JSONB as dict, but check both dict and string
                    if isinstance(reward_data, dict):
                        score = reward_data.get("score")
                        if score is not None:
                            reward_scores.append(float(score))
                    elif isinstance(reward_data, str):
                        try:
                            import json
                            reward_dict = json.loads(reward_data)
                            score = reward_dict.get("score")
                            if score is not None:
                                reward_scores.append(float(score))
                        except:
                            pass
                
                if reward_scores:
                    results["reward_system"] = True
                    print("✅ Reward System: WORKING")
                    print(f"   Sessions with rewards: {len(reward_scores)}")
                    print(f"   Avg Reward: {sum(reward_scores)/len(reward_scores):.3f}")
                    print(f"   Min Reward: {min(reward_scores):.3f}")
                    print(f"   Max Reward: {max(reward_scores):.3f}")
                    print(f"   Recent scores: {[f'{s:.3f}' for s in reward_scores[:5]]}")
                    results["details"]["rewards"] = {
                        "count": len(reward_scores),
                        "avg": sum(reward_scores) / len(reward_scores),
                        "min": min(reward_scores),
                        "max": max(reward_scores),
                        "scores": reward_scores[:10]  # Store first 10 for analysis
                    }
                else:
                    results["issues"].append("Reward JSONB found but no score field")
                    print("⚠️  Reward System: Reward data exists but no scores extracted")
            else:
                results["issues"].append("No rewards found in sessions")
                print("❌ Reward System: NOT WORKING (no rewards found)")
            
            print()
            
            # 4. Check teacher interventions (teacher supervision)
            teacher_query = """
                SELECT COUNT(*) as count
                FROM sessions
                WHERE metadata->>'source' = 'crm-benchmark'
                AND metadata->'teacher_interventions' IS NOT NULL
            """
            teacher_row = await conn.fetchrow(teacher_query)
            
            if teacher_row and teacher_row["count"] > 0:
                results["teacher_supervision"] = True
                print(f"✅ Teacher Supervision: WORKING ({teacher_row['count']} sessions)")
                results["details"]["teacher"] = {"sessions_with_interventions": teacher_row["count"]}
            else:
                # Teacher supervision might be working but not tracked in metadata
                print("⚠️  Teacher Supervision: Cannot verify (not tracked in metadata)")
                results["details"]["teacher"] = {"note": "Not tracked in metadata"}
            
            print()
            
            # 5. Check student execution (sessions exist)
            if session_rows:
                results["student_execution"] = True
                print(f"✅ Student Execution: WORKING ({len(session_rows)} sessions)")
            else:
                results["issues"].append("No student execution sessions found")
                print("❌ Student Execution: NOT WORKING (no sessions)")
            
        finally:
            await conn.close()
            
    except ImportError:
        results["issues"].append("asyncpg not available - cannot check database")
        print("⚠️  Cannot verify database (asyncpg not available)")
    except Exception as exc:
        results["issues"].append(f"Database check failed: {exc}")
        print(f"❌ Database check failed: {exc}")
    
    print()
    return results


def verify_evaluation_flow() -> Dict[str, Any]:
    """Verify evaluation flow: Eval > Task Completion > Judge Verification."""
    print("=" * 80)
    print("EVALUATION FLOW VERIFICATION")
    print("=" * 80)
    print()
    
    results = {
        "task_execution": False,
        "task_completion": False,
        "judge_verification": False,
        "exact_match": False,
        "issues": [],
        "details": {}
    }
    
    # Check evaluation results
    sessions_file = Path("artifacts/evaluation/atlas_400_results/atlas/sessions.jsonl")
    
    if not sessions_file.exists():
        results["issues"].append("No evaluation sessions file found")
        print("❌ Task Execution: NOT WORKING (no sessions file)")
        return results
    
    sessions = []
    with sessions_file.open() as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))
    
    if not sessions:
        results["issues"].append("No sessions in evaluation file")
        print("❌ Task Execution: NOT WORKING (no sessions)")
        return results
    
    results["task_execution"] = True
    print(f"✅ Task Execution: WORKING ({len(sessions)} sessions)")
    print()
    
    # Analyze evaluation flow
    total_turns = 0
    exact_matches = 0
    judge_evaluations = 0
    judge_passes = 0
    tool_successes = 0
    response_successes = 0
    
    verification_methods = Counter()
    judge_usage = Counter()
    
    for session in sessions:
        result = session.get("final_payload", {}).get("conversation_result", {})
        per_turn = result.get("per_turn_results", [])
        
        for turn in per_turn:
            total_turns += 1
            
            verification = turn.get("verification", "unknown")
            verification_methods[verification] += 1
            
            tool_success = turn.get("tool_success", False)
            response_success = turn.get("response_success", False)
            
            if tool_success:
                tool_successes += 1
            if response_success:
                response_successes += 1
            
            if verification == "exact_match":
                exact_matches += 1
                results["exact_match"] = True
            
            judge_used = turn.get("judge_used", False)
            judge_usage[judge_used] += 1
            
            if judge_used:
                judge_evaluations += 1
                results["judge_verification"] = True
                if turn.get("judge_pass", False):
                    judge_passes += 1
    
    # Task completion
    conversations_with_tool_success = sum(
        1 for s in sessions 
        if any(t.get("tool_success", False) for t in s.get("final_payload", {}).get("conversation_result", {}).get("per_turn_results", []))
    )
    
    if conversations_with_tool_success > 0:
        results["task_completion"] = True
        print("✅ Task Completion: WORKING")
        print(f"   Conversations with ≥1 tool success: {conversations_with_tool_success}/{len(sessions)} ({conversations_with_tool_success/len(sessions)*100:.1f}%)")
        print(f"   Turn-level tool success: {tool_successes}/{total_turns} ({tool_successes/total_turns*100:.1f}%)")
    else:
        results["issues"].append("No conversations with tool success")
        print("❌ Task Completion: NOT WORKING (no tool successes)")
    
    print()
    
    # Exact match
    if exact_matches > 0:
        print("✅ Exact Match Verification: WORKING")
        print(f"   Exact matches: {exact_matches}/{total_turns} ({exact_matches/total_turns*100:.1f}%)")
    else:
        results["issues"].append("No exact matches found")
        print("⚠️  Exact Match Verification: No exact matches (may be normal)")
    
    print()
    
    # Judge verification
    if judge_evaluations > 0:
        print("✅ Judge Verification: WORKING")
        print(f"   Judge evaluations: {judge_evaluations}/{total_turns} ({judge_evaluations/total_turns*100:.1f}%)")
        print(f"   Judge passes: {judge_passes}/{judge_evaluations} ({judge_passes/judge_evaluations*100:.1f}%)")
        print()
        print("   Verification Methods:")
        for method, count in verification_methods.most_common():
            print(f"     {method}: {count} ({count/total_turns*100:.1f}%)")
    else:
        results["issues"].append("No judge evaluations found")
        print("❌ Judge Verification: NOT WORKING (no judge evaluations)")
    
    results["details"] = {
        "total_sessions": len(sessions),
        "total_turns": total_turns,
        "exact_matches": exact_matches,
        "judge_evaluations": judge_evaluations,
        "judge_passes": judge_passes,
        "tool_successes": tool_successes,
        "response_successes": response_successes,
        "verification_methods": dict(verification_methods),
        "judge_usage": dict(judge_usage)
    }
    
    print()
    return results


async def main():
    """Run all verifications."""
    print("\n" + "=" * 80)
    print("END-TO-END FLOW VERIFICATION")
    print("=" * 80)
    print()
    
    # Verify Atlas learning loop
    atlas_results = await verify_atlas_learning_loop()
    
    print("\n" + "=" * 80 + "\n")
    
    # Verify evaluation flow
    eval_results = verify_evaluation_flow()
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print()
    
    print("ATLAS LEARNING LOOP:")
    print(f"  Student Execution: {'✅' if atlas_results['student_execution'] else '❌'}")
    print(f"  Teacher Supervision: {'✅' if atlas_results['teacher_supervision'] else '⚠️'}")
    print(f"  Reward System: {'✅' if atlas_results['reward_system'] else '❌'}")
    print(f"  Learning Synthesis: {'✅' if atlas_results['learning_synthesis'] else '❌'}")
    print(f"  Learning Re-injection: {'✅' if atlas_results['learning_reinjection'] else '❌'}")
    
    print()
    print("EVALUATION FLOW:")
    print(f"  Task Execution: {'✅' if eval_results['task_execution'] else '❌'}")
    print(f"  Task Completion: {'✅' if eval_results['task_completion'] else '❌'}")
    print(f"  Exact Match: {'✅' if eval_results['exact_match'] else '⚠️'}")
    print(f"  Judge Verification: {'✅' if eval_results['judge_verification'] else '❌'}")
    
    print()
    
    # Overall status
    atlas_ok = all([
        atlas_results['student_execution'],
        atlas_results['reward_system'],
        atlas_results['learning_synthesis'],
        atlas_results['learning_reinjection']
    ])
    
    eval_ok = all([
        eval_results['task_execution'],
        eval_results['task_completion'],
        eval_results['judge_verification']
    ])
    
    if atlas_ok and eval_ok:
        print("✅ OVERALL STATUS: ALL SYSTEMS OPERATIONAL")
    elif atlas_ok:
        print("⚠️  OVERALL STATUS: Atlas OK, Evaluation needs attention")
    elif eval_ok:
        print("⚠️  OVERALL STATUS: Evaluation OK, Atlas needs attention")
    else:
        print("❌ OVERALL STATUS: BOTH SYSTEMS NEED ATTENTION")
    
    # Issues
    all_issues = atlas_results['issues'] + eval_results['issues']
    if all_issues:
        print()
        print("ISSUES FOUND:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
    
    print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

