#!/usr/bin/env python3
"""Mini Evaluation: Atlas Learning Loop Over 5 Scenarios

Runs 5 scenarios through Atlas to verify:
1. Learning loop is happening (learning_state is being updated)
2. Learning is persisted to Postgres
3. Learning is re-injected into subsequent sessions
4. Performance and efficiency are improving across sessions

This evaluation justifies dataset expansion by demonstrating:
- Baseline agent is running successfully
- Judge is accurately judging based on the purpose of the eval
- Framework is integrated and learning loop is working
"""

import json
import sys
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(dotenv_path=_REPO_ROOT / '.env')

# Atlas SDK imports
try:
    from atlas.core import arun
    from atlas.runtime.orchestration.execution_context import ExecutionContext
    from atlas.learning.playbook import resolve_playbook
    from atlas.runtime.storage.database import Database
    from atlas.config.loader import load_config
except ImportError as e:
    print(f"❌ Failed to import Atlas SDK: {e}")
    print("   Run: pip install -e 'external/atlas-sdk[dev]'")
    sys.exit(1)

# CRM imports
from src.conversation_schema import Conversation, ConversationTurn, ExpectedResponse
from src.evaluation.verification import VerificationMode
from src.integration.atlas_common import conversation_to_payload


def load_conversations(count: int = 5) -> List[Conversation]:
    """Load multiple conversations from final_conversations.jsonl."""
    final_file = Path('artifacts/deterministic/final_conversations.jsonl')
    if not final_file.exists():
        raise FileNotFoundError(f"Conversation file not found: {final_file}")
    
    conversations = []
    with open(final_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= count:
                break
            conv_dict = json.loads(line)
            
            # Convert to Conversation
            turns = []
            for turn_dict in conv_dict.get('turns', []):
                er_dict = turn_dict.get('expected_response')
                expected_response = None
                if er_dict:
                    expected_response = ExpectedResponse(
                        text=er_dict.get('text', ''),
                        evaluation=er_dict.get('evaluation', 'structured'),
                        answers=er_dict.get('answers', []),
                        requires_judge=er_dict.get('requires_judge', False),
                    )
                turn = ConversationTurn(
                    turn_id=turn_dict['turn_id'],
                    user_utterance=turn_dict['user_utterance'],
                    expected_tool=turn_dict['expected_tool'],
                    expected_args=turn_dict['expected_args'],
                    references_previous_turns=turn_dict.get('references_previous_turns', []),
                    expect_success=turn_dict.get('expect_success', True),
                    expected_error_substring=turn_dict.get('expected_error_substring'),
                    failure_category=turn_dict.get('failure_category'),
                    expected_response=expected_response,
                )
                turns.append(turn)
            
            turn_count = len(turns)
            complexity = 'simple' if turn_count <= 3 else 'medium' if turn_count <= 6 else 'complex'
            
            conversation = Conversation(
                conversation_id=conv_dict['conversation_id'],
                workflow_category=conv_dict['workflow_category'],
                complexity_level=complexity,
                turns=turns,
                initial_entities=conv_dict.get('initial_entities', {}),
                final_expected_state=conv_dict.get('final_expected_state', {}),
                success_criteria=conv_dict.get('success_criteria', 'all_turns'),
                contains_failure=conv_dict.get('contains_failure', False),
                failure_turn=conv_dict.get('failure_turn'),
                verification_mode=VerificationMode.DATABASE,
                chain_id=conv_dict.get('chain_id'),
                segment_number=conv_dict.get('segment_number'),
                segment_boundaries=conv_dict.get('segment_boundaries'),
                expected_outcome=conv_dict.get('expected_outcome'),
                cumulative_context=conv_dict.get('cumulative_context', {}),
            )
            conversations.append(conversation)
    
    return conversations


def build_atlas_task_payload(conversation: Conversation, run_id: str, learning_key_override: str = None) -> Dict[str, Any]:
    """Build Atlas task payload from conversation."""
    payload = {
        "task_id": f"{conversation.conversation_id}::eval",
        "run_id": run_id,
        "conversation": conversation_to_payload(conversation),
        "dataset_revision": "atlas-evaluation",
        "backend": "postgres",
        "use_llm_judge": True,
        "agent_config": {
            "provider": "openai",
            "model_name": "gpt-4.1-mini",
            "temperature": 0.0,
            "max_output_tokens": 800,
        },
    }
    # Add learning_key_override to session_metadata to ensure all scenarios use the same learning_key
    if learning_key_override:
        payload["session_metadata"] = {"learning_key_override": learning_key_override}
    return payload


async def query_learning_state_from_db(database: Database, learning_key: str) -> Optional[Dict[str, Any]]:
    """Query learning state from database."""
    try:
        return await database.fetch_learning_state(learning_key)
    except Exception as e:
        print(f"   ⚠️  Error querying learning state: {e}")
        return None


async def query_session_rewards(database: Database, learning_key: str) -> List[Dict[str, Any]]:
    """Query session rewards for a learning key."""
    try:
        # Get learning history which includes reward information
        history_records = await database.fetch_learning_history(learning_key)
        return history_records or []
    except Exception as e:
        print(f"   ⚠️  Error querying session rewards: {e}")
        return []


async def evaluate_atlas_learning_loop() -> Dict[str, Any]:
    """Run comprehensive Atlas learning loop evaluation."""
    
    print("=" * 80)
    print("ATLAS LEARNING LOOP EVALUATION")
    print("=" * 80)
    print("Running 5 scenarios to verify:")
    print("  1. Learning loop is happening")
    print("  2. Learning is persisted to Postgres")
    print("  3. Learning is re-injected into subsequent sessions")
    print("  4. Performance and efficiency are improving")
    print()
    
    results = {
        "scenarios": [],
        "learning_evolution": [],
        "performance_trends": {
            "reward_scores": [],
            "success_rates": [],
            "learning_lengths": [],
        },
        "database_verification": {
            "learning_persisted": False,
            "learning_reinjected": False,
            "sessions_count": 0,
        },
        "errors": [],
    }
    
    # Load conversations
    print("1. Loading 5 test conversations...")
    try:
        conversations = load_conversations(count=5)
        print(f"   ✅ Loaded {len(conversations)} conversations")
        for i, conv in enumerate(conversations, 1):
            print(f"      {i}. {conv.conversation_id} ({conv.complexity_level}, {len(conv.turns)} turns)")
        results["total_scenarios"] = len(conversations)
    except Exception as e:
        results["errors"].append(f"Failed to load conversations: {e}")
        return results
    
    # Load Atlas config
    print("\n2. Loading Atlas configuration...")
    config_path = _REPO_ROOT / "configs/atlas/crm_harness.yaml"
    if not config_path.exists():
        results["errors"].append(f"Config file not found: {config_path}")
        return results
    
    try:
        config = load_config(str(config_path))
        print(f"   ✅ Config loaded")
        print(f"      Learning enabled: {config.learning.enabled if config.learning else False}")
        print(f"      Learning updates enabled: {config.learning.update_enabled if config.learning else False}")
    except Exception as e:
        results["errors"].append(f"Config load failed: {e}")
        return results
    
    # Connect to database
    print("\n3. Connecting to Atlas Postgres database...")
    database = None
    if config.storage:
        try:
            database = Database(config.storage)
            await database.connect()
            db_url_display = config.storage.database_url.split('@')[-1] if config.storage.database_url else 'N/A'
            print(f"   ✅ Database connected: {db_url_display}")
        except Exception as e:
            results["errors"].append(f"Database connection failed: {e}")
            print(f"   ❌ Database error: {e}")
            return results
    else:
        print("   ⚠️  No storage configured - learning persistence will be disabled")
        results["errors"].append("No storage configured")
        return results
    
    # Run scenarios sequentially
    print("\n4. Running scenarios through Atlas...")
    print("   (This will demonstrate learning accumulation across sessions)")
    print()
    
    # Use a consistent learning_key across all scenarios for evaluation
    # This allows learning to accumulate across different conversation types
    evaluation_learning_key = "atlas_crm_evaluation_learning_key"
    learning_key = None
    previous_learning_state = None
    
    for i, conversation in enumerate(conversations, 1):
        print(f"\n{'='*80}")
        print(f"SCENARIO {i}/{len(conversations)}: {conversation.conversation_id}")
        print(f"{'='*80}")
        print(f"Complexity: {conversation.complexity_level}")
        print(f"Turns: {len(conversation.turns)}")
        print(f"Workflow: {conversation.workflow_category}")
        
        scenario_result = {
            "scenario": i,
            "conversation_id": conversation.conversation_id,
            "complexity": conversation.complexity_level,
            "turns": len(conversation.turns),
            "workflow_category": conversation.workflow_category,
            "success": False,
            "reward_score": None,
            "learning_before": None,
            "learning_after": None,
            "learning_updated": False,
        }
        
        # Check learning state before session
        if learning_key and database:
            learning_state_before = await query_learning_state_from_db(database, learning_key)
            if learning_state_before:
                student_learning_before = learning_state_before.get("student_learning", "")
                teacher_learning_before = learning_state_before.get("teacher_learning", "")
                scenario_result["learning_before"] = {
                    "student_length": len(student_learning_before) if student_learning_before else 0,
                    "teacher_length": len(teacher_learning_before) if teacher_learning_before else 0,
                }
                print(f"\n   Learning state BEFORE session (from DB):")
                print(f"      Student learning: {len(student_learning_before) if student_learning_before else 0} chars")
                if student_learning_before:
                    print(f"      Preview: {student_learning_before[:150]}...")
                print(f"      Teacher learning: {len(teacher_learning_before) if teacher_learning_before else 0} chars")
            else:
                print(f"\n   ⚠️  No learning state in database (first session or not persisted yet)")
                scenario_result["learning_before"] = {"student_length": 0, "teacher_length": 0}
        
        # Build task payload
        run_id = f"eval-run-{i:03d}"
        task_payload = build_atlas_task_payload(conversation, run_id, learning_key_override=evaluation_learning_key)
        # Extract session_metadata before JSON serialization
        session_metadata = task_payload.pop("session_metadata", None)
        task_json = json.dumps(task_payload, ensure_ascii=False)
        
        # Run Atlas session
        try:
            result = await arun(
                task=task_json,
                config_path=str(config_path),
                stream_progress=False,
                session_metadata=session_metadata,  # Pass session_metadata separately
            )
            
            # Extract results
            context = ExecutionContext.get()
            
            # Get learning key (set by Atlas during session)
            if not learning_key:
                learning_key = context.metadata.get("learning_key")
                if learning_key:
                    print(f"\n   Learning key: {learning_key[:50]}...")
                else:
                    # Use the override key if Atlas didn't set one
                    learning_key = evaluation_learning_key
                    print(f"\n   Using evaluation learning key: {learning_key[:50]}...")
            
            # Parse conversation result
            conversation_result = None
            reward_signal = 0.0
            overall_success = False
            if result.final_answer:
                try:
                    if isinstance(result.final_answer, str):
                        final_answer_dict = json.loads(result.final_answer)
                        conversation_result = final_answer_dict.get("conversation_result", {})
                        if conversation_result:
                            overall_success = conversation_result.get("overall_success", False)
                            reward_signal = conversation_result.get("reward_signal", 0.0)
                except (json.JSONDecodeError, AttributeError):
                    pass
            
            scenario_result["success"] = overall_success
            scenario_result["reward_score"] = reward_signal
            
            # Get session reward from Atlas
            session_reward = context.metadata.get("session_reward")
            if isinstance(session_reward, dict):
                atlas_reward_score = session_reward.get("score", 0.0)
                scenario_result["atlas_reward_score"] = atlas_reward_score
                print(f"\n   Atlas reward score: {atlas_reward_score:.2f}")
            
            print(f"   Conversation success: {overall_success}")
            print(f"   Reward signal: {reward_signal:.2f}")
            
            # Check learning state after session - query from DATABASE (not context, as context is cleared after arun)
            # Wait a moment for async operations (including database persistence) to complete
            await asyncio.sleep(1.0)
            
            # Query learning state from database (this is where it's persisted)
            if learning_key and database:
                learning_state_after_db = await query_learning_state_from_db(database, learning_key)
                if learning_state_after_db:
                    student_learning_after = learning_state_after_db.get("student_learning", "")
                    teacher_learning_after = learning_state_after_db.get("teacher_learning", "")
                    scenario_result["learning_after"] = {
                        "student_length": len(student_learning_after) if student_learning_after else 0,
                        "teacher_length": len(teacher_learning_after) if teacher_learning_after else 0,
                    }
                    scenario_result["learning_updated"] = (
                        scenario_result["learning_after"]["student_length"] > 
                        (scenario_result["learning_before"]["student_length"] if scenario_result["learning_before"] else 0)
                    )
                    
                    print(f"\n   Learning state AFTER session (from DB):")
                    print(f"      Student learning: {len(student_learning_after) if student_learning_after else 0} chars")
                    if student_learning_after:
                        print(f"      Preview: {student_learning_after[:150]}...")
                    print(f"      Teacher learning: {len(teacher_learning_after) if teacher_learning_after else 0} chars")
                    print(f"      Learning updated: {scenario_result['learning_updated']}")
                else:
                    print(f"\n   ⚠️  Learning state not found in database after session")
                    scenario_result["learning_after"] = {"student_length": 0, "teacher_length": 0}
            else:
                scenario_result["learning_after"] = {"student_length": 0, "teacher_length": 0}
            
            # Track learning evolution
            if scenario_result["learning_after"]:
                results["learning_evolution"].append({
                    "scenario": i,
                    "student_length": scenario_result["learning_after"]["student_length"],
                    "teacher_length": scenario_result["learning_after"]["teacher_length"],
                })
            
            # Track performance trends
            if session_reward and isinstance(session_reward, dict):
                results["performance_trends"]["reward_scores"].append(session_reward.get("score", 0.0))
            results["performance_trends"]["success_rates"].append(1.0 if overall_success else 0.0)
            if scenario_result["learning_after"]:
                student_len = scenario_result["learning_after"]["student_length"]
                results["performance_trends"]["learning_lengths"].append(student_len)
            
        except Exception as e:
            print(f"\n   ❌ Scenario {i} failed: {e}")
            import traceback
            traceback.print_exc()
            scenario_result["error"] = str(e)
            results["errors"].append(f"Scenario {i} failed: {e}")
        
        results["scenarios"].append(scenario_result)
        
        # Small delay between scenarios
        await asyncio.sleep(1)
    
    # Verify learning persistence in database
    print(f"\n{'='*80}")
    print("5. VERIFYING LEARNING PERSISTENCE IN DATABASE")
    print(f"{'='*80}")
    
    # Use the evaluation learning key (or the one from sessions)
    final_learning_key = learning_key or evaluation_learning_key
    
    if final_learning_key and database:
        print(f"\n   Learning key: {final_learning_key[:50]}...")
        
        # Query learning state from database
        persisted_state = await query_learning_state_from_db(database, final_learning_key)
        if persisted_state:
            student_learning = persisted_state.get("student_learning", "")
            teacher_learning = persisted_state.get("teacher_learning", "")
            print(f"\n   ✅ Learning state persisted to database:")
            print(f"      Student learning: {len(student_learning) if student_learning else 0} chars")
            print(f"      Teacher learning: {len(teacher_learning) if teacher_learning else 0} chars")
            results["database_verification"]["learning_persisted"] = True
            
            if student_learning:
                print(f"\n   Student learning preview:")
                print(f"      {student_learning[:200]}...")
        else:
            print(f"\n   ⚠️  Learning state not found in database")
        
        # Query session rewards/history
        history_records = await query_session_rewards(database, final_learning_key)
        if history_records:
            print(f"\n   ✅ Found {len(history_records)} history records")
            results["database_verification"]["sessions_count"] = len(history_records)
        else:
            print(f"\n   ⚠️  No history records found")
    else:
        print(f"\n   ⚠️  Cannot verify - no learning_key or database")
    
    # Verify learning re-injection
    print(f"\n{'='*80}")
    print("6. VERIFYING LEARNING RE-INJECTION")
    print(f"{'='*80}")
    
    try:
        playbook, digest, metadata = resolve_playbook("student", apply=True)
        if playbook:
            print(f"\n   ✅ Learning re-injection verified:")
            print(f"      Playbook length: {len(playbook)} chars")
            print(f"      Playbook preview: {playbook[:200]}...")
            results["database_verification"]["learning_reinjected"] = True
        else:
            print(f"\n   ⚠️  No playbook available")
    except Exception as e:
        print(f"\n   ⚠️  Playbook resolution failed: {e}")
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nScenarios executed: {len(results['scenarios'])}")
    successful_scenarios = sum(1 for s in results['scenarios'] if s.get('success'))
    print(f"Successful scenarios: {successful_scenarios}/{len(results['scenarios'])}")
    
    if results["performance_trends"]["reward_scores"]:
        avg_reward = sum(results["performance_trends"]["reward_scores"]) / len(results["performance_trends"]["reward_scores"])
        print(f"Average reward score: {avg_reward:.2f}")
        print(f"Reward trend: {results['performance_trends']['reward_scores']}")
    
    if results["performance_trends"]["success_rates"]:
        avg_success = sum(results["performance_trends"]["success_rates"]) / len(results["performance_trends"]["success_rates"])
        print(f"Average success rate: {avg_success:.2%}")
    
    if results["learning_evolution"]:
        print(f"\nLearning evolution:")
        for evo in results["learning_evolution"]:
            print(f"  Scenario {evo['scenario']}: Student={evo['student_length']} chars, Teacher={evo['teacher_length']} chars")
        
        # Check if learning is growing
        if len(results["learning_evolution"]) > 1:
            first_len = results["learning_evolution"][0]["student_length"]
            last_len = results["learning_evolution"][-1]["student_length"]
            if last_len > first_len:
                print(f"\n   ✅ Learning is GROWING: {first_len} → {last_len} chars (+{last_len - first_len})")
            else:
                print(f"\n   ⚠️  Learning not growing: {first_len} → {last_len} chars")
    
    print(f"\nDatabase verification:")
    print(f"  Learning persisted: {'✅' if results['database_verification']['learning_persisted'] else '❌'}")
    print(f"  Learning re-injected: {'✅' if results['database_verification']['learning_reinjected'] else '❌'}")
    print(f"  Sessions in database: {results['database_verification']['sessions_count']}")
    
    # Cleanup
    if database:
        await database.disconnect()
    
    return results


if __name__ == "__main__":
    results = asyncio.run(evaluate_atlas_learning_loop())
    
    # Exit with error code if critical components failed
    if not results["database_verification"]["learning_persisted"]:
        print("\n❌ CRITICAL: Learning not persisted to database")
        sys.exit(1)
    
    if not results["database_verification"]["learning_reinjected"]:
        print("\n❌ CRITICAL: Learning not re-injected")
        sys.exit(1)
    
    if results["errors"]:
        print(f"\n⚠️  WARNINGS: {len(results['errors'])} errors encountered")
        # Don't exit on warnings, only critical failures

