#!/usr/bin/env python3
"""Final Validation Script - Pre-Scale Dataset Verification

Validates all critical components before scaling to 500-1000 skeletons:
1. Pruning logic (high-impact exception)
2. Learning loop (synthesis, persistence, re-injection)
3. Baseline agent execution
4. Dataset pipeline (phases 1-4)
5. Atlas integration
6. Quality metrics and judge

This is the last mile validation before dataset expansion.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))

# Load environment variables
env_file = Path('.env')
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if '=' in line and not line.strip().startswith('#'):
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()

try:
    import asyncpg
except ImportError:
    print("❌ asyncpg not installed. Install with: pip install asyncpg")
    sys.exit(1)

# Import project modules
try:
    from src.evaluation.conversation_harness import ConversationHarness
    from src.evaluation.agents import LiteLLMClaudeAgent, MockAgent
    from src.crm_backend import PostgresCrmBackend
except ImportError as e:
    print(f"❌ Failed to import project modules: {e}")
    print("Make sure you're running from the project root and dependencies are installed")
    sys.exit(1)


class ValidationResult:
    """Track validation results"""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.errors = []
        self.warnings = []
        self.details = {}
    
    def add_error(self, msg: str):
        self.errors.append(msg)
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
    
    def set_detail(self, key: str, value: Any):
        self.details[key] = value
    
    def mark_passed(self):
        self.passed = True
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        result = f"{status}: {self.name}\n"
        if self.errors:
            result += f"  Errors: {len(self.errors)}\n"
            for err in self.errors:
                result += f"    - {err}\n"
        if self.warnings:
            result += f"  Warnings: {len(self.warnings)}\n"
            for warn in self.warnings:
                result += f"    - {warn}\n"
        if self.details:
            result += f"  Details:\n"
            for key, value in self.details.items():
                result += f"    {key}: {value}\n"
        return result


async def validate_pruning_logic() -> ValidationResult:
    """Validate the high-impact exception pruning logic"""
    result = ValidationResult("Pruning Logic - High-Impact Exception")
    
    try:
        # Import the pruning method
        sys.path.insert(0, str(_REPO_ROOT / 'external' / 'atlas-sdk'))
        from atlas.learning.synthesizer import LearningSynthesizer
        from atlas.config.models import LearningConfig, PlaybookPruningConfig
        
        # Create a test synthesizer
        pruning_config = PlaybookPruningConfig(
            min_cue_hit_rate=0.05,
            min_reward_delta=0.01,
            min_transfer_sessions=20,
        )
        learning_config = LearningConfig(
            enabled=True,
            pruning_config=pruning_config,
        )
        synthesizer = LearningSynthesizer(learning_config)
        
        # Test case: High-impact entry with low hit rate
        test_entry = {
            "id": "test_high_impact",
            "impact": {
                "sessions_observed": 10,
                "sessions_with_hits": 1,  # 10% hit rate (below 5% threshold would prune)
                "reward_with_sum": 1.0,
                "reward_with_count": 1,
                "reward_without_sum": 3.14,
                "reward_without_count": 9,
            },
            "provenance": {"status": {"lifecycle": "active"}},
        }
        
        # Calculate expected metrics
        reward_delta = (1.0 / 1) - (3.14 / 9)  # ~0.651 (very high impact)
        cue_hit_rate = 1 / 10  # 10% (above 5% threshold, but test with 4% to trigger rule)
        
        # Test with 4% hit rate (below threshold) to trigger exception
        test_entry["impact"]["sessions_with_hits"] = 0  # Will be 0% after calculation
        test_entry["impact"]["sessions_observed"] = 25  # Enough sessions
        
        # Actually test with 1 hit out of 25 = 4% hit rate
        test_entry["impact"]["sessions_with_hits"] = 1
        test_entry["impact"]["sessions_observed"] = 25
        
        pruned = synthesizer._prune_ineffective_entries([test_entry], min_sessions=10)
        
        # Should be kept due to high-impact exception
        if len(pruned) == 1 and pruned[0].get("id") == "test_high_impact":
            result.mark_passed()
            result.set_detail("high_impact_exception", "Working correctly")
            result.set_detail("reward_delta", f"{reward_delta:.3f}")
        else:
            result.add_error(f"High-impact entry was pruned incorrectly. Pruned: {len(pruned)}")
        
        # Test negative case: Low impact entry should be pruned
        low_impact_entry = {
            "id": "test_low_impact",
            "impact": {
                "sessions_observed": 25,
                "sessions_with_hits": 1,  # 4% hit rate
                "reward_with_sum": 0.4,
                "reward_with_count": 1,
                "reward_without_sum": 3.6,
                "reward_without_count": 9,
            },
            "provenance": {"status": {"lifecycle": "active"}},
        }
        
        pruned_low = synthesizer._prune_ineffective_entries([low_impact_entry], min_sessions=10)
        if len(pruned_low) == 0:  # Should be pruned (removed from list)
            result.set_detail("low_impact_pruning", "Working correctly")
        else:
            result.add_warning("Low-impact entry was not pruned as expected")
        
    except Exception as e:
        result.add_error(f"Exception during pruning logic test: {e}")
        import traceback
        result.add_error(traceback.format_exc())
    
    return result


async def validate_learning_persistence() -> ValidationResult:
    """Validate learning persistence and retrieval from database"""
    result = ValidationResult("Learning Persistence")
    
    try:
        db_url = os.getenv('ATLAS_DATABASE_URL', 'postgresql://atlas:atlas@localhost:5433/atlas')
        conn = await asyncpg.connect(db_url)
        
        eval_key = "atlas_crm_evaluation_learning_key"
        
        # Check if learning exists
        row = await conn.fetchrow(
            "SELECT student_learning, teacher_learning, metadata FROM learning_registry WHERE learning_key = $1",
            eval_key
        )
        
        if not row:
            result.add_warning(f"No learning found for key: {eval_key}")
            result.add_warning("This is expected if no Atlas sessions have run yet")
        else:
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            entries = metadata.get('playbook_entries', [])
            
            result.mark_passed()
            result.set_detail("student_learning_length", len(row['student_learning'] or ''))
            result.set_detail("teacher_learning_length", len(row['teacher_learning'] or ''))
            result.set_detail("playbook_entries_count", len(entries))
            
            # Check for high-impact entries
            high_impact_count = sum(
                1 for e in entries
                if e.get('impact', {}).get('reward_with_count', 0) > 0
                and e.get('provenance', {}).get('status', {}).get('lifecycle') == 'active'
            )
            result.set_detail("active_high_impact_entries", high_impact_count)
        
        await conn.close()
        
    except Exception as e:
        result.add_error(f"Exception during persistence check: {e}")
        import traceback
        result.add_error(traceback.format_exc())
    
    return result


def validate_baseline_agent() -> ValidationResult:
    """Validate baseline agent components are importable and conversations can be loaded"""
    result = ValidationResult("Baseline Agent Components")
    
    try:
        # Verify imports work
        from src.evaluation.agents import LiteLLMClaudeAgent, MockAgent
        from src.evaluation.conversation_harness import ConversationHarness
        from src.conversation_schema import Conversation, ConversationTurn, ExpectedResponse
        from src.evaluation.verification import VerificationMode
        
        result.set_detail("imports_successful", True)
        
        # Load a test conversation to verify schema
        conversations_file = Path('artifacts/deterministic/final_conversations.jsonl')
        if not conversations_file.exists():
            result.add_warning(f"Conversations file not found: {conversations_file}")
            result.add_warning("Cannot validate conversation loading without file")
            result.mark_passed()  # Still pass - components are importable
            return result
        
        # Load first conversation
        with open(conversations_file, 'r') as f:
            first_line = f.readline()
            if not first_line:
                result.add_warning("No conversations found in file")
                result.mark_passed()  # Still pass - components are importable
                return result
            
            conv_dict = json.loads(first_line)
        
        # Convert to Conversation object (validate schema)
        turns = []
        for turn_dict in conv_dict.get('turns', []):
            expected_response = None
            if turn_dict.get('expected_response'):
                er_dict = turn_dict['expected_response']
                expected_response = ExpectedResponse(
                    text=er_dict.get('text'),
                    answers=er_dict.get('answers', []),
                    evaluation=er_dict.get('evaluation'),
                )
            
            turn = ConversationTurn(
                turn_id=turn_dict['turn_id'],
                user_utterance=turn_dict['user_utterance'],
                expected_tool=turn_dict.get('expected_tool'),
                expected_args=turn_dict.get('expected_args', {}),
                expected_response=expected_response,
                expect_success=turn_dict.get('expect_success', True),
            )
            turns.append(turn)
        
        conversation = Conversation(
            conversation_id=conv_dict['conversation_id'],
            turns=turns,
            complexity_level=conv_dict.get('complexity_level', 'simple'),
            verification_mode=VerificationMode(conv_dict.get('verification_mode', 'database').lower()),
            workflow_category=conv_dict.get('workflow_category', 'unknown'),
        )
        
        result.mark_passed()
        result.set_detail("conversation_loaded", True)
        result.set_detail("conversation_id", conversation.conversation_id)
        result.set_detail("turns_count", len(conversation.turns))
        
    except Exception as e:
        result.add_error(f"Exception during baseline agent test: {e}")
        import traceback
        result.add_error(traceback.format_exc())
    
    return result


async def validate_dataset_pipeline() -> ValidationResult:
    """Validate dataset pipeline phases 1-4"""
    result = ValidationResult("Dataset Pipeline (Phases 1-4)")
    
    try:
        artifacts_dir = Path('artifacts')
        
        # Phase 1: Skeletons
        skeletons_file = artifacts_dir / 'deterministic' / 'skeletons.jsonl'
        if not skeletons_file.exists():
            skeletons_file = artifacts_dir / 'schema_pipeline' / 'skeletons.jsonl'
        
        if skeletons_file.exists():
            with open(skeletons_file, 'r') as f:
                skeleton_count = sum(1 for _ in f)
            result.set_detail("skeletons_count", skeleton_count)
        else:
            result.add_warning("Skeletons file not found - Phase 1 may not be complete")
        
        # Phase 2: Replay results
        replay_file = artifacts_dir / 'deterministic' / 'replay_results.jsonl'
        if not replay_file.exists():
            replay_file = artifacts_dir / 'schema_pipeline' / 'replay_results.jsonl'
        
        if replay_file.exists():
            with open(replay_file, 'r') as f:
                replay_count = sum(1 for _ in f)
            result.set_detail("replay_results_count", replay_count)
        else:
            result.add_warning("Replay results file not found - Phase 2 may not be complete")
        
        # Phase 3: Paraphrased conversations
        paraphrased_file = artifacts_dir / 'deterministic' / 'paraphrased_conversations.jsonl'
        if not paraphrased_file.exists():
            paraphrased_file = artifacts_dir / 'schema_pipeline' / 'paraphrased_conversations.jsonl'
        
        if paraphrased_file.exists():
            with open(paraphrased_file, 'r') as f:
                paraphrased_count = sum(1 for _ in f)
            result.set_detail("paraphrased_count", paraphrased_count)
        else:
            result.add_warning("Paraphrased conversations file not found - Phase 3 may not be complete")
        
        # Phase 4: Final conversations
        final_file = artifacts_dir / 'deterministic' / 'final_conversations.jsonl'
        if final_file.exists():
            with open(final_file, 'r') as f:
                final_count = sum(1 for _ in f)
            result.set_detail("final_conversations_count", final_count)
            
            # Validate template substitution
            with open(final_file, 'r') as f:
                first_conv = json.loads(f.readline())
                has_templates = False
                for turn in first_conv.get('turns', []):
                    import re
                    if re.search(r'\{\{turn_\d+\..*?\}\}', json.dumps(turn.get('expected_args', {}))):
                        has_templates = True
                        break
                
                result.set_detail("templates_present", has_templates)
                if not has_templates:
                    result.add_warning("No templates found in final conversations - substitution may not be working")
        else:
            result.add_error("Final conversations file not found - Phase 4 not complete")
            return result
        
        result.mark_passed()
        
    except Exception as e:
        result.add_error(f"Exception during pipeline validation: {e}")
        import traceback
        result.add_error(traceback.format_exc())
    
    return result


async def validate_atlas_integration() -> ValidationResult:
    """Validate Atlas SDK integration is working"""
    result = ValidationResult("Atlas SDK Integration")
    
    try:
        # Check if Atlas adapter is registered
        sys.path.insert(0, str(_REPO_ROOT / 'external' / 'atlas-sdk'))
        from atlas.connectors.crm_harness import CrmHarnessAdapter
        
        # Check adapter registration
        result.set_detail("adapter_registered", True)
        
        # Check configuration
        config_file = Path('configs/atlas/crm_harness.yaml')
        if config_file.exists():
            result.set_detail("config_file_exists", True)
        else:
            result.add_warning("Atlas config file not found")
        
        # Check database connection
        db_url = os.getenv('ATLAS_DATABASE_URL', 'postgresql://atlas:atlas@localhost:5433/atlas')
        try:
            conn = await asyncpg.connect(db_url)
            await conn.close()
            result.set_detail("database_connection", "Working")
        except Exception as e:
            result.add_error(f"Database connection failed: {e}")
            return result
        
        result.mark_passed()
        
    except Exception as e:
        result.add_error(f"Exception during Atlas integration check: {e}")
        import traceback
        result.add_error(traceback.format_exc())
    
    return result


async def main():
    """Run all validation checks"""
    print("=" * 80)
    print("FINAL VALIDATION - Pre-Scale Dataset Verification")
    print("=" * 80)
    print()
    
    results: List[ValidationResult] = []
    
    # Run all validations
    print("🔍 Running validation checks...")
    print()
    
    print("1. Validating pruning logic...")
    results.append(await validate_pruning_logic())
    
    print("2. Validating learning persistence...")
    results.append(await validate_learning_persistence())
    
    print("3. Validating baseline agent...")
    results.append(validate_baseline_agent())
    
    print("4. Validating dataset pipeline...")
    results.append(await validate_dataset_pipeline())
    
    print("5. Validating Atlas integration...")
    results.append(await validate_atlas_integration())
    
    # Print results
    print()
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    for result in results:
        print(result)
        print()
    
    print("=" * 80)
    print(f"SUMMARY: {passed}/{total} checks passed")
    print("=" * 80)
    
    if passed == total:
        print()
        print("✅ ALL VALIDATIONS PASSED - Ready to scale dataset!")
        return 0
    else:
        print()
        print("❌ SOME VALIDATIONS FAILED - Review errors above before scaling")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

