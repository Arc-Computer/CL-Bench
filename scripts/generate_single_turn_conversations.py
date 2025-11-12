#!/usr/bin/env python3
"""
Generate single-turn conversations weighted by task distribution from Agent_tasks.csv.

This script samples single-turn scenarios proportionally to real customer usage patterns,
generating conversations where each conversation = 1 tool call (vs multi-turn workflows).
"""

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.scenario_repository import ScenarioRepository


def normalize_tool_name(tool_name: str) -> str:
    """Normalize tool name to match Agent_tasks.csv task names."""
    # Map tool names to task names from CSV
    mapping = {
        'create_new_opportunity': 'CREATE NEW OPPORTUNITY',
        'create_opportunity': 'CREATE NEW OPPORTUNITY',
        'modify_opportunity': 'MODIFY OPPORTUNITY',
        'opportunity_search': 'OPPORTUNITY SEARCH',
        'delete_opportunity': 'DELETE OPPORTUNITY',
        'create_new_client': 'CREATE NEW CLIENT',
        'create_client': 'CREATE NEW CLIENT',
        'modify_client': 'MODIFY CLIENT',
        'client_search': 'CLIENT SEARCH',
        'create_new_contact': 'CREATE NEW CONTACT',
        'create_contact': 'CREATE NEW CONTACT',
        'modify_contact': 'MODIFY CONTACT',
        'contact_search': 'CONTACT SEARCH',
        'create_quote': 'CREATE QUOTE',
        'modify_quote': 'MODIFY QUOTE',
        'compare_quotes': 'COMPARE QUOTES',
        'cancel_quote': 'CANCEL QUOTE',
        'delete_quote': 'DELETE QUOTE',
        'quote_search': 'QUOTE SEARCH',
        'quote_details': 'QUOTE DETAILS',
        'upload_document': 'UPLOAD DOCUMENT',
        'create_new_contract': 'CREATE NEW CONTRACT',
        'create_contract': 'CREATE NEW CONTRACT',
        'modify_contract': 'MODIFY CONTRACT',
        'contract_search': 'CONTRACT SEARCH',
        'summarize_opportunities': 'SUMMARIZE OPPORTUNITIES',
        'view_opportunity_details': 'VIEW OPPORTUNITY DETAILS',
        'opportunity_details': 'OPPORTUNITY DETAILS',
        'clone_opportunity': 'CLONE OPPORTUNITY',
        'add_note': 'ADD NOTE',
        'company_search': 'COMPANY SEARCH',
    }
    return mapping.get(tool_name, tool_name.upper().replace('_', ' '))


def sample_weighted_scenarios(
    repository: ScenarioRepository,
    count: int,
    success_ratio: float,
    seed: int,
) -> List[Any]:
    """Sample scenarios weighted by task distribution from Agent_tasks.csv."""

    rng = random.Random(seed)

    # Get task weights
    task_weights = repository.task_weights
    total_weight = sum(task_weights.values())

    # Group scenarios by tool
    success_scenarios_by_tool = repository.success_scenarios_by_tool
    failure_scenarios_by_tool = repository.failure_scenarios_by_tool

    # Calculate how many success vs failure we need
    success_count = int(count * success_ratio)
    failure_count = count - success_count

    print(f"Target distribution: {success_count} success, {failure_count} failure")
    print()

    # Sample success scenarios weighted by task frequency
    success_samples = []
    failure_samples = []

    # Create weighted pool for success scenarios
    # Note: task_weights keys are already tool names (lowercase, underscores)
    weighted_success_pool = []
    for tool_name, scenarios in success_scenarios_by_tool.items():
        weight = task_weights.get(tool_name, 0)
        if weight > 0 and len(scenarios) > 0:
            weighted_success_pool.extend([(tool_name, weight)] * len(scenarios))

    # Sample success scenarios
    print(f"Sampling {success_count} success scenarios...")
    tool_samples = rng.choices(
        [tool for tool, _ in weighted_success_pool],
        weights=[weight for _, weight in weighted_success_pool],
        k=success_count
    )

    # For each sampled tool, pick a random scenario
    for tool_name in tool_samples:
        scenarios = success_scenarios_by_tool[tool_name]
        scenario = rng.choice(scenarios)
        success_samples.append(scenario)

    # Sample failure scenarios (also weighted by task frequency, but from failure pool)
    weighted_failure_pool = []
    for tool_name, scenarios in failure_scenarios_by_tool.items():
        weight = task_weights.get(tool_name, 0)
        if weight > 0 and len(scenarios) > 0:
            weighted_failure_pool.extend([(tool_name, weight)] * len(scenarios))

    if len(weighted_failure_pool) > 0:
        print(f"Sampling {failure_count} failure scenarios...")
        tool_samples = rng.choices(
            [tool for tool, _ in weighted_failure_pool],
            weights=[weight for _, weight in weighted_failure_pool],
            k=failure_count
        )

        for tool_name in tool_samples:
            scenarios = failure_scenarios_by_tool[tool_name]
            scenario = rng.choice(scenarios)
            failure_samples.append(scenario)
    else:
        print(f"WARNING: No failure scenarios available, generating {failure_count} from success scenarios")
        # Fallback: sample from success scenarios and mark as failures
        tool_samples = rng.choices(
            [tool for tool, _ in weighted_success_pool],
            weights=[weight for _, weight in weighted_success_pool],
            k=failure_count
        )
        for tool_name in tool_samples:
            scenarios = success_scenarios_by_tool[tool_name]
            scenario = rng.choice(scenarios)
            # Mark as failure
            failure_samples.append(scenario)

    # Combine and shuffle
    all_samples = success_samples + failure_samples
    rng.shuffle(all_samples)

    print()
    print("Sampled scenarios by tool:")
    tool_counts = Counter([s.expected_tool for s in all_samples])
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1])[:15]:
        target_pct = (task_weights.get(tool, 0) / total_weight) * 100
        actual_pct = (count / len(all_samples)) * 100
        print(f"  {tool}: {count} ({actual_pct:.1f}% | target {target_pct:.1f}%)")

    return all_samples


def generate_single_turn_conversation(
    scenario: Any,
    conversation_id: str,
    use_curator: bool = False,
) -> Dict[str, Any]:
    """Generate a single-turn conversation from a scenario."""

    # Generate user utterance
    if use_curator:
        # TODO: Implement Curator-based utterance generation
        # For now, use scenario utterance
        user_utterance = scenario.utterance or scenario.intent or f"Use {scenario.expected_tool}"
    else:
        # Use high-quality utterance from scenario
        user_utterance = scenario.utterance or scenario.intent or f"Use {scenario.expected_tool}"

    # Build turn
    turn = {
        "turn_id": 1,
        "user_utterance": user_utterance,
        "expected_tool": scenario.expected_tool,
        "expected_args": scenario.expected_args,
        "expect_success": scenario.expect_success,
        "references_previous_turns": [],
    }

    # Build conversation
    conversation = {
        "conversation_id": conversation_id,
        "workflow_category": "Single-turn operation",
        "complexity_level": "simple",
        "scenario_id": scenario.scenario_id,
        "contains_failure": not scenario.expect_success,
        "failure_turn": 1 if not scenario.expect_success else None,
        "turns": [turn],
        "initial_entities": scenario.setup_entities,
        "success_criteria": {
            "all_turns_succeed": scenario.expect_success,
        },
    }

    return conversation


def main():
    parser = argparse.ArgumentParser(
        description="Generate single-turn conversations weighted by task distribution"
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of conversations to generate"
    )
    parser.add_argument(
        "--task-weights",
        type=Path,
        default=Path("data/Agent_tasks.csv"),
        help="Path to task weights CSV (default: data/Agent_tasks.csv)"
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("artifacts/scenarios_single_turn/scenarios_clean.jsonl"),
        help="Path to scenarios JSONL (default: artifacts/scenarios_single_turn/scenarios_clean.jsonl)"
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("data/fake_crm_tables_schema.json"),
        help="Path to CRM schema (default: data/fake_crm_tables_schema.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated conversations"
    )
    parser.add_argument(
        "--success-ratio",
        type=float,
        default=0.6,
        help="Target success ratio (default: 0.6)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--use-curator",
        action="store_true",
        help="Use Curator for natural language generation (requires API key)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SINGLE-TURN CONVERSATION GENERATION")
    print("=" * 80)
    print()
    print(f"Count: {args.count}")
    print(f"Success ratio: {args.success_ratio:.0%}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print(f"Use Curator: {args.use_curator}")
    print()

    # Load scenario repository
    print("Loading scenario repository...")
    repository = ScenarioRepository(
        scenario_path=args.scenarios,
        schema_path=args.schema,
        task_weights_path=args.task_weights,
    )

    stats = repository.stats()
    print(f"  Total scenarios: {stats['scenario_count']}")
    print(f"  Success scenarios by tool: {len(repository.success_scenarios_by_tool)} tools")
    print(f"  Failure scenarios by tool: {len(repository.failure_scenarios_by_tool)} tools")
    print()

    # Sample scenarios
    print("Sampling scenarios weighted by task distribution...")
    sampled_scenarios = sample_weighted_scenarios(
        repository=repository,
        count=args.count,
        success_ratio=args.success_ratio,
        seed=args.seed,
    )
    print()

    # Generate conversations
    print(f"Generating {len(sampled_scenarios)} single-turn conversations...")
    conversations = []

    for i, scenario in enumerate(sampled_scenarios):
        conversation_id = f"SINGLE-{i:04d}"
        conversation = generate_single_turn_conversation(
            scenario=scenario,
            conversation_id=conversation_id,
            use_curator=args.use_curator,
        )
        conversations.append(conversation)

    # Write output
    output_path = args.output_dir / "conversations.jsonl"
    print(f"Writing {len(conversations)} conversations to {output_path}...")

    with output_path.open("w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    # Print summary
    print()
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Generated: {len(conversations)} conversations")
    print(f"Output file: {output_path}")
    print(f"Success: {sum(1 for c in conversations if not c['contains_failure'])} ({sum(1 for c in conversations if not c['contains_failure'])/len(conversations)*100:.1f}%)")
    print(f"Failure: {sum(1 for c in conversations if c['contains_failure'])} ({sum(1 for c in conversations if c['contains_failure'])/len(conversations)*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
