#!/usr/bin/env python
"""Generate multi-turn CRM conversations using the lean pipeline."""

from __future__ import annotations

import argparse
import os
import json
import math
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from src.evaluation.conversation_harness import ConversationHarness
from src.conversation_templates import (
    WORKFLOW_CHAINS,
    WORKFLOW_TEMPLATES,
    WorkflowChain,
    WorkflowTemplate,
)
from src.generation.chain_conversation_generator import instantiate_chained_conversation
from src.generation.chain_curator import ChainUtteranceGenerator, ScenarioSelector
from src.generation.conversation_generator import instantiate_conversation
from src.generation.curator_utterances import CuratorUtteranceGenerator
from src.pipeline.scenario_repository import ScenarioRepository


DEFAULT_OUTPUT_DIR = Path("artifacts/conversations_multiturn")
DEFAULT_CHAIN_OUTPUT_DIR = Path("artifacts/conversations_chains")
DEFAULT_COUNT = 1000
SMOKE_TEST_COUNT = 10

CHAIN_BY_ID: Dict[str, WorkflowChain] = {
    chain.chain_id: chain for chain in WORKFLOW_CHAINS.values()
}

WORKFLOW_WEIGHTS: Mapping[str, float] = {
    "client_management": 0.12,
    "contact_management": 0.1,
    "opportunity_management": 0.23,
    "quote_generation": 0.18,
    "client_onboarding": 0.12,
    "deal_pipeline": 0.15,
    "document_workflow": 0.05,
    "multi_entity_search": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-turn-scenarios", type=Path, default=Path("artifacts/scenarios_500/scenarios_clean.jsonl"), help="Path to validated single-turn scenarios")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of multi-turn conversations to generate")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for deterministic sampling")
    parser.add_argument(
        "--mode",
        choices=("workflow", "chain"),
        default="workflow",
        help="Generation mode: 'workflow' for single templates, 'chain' for multi-segment workflows.",
    )
    parser.add_argument(
        "--chain-id",
        dest="chain_ids",
        action="append",
        help="Workflow chain identifier to generate (repeatable). Defaults to all chains when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the conversations.jsonl file will be written",
    )
    parser.add_argument("--model-name", default="gpt-4.1-mini", help="Curator-backed model name")
    parser.add_argument("--smoke-test", action="store_true", help="Generate a small deterministic sample (10 conversations)")
    return parser.parse_args()


def compute_plan(total_count: int, smoke_test: bool) -> Dict[str, int]:
    if smoke_test:
        plan = {key: 1 for key in WORKFLOW_TEMPLATES}
        remainder = max(0, total_count - len(plan))
        keys = list(WORKFLOW_TEMPLATES.keys())
        idx = 0
        while remainder > 0:
            plan[keys[idx % len(keys)]] += 1
            remainder -= 1
            idx += 1
        return plan

    weights = WORKFLOW_WEIGHTS
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=1e-3):
        raise ValueError("Workflow weights must sum to 1.0")

    plan: Dict[str, int] = {}
    remaining = total_count
    for template_key, weight in weights.items():
        count = int(total_count * weight)
        plan[template_key] = count
        remaining -= count

    # Distribute leftovers starting from highest weight templates.
    for template_key in sorted(weights, key=weights.get, reverse=True):
        if remaining <= 0:
            break
        plan[template_key] += 1
        remaining -= 1

    return plan


def compute_chain_plan(total_count: int, chain_ids: Sequence[str], smoke_test: bool) -> Dict[str, int]:
    unique_ids = list(dict.fromkeys(chain_ids))
    if not unique_ids:
        raise ValueError("At least one chain ID must be provided for chain generation.")

    if smoke_test:
        plan = {chain_id: 1 for chain_id in unique_ids}
    else:
        plan = {chain_id: 0 for chain_id in unique_ids}

    allocated = sum(plan.values())
    remaining = max(0, total_count - allocated)

    idx = 0
    while remaining > 0:
        chain_id = unique_ids[idx % len(unique_ids)]
        plan[chain_id] += 1
        remaining -= 1
        idx += 1

    return plan


def conversation_to_dict(conversation) -> Dict:
    data = asdict(conversation)
    data["verification_mode"] = conversation.verification_mode.value
    for turn in data["turns"]:
        turn.pop("failure_category", None)
    return data


def generate_workflow_conversations(
    plan: Mapping[str, int],
    repo: ScenarioRepository,
    curator: CuratorUtteranceGenerator,
    rng: random.Random,
) -> List:
    conversations = []
    for template_key, desired_count in plan.items():
        template: WorkflowTemplate = WORKFLOW_TEMPLATES[template_key]
        for index in range(desired_count):
            conversation_id = f"{template.workflow_id}-{index:04d}"
            conversation = instantiate_conversation(
                template,
                repo,
                curator,
                rng,
                conversation_id=conversation_id,
            )

            harness = ConversationHarness([conversation])
            result = harness.run()[0]
            if not result.overall_success:
                raise RuntimeError(
                    f"Conversation {conversation_id} failed validation at turn "
                    f"{result.failed_at_turn}: {result.error_message}"
                )

            conversations.append(conversation)
    return conversations


def generate_chain_conversations(
    chain_plan: Mapping[str, int],
    repo: ScenarioRepository,
    scenario_selector: Optional[ScenarioSelector],
    utterance_generator: Optional[ChainUtteranceGenerator],
    rng: random.Random,
) -> List:
    conversations = []
    for chain_key, desired_count in chain_plan.items():
        chain: WorkflowChain = WORKFLOW_CHAINS[chain_key]
        expected_failure = any(not outcome for outcome in chain.success_pattern)
        for index in range(desired_count):
            conversation_id = f"{chain.chain_id}-{index:04d}"
            conversation = instantiate_chained_conversation(
                chain,
                repo,
                scenario_selector,
                utterance_generator,
                rng,
                conversation_id=conversation_id,
            )

            harness = ConversationHarness([conversation])
            result = harness.run()[0]
            if not result.chain_success and not expected_failure:
                raise RuntimeError(
                    f"Chained conversation {conversation_id} failed unexpectedly at "
                    f"turn {result.failed_at_turn}: {result.error_message}"
                )
            if expected_failure and not result.metadata.get("expected_failure"):
                raise RuntimeError(
                    f"Chained conversation {conversation_id} was expected to fail, but "
                    "the harness did not surface the failure."
                )

            conversations.append(conversation)
    return conversations


def main() -> None:
    args = parse_args()
    total_count = SMOKE_TEST_COUNT if args.smoke_test else args.count

    repo = ScenarioRepository(
        scenario_path=args.single_turn_scenarios,
        schema_path=Path("data/fake_crm_tables_schema.json"),
        task_weights_path=Path("data/Agent_tasks.csv"),
    )
    rng = random.Random(args.seed)

    if args.mode == "workflow":
        plan = compute_plan(total_count, args.smoke_test)
        curator = CuratorUtteranceGenerator(model_name=args.model_name)
        conversations = generate_workflow_conversations(plan, repo, curator, rng)
        output_dir = args.output_dir
        output_filename = "conversations.jsonl"
        summary_counts = Counter(conv.workflow_category for conv in conversations)
        failure_counts: Counter[str] = Counter()
    else:
        chain_ids = args.chain_ids or list(WORKFLOW_CHAINS.keys())
        plan = compute_chain_plan(total_count, chain_ids, args.smoke_test)
        offline_mode = os.environ.get("CURATOR_SIMPLE_DATASET") == "1"
        scenario_selector: Optional[ScenarioSelector] = None
        chain_curator: Optional[ChainUtteranceGenerator] = None
        if not offline_mode:
            scenario_selector = ScenarioSelector(model_name=args.model_name)
            chain_curator = ChainUtteranceGenerator(model_name=args.model_name)
        conversations = generate_chain_conversations(
            plan,
            repo,
            scenario_selector,
            chain_curator,
            rng,
        )
        output_dir = args.output_dir
        if output_dir == DEFAULT_OUTPUT_DIR:
            output_dir = DEFAULT_CHAIN_OUTPUT_DIR
        output_filename = "chains.jsonl"
        summary_counts = Counter((conv.chain_id or "unknown") for conv in conversations)
        failure_counts = Counter((conv.chain_id or "unknown") for conv in conversations if conv.contains_failure)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    with output_path.open("w", encoding="utf-8") as handle:
        for conversation in conversations:
            handle.write(json.dumps(conversation_to_dict(conversation)) + "\n")

    print(f"Wrote {len(conversations)} conversations to {output_path}")
    if args.mode == "workflow":
        for category, count in summary_counts.items():
            print(f"  {category}: {count}")
    else:
        for chain_id, count in summary_counts.items():
            chain = CHAIN_BY_ID.get(chain_id)
            description = chain.description if chain else "Unknown chain"
            failure_count = failure_counts.get(chain_id, 0)
            expected_failure = bool(chain and any(not outcome for outcome in chain.success_pattern))
            if expected_failure:
                failure_suffix = f", failures={failure_count} (expected failure)"
            else:
                failure_suffix = f", failures={failure_count}" if failure_count else ""
            print(f"  {chain_id}: {count} ({description}{failure_suffix})")


if __name__ == "__main__":
    main()
