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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.evaluation.conversation_harness import ConversationHarness
from src.conversation_templates import (
    CHAIN_FAILURE_RATIO,
    CHAIN_RATIO_TOLERANCE,
    WORKFLOW_CHAINS,
    WORKFLOW_TEMPLATES,
    WorkflowChain,
    WorkflowTemplate,
    expand_chain_ids,
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


def _chain_contains_failure(chain: WorkflowChain) -> bool:
    return any(not outcome for outcome in chain.success_pattern)


def _distribute_counts(total: int, keys: Sequence[str]) -> Dict[str, int]:
    plan = {key: 0 for key in keys}
    if total <= 0 or not keys:
        return plan
    base, remainder = divmod(total, len(keys))
    for key in keys:
        plan[key] = base
    for index in range(remainder):
        key = keys[index % len(keys)]
        plan[key] += 1
    return plan


class _OfflineScenarioSelector:
    def __call__(self, dataset: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
        raise RuntimeError(
            "ScenarioSelector stub invoked unexpectedly in offline mode. "
            "Ensure CURATOR_SIMPLE_DATASET=1 for deterministic generation."
        )


class _OfflineUtteranceGenerator:
    def __call__(self, dataset: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
        raise RuntimeError(
            "ChainUtteranceGenerator stub invoked unexpectedly in offline mode. "
            "Ensure CURATOR_SIMPLE_DATASET=1 for deterministic generation."
        )


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
    expanded_ids = expand_chain_ids(chain_ids)
    unique_ids = list(dict.fromkeys(expanded_ids))
    if not unique_ids:
        raise ValueError("At least one chain ID must be provided for chain generation.")

    success_keys: List[str] = []
    failure_keys: List[str] = []
    for chain_key in unique_ids:
        chain = WORKFLOW_CHAINS[chain_key]
        if _chain_contains_failure(chain):
            failure_keys.append(chain_key)
        else:
            success_keys.append(chain_key)

    if not success_keys:
        raise ValueError(
            "Chain generation requires at least one success-only workflow chain to honor the target ratio."
        )
    if not failure_keys:
        raise ValueError(
            "Chain generation requires at least one failure-bearing workflow chain to honor the target ratio."
        )

    if total_count <= 0:
        return {}

    target_failure = int(round(total_count * CHAIN_FAILURE_RATIO))
    target_success = total_count - target_failure

    if total_count >= 3:
        if target_failure == 0:
            target_failure = 1
            target_success = total_count - target_failure
        if target_success == 0:
            target_success = 1
            target_failure = total_count - target_success

    failure_distribution = _distribute_counts(target_failure, failure_keys)
    success_distribution = _distribute_counts(target_success, success_keys)

    plan: Dict[str, int] = {key: 0 for key in unique_ids}
    for key, value in {**success_distribution, **failure_distribution}.items():
        plan[key] = value

    # Remove chains that received zero allocations to keep the plan concise.
    plan = {key: value for key, value in plan.items() if value > 0}

    allocated = sum(plan.values())
    if allocated != total_count:
        raise RuntimeError(
            f"Chain plan allocation mismatch: requested {total_count}, allocated {allocated}."
        )

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
    *,
    smoke_test: bool = False,
) -> List:
    conversations: List = []
    failure_conversations = 0
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

            if smoke_test:
                _print_chain_summary(conversation, result)

            conversations.append(conversation)
            if conversation.contains_failure:
                failure_conversations += 1
    if conversations:
        ratio = failure_conversations / len(conversations)
        allowed_deviation = max(CHAIN_RATIO_TOLERANCE, 1.0 / len(conversations))
        if abs(ratio - CHAIN_FAILURE_RATIO) > allowed_deviation:
            raise RuntimeError(
                f"Chained generation produced failure ratio {ratio:.3f}, "
                f"expected {CHAIN_FAILURE_RATIO:.2f}±{allowed_deviation:.2f}."
            )
    return conversations


def _print_chain_summary(conversation, result) -> None:
    print("=" * 80)
    outcome = "success" if result.chain_success else "failure"
    print(f"{conversation.conversation_id} | chain={conversation.chain_id} | overall={outcome}")
    if not result.per_segment_results:
        print("  (no segment results)")
        return
    for segment in result.per_segment_results:
        expected = segment.get("expected_outcome")
        actual = segment.get("actual_outcome")
        status = "OK" if expected == actual else "MISMATCH"
        turn_range = f"{segment.get('start_turn')}–{segment.get('end_turn')}"
        failure_note = ""
        if segment.get("failed_at_turn"):
            failure_note = f", failed_at={segment['failed_at_turn']}"
        print(
            f"  • segment {segment['segment_number']} (turns {turn_range}): "
            f"expected={expected}, actual={actual} [{status}]{failure_note}"
        )


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
        requested_chain_ids = args.chain_ids or list(WORKFLOW_CHAINS.keys())
        plan = compute_chain_plan(total_count, requested_chain_ids, args.smoke_test)
        offline_mode = os.environ.get("CURATOR_SIMPLE_DATASET") == "1"
        scenario_selector: Optional[ScenarioSelector] = None
        chain_curator: Optional[ChainUtteranceGenerator] = None
        if not offline_mode:
            scenario_selector = ScenarioSelector(model_name=args.model_name)
            chain_curator = ChainUtteranceGenerator(model_name=args.model_name)
        else:
            scenario_selector = _OfflineScenarioSelector()
            chain_curator = _OfflineUtteranceGenerator()
        conversations = generate_chain_conversations(
            plan,
            repo,
            scenario_selector,
            chain_curator,
            rng,
            smoke_test=args.smoke_test,
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
        total_conversations = len(conversations)
        failure_total = sum(conv.contains_failure for conv in conversations)
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
        if total_conversations:
            observed_ratio = failure_total / total_conversations
            allowed_deviation = max(CHAIN_RATIO_TOLERANCE, 1.0 / total_conversations)
            print(
                f"  Failure ratio: {failure_total}/{total_conversations} "
                f"({observed_ratio * 100:.1f}% | target {CHAIN_FAILURE_RATIO * 100:.1f}% +/- {allowed_deviation * 100:.1f}%)"
            )


if __name__ == "__main__":
    main()
