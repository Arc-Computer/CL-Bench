#!/usr/bin/env python
"""Generate multi-turn CRM conversations using the lean pipeline."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from src.generation.conversation_generator import instantiate_conversation
from src.evaluation.conversation_harness import ConversationHarness
from src.conversation_templates import WORKFLOW_TEMPLATES, WorkflowTemplate
from src.generation.curator_utterances import CuratorUtteranceGenerator
from src.pipeline.scenario_repository import ScenarioRepository


DEFAULT_OUTPUT_DIR = Path("artifacts/conversations_multiturn")
DEFAULT_COUNT = 1000
SMOKE_TEST_COUNT = 10

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


def conversation_to_dict(conversation) -> Dict:
    data = asdict(conversation)
    data["verification_mode"] = conversation.verification_mode.value
    for turn in data["turns"]:
        turn.pop("failure_category", None)
    return data


def main() -> None:
    args = parse_args()
    total_count = SMOKE_TEST_COUNT if args.smoke_test else args.count
    plan = compute_plan(total_count, args.smoke_test)

    repo = ScenarioRepository(
        scenario_path=args.single_turn_scenarios,
        schema_path=Path("data/fake_crm_tables_schema.json"),
        task_weights_path=Path("data/Agent_tasks.csv"),
    )
    curator = CuratorUtteranceGenerator(model_name=args.model_name)
    rng = random.Random(args.seed)

    conversations = []
    for template_key, desired_count in plan.items():
        template: WorkflowTemplate = WORKFLOW_TEMPLATES[template_key]
        for index in range(desired_count):
            conversation_id = f"{template.workflow_id}-{index:04d}"
            conversation = instantiate_conversation(template, repo, curator, rng, conversation_id=conversation_id)

            harness = ConversationHarness([conversation])
            result = harness.run()[0]
            if not result.overall_success:
                raise RuntimeError(
                    f"Conversation {conversation_id} failed validation at turn {result.failed_at_turn}: {result.error_message}"
                )

            conversations.append(conversation)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "conversations.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for conversation in conversations:
            handle.write(json.dumps(conversation_to_dict(conversation)) + "\n")

    counts = Counter(conv.workflow_category for conv in conversations)
    print(f"Wrote {len(conversations)} conversations to {output_path}")
    for category, count in counts.items():
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
