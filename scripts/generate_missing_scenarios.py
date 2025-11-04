#!/usr/bin/env python
"""Generate missing tool scenarios using Curator structured output.

This script generates scenarios for tools that are missing from the scenario corpus,
validates them against the mock harness, and appends them to scenarios_clean.jsonl.
"""

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from pydantic import BaseModel, Field

try:
    from bespokelabs import curator
except ImportError as exc:
    raise ImportError(
        "bespokelabs-curator is required. Install via: pip install -r requirements.txt"
    ) from exc

from src.conversation_schema import Conversation, ConversationTurn
from src.evaluation.conversation_harness import ConversationHarness
from src.reference_resolver import resolve_template


class Scenario(BaseModel):
    """Single CRM scenario structure."""

    task: str = Field(description="Task name matching the tool")
    utterance: str = Field(description="Natural language user request")
    expected_tool: str = Field(description="Tool name to be called")
    expected_args: Dict[str, Any] = Field(description="Expected tool arguments")
    setup_entities: Dict[str, Any] = Field(default_factory=dict, description="Pre-existing entities")
    expect_success: bool = Field(description="Whether scenario should succeed")
    expected_error_substring: str | None = Field(
        default=None, description="Expected error substring if expect_success=False"
    )


class ScenariosResponse(BaseModel):
    """Structured output containing multiple scenarios."""

    scenarios: List[Scenario] = Field(description="List of CRM scenarios")


class ScenarioGenerator(curator.LLM):
    """Curator LLM for generating CRM scenarios."""

    response_format = ScenariosResponse
    batch = True

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        *,
        backend_params: Dict[str, Any] | None = None,
        generation_params: Dict[str, Any] | None = None,
    ) -> None:
        default_backend = {
            "max_requests_per_minute": 200,
            "max_tokens_per_minute": 60_000,
        }
        merged_backend = dict(default_backend)
        if backend_params:
            merged_backend.update(backend_params)

        default_generation = {
            "temperature": 0.7,
            "max_output_tokens": 2000,
        }
        merged_generation = dict(default_generation)
        if generation_params:
            merged_generation.update(generation_params)

        super().__init__(
            model_name=model_name,
            backend="litellm",
            backend_params=merged_backend,
            generation_params=merged_generation,
        )

    def prompt(self, input: Dict[str, Any]) -> str:
        """Build prompt for scenario generation."""
        tool_name = input["tool_name"]
        count = input["count"]
        success_type = input["success_type"]  # "success" or "failure"

        return f"""Generate {count} {success_type} scenarios for the CRM tool '{tool_name}'.

Each scenario must include:
- task: Task name matching the tool
- utterance: Natural language user request
- expected_tool: Tool name (must be '{tool_name}')
- expected_args: Dictionary of expected arguments with proper types
- setup_entities: Pre-existing entities needed (e.g., {{"client_id": "..."}})
- expect_success: {success_type == "success"}
- expected_error_substring: Error substring if failure scenario

For upload_document tool, expected_args should include:
- entity_type: "Client", "Contact", "Opportunity", "Quote", or "Contract"
- entity_id: UUID string of the entity
- file_name: String filename

Return scenarios as JSON following the ScenariosResponse schema."""

    def parse(self, input: Dict[str, Any], response: ScenariosResponse) -> List[Dict[str, Any]]:
        """Parse response into list of scenario dictionaries."""
        tool_name = input["tool_name"]
        scenarios = []
        for scenario in response.scenarios:
            scenario_dict = scenario.model_dump()
            scenario_dict["scenario_id"] = f"SC-{uuid.uuid4().hex[:8].upper()}"
            scenario_dict["expected_tool"] = tool_name  # Ensure tool name matches
            scenario_dict["verification_mode"] = "database"
            scenario_dict["failure_category"] = None if scenario_dict["expect_success"] else "validation"
            scenarios.append(scenario_dict)
        return scenarios


def validate_scenario(scenario: Dict[str, Any]) -> bool:
    """Validate a scenario by creating a single-turn conversation and running it."""
    try:
        # Create single-turn conversation
        turn = ConversationTurn(
            turn_id=1,
            user_utterance=scenario.get("utterance", ""),
            expected_tool=scenario["expected_tool"],
            expected_args=scenario.get("expected_args", {}),
            expect_success=scenario.get("expect_success", True),
        )

        conversation = Conversation(
            conversation_id=f"VALIDATE-{scenario['scenario_id']}",
            workflow_category="Validation",
            complexity_level="simple",
            turns=[turn],
            initial_entities=scenario.get("setup_entities", {}),
        )

        # Run through harness
        harness = ConversationHarness([conversation])
        results = harness.run()

        if not results:
            return False

        result = results[0]
        if scenario.get("expect_success", True):
            return result.overall_success
        else:
            # For failure scenarios, check that execution failed appropriately
            return not result.overall_success or result.failed_at_turn == 1

    except Exception as exc:
        print(f"  Validation failed: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool",
        type=str,
        required=True,
        help="Tool name to generate scenarios for (e.g., upload_document)",
    )
    parser.add_argument(
        "--success-count",
        type=int,
        default=10,
        help="Number of success scenarios to generate",
    )
    parser.add_argument(
        "--failure-count",
        type=int,
        default=7,
        help="Number of failure scenarios to generate (for 60/40 split)",
    )
    parser.add_argument(
        "--scenarios-file",
        type=Path,
        default=Path("artifacts/scenarios_500/scenarios_clean.jsonl"),
        help="Path to scenarios JSONL file to append to",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-4.1-mini",
        help="Curator model name",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate but don't append")
    args = parser.parse_args()

    generator = ScenarioGenerator(model_name=args.model_name)

    all_scenarios = []

    # Generate success scenarios
    print(f"Generating {args.success_count} success scenarios for {args.tool}...")
    success_dataset = Dataset.from_list([
        {"tool_name": args.tool, "count": args.success_count, "success_type": "success"}
    ])
    success_result = generator(success_dataset)
    success_scenarios = list(success_result.dataset)

    # Generate failure scenarios
    print(f"Generating {args.failure_count} failure scenarios for {args.tool}...")
    failure_dataset = Dataset.from_list([
        {"tool_name": args.tool, "count": args.failure_count, "success_type": "failure"}
    ])
    failure_result = generator(failure_dataset)
    failure_scenarios = list(failure_result.dataset)

    # Validate scenarios
    print("Validating scenarios...")
    valid_scenarios = []
    for scenario in success_scenarios + failure_scenarios:
        scenario_dict = scenario if isinstance(scenario, dict) else dict(scenario)
        if validate_scenario(scenario_dict):
            valid_scenarios.append(scenario_dict)
            print(f"  ✓ {scenario_dict.get('scenario_id', 'unknown')}")
        else:
            print(f"  ✗ {scenario_dict.get('scenario_id', 'unknown')} - validation failed")

    if not valid_scenarios:
        print("❌ No valid scenarios generated. Aborting.")
        return

    print(f"\n✅ Generated {len(valid_scenarios)} valid scenarios ({len(success_scenarios)} success, {len(failure_scenarios)} failure)")

    if args.dry_run:
        print("\n[DRY RUN] Would append scenarios:")
        for scenario in valid_scenarios:
            print(f"  {scenario.get('scenario_id')}: {scenario.get('utterance', '')[:60]}...")
        return

    # Append to scenarios file
    print(f"\nAppending {len(valid_scenarios)} scenarios to {args.scenarios_file}...")
    args.scenarios_file.parent.mkdir(parents=True, exist_ok=True)
    with args.scenarios_file.open("a", encoding="utf-8") as handle:
        for scenario in valid_scenarios:
            handle.write(json.dumps(scenario) + "\n")

    print(f"✅ Successfully appended {len(valid_scenarios)} scenarios")


if __name__ == "__main__":
    main()

