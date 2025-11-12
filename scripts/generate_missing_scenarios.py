#!/usr/bin/env python
"""Generate missing tool scenarios using Curator structured output.

This script generates scenarios for tools that are missing from the scenario corpus,
validates them against the mock harness, and appends them to scenarios_clean.jsonl.
"""

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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
    expected_response: Dict[str, Any] = Field(default_factory=dict, description="Expected assistant response metadata")


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

        normalized_model = model_name.lower()
        default_generation: Dict[str, Any] = {"temperature": 0.7}
        if any(prefix in normalized_model for prefix in ("gpt-4.1", "gpt-4o", "gpt-5", "o1", "o3")):
            default_generation["max_completion_tokens"] = 2000
        elif "gpt" in normalized_model:
            default_generation["max_tokens"] = 2000
        else:
            default_generation["max_output_tokens"] = 2000
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
- expected_response: {{ "text": "...", "evaluation": "structured" or "judge", "answers": ["..."], "requires_judge": false }}
  * For success scenarios, provide a clear acknowledgment that reflects the tool result.
  * For failure scenarios, explain the error surfaced to the user.

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
            _normalise_expected_response(scenario_dict)
            scenarios.append(scenario_dict)
        return scenarios


def _summarize_expected_response(
    tool_name: str,
    expected_args: Mapping[str, Any],
    expect_success: bool,
    expected_error: Optional[str],
) -> str:
    """Create a concise natural-language summary for the tool outcome."""
    arg_items: List[str] = []
    for key, value in list(expected_args.items())[:3]:
        if isinstance(value, Mapping):
            arg_items.append(f"{key}=…")
        else:
            arg_items.append(f"{key}={value}")
    argument_summary = ", ".join(arg_items)
    if expect_success:
        if argument_summary:
            return f"Completed {tool_name} with {argument_summary}"
        return f"Completed {tool_name} successfully"
    error_text = expected_error or "validation error"
    return f"{tool_name} failed as expected: {error_text}"


def _normalise_expected_response(scenario: Dict[str, Any]) -> None:
    """Ensure every scenario carries a well-formed expected_response stanza."""
    payload = scenario.get("expected_response") or {}
    expect_success = bool(scenario.get("expect_success", True))
    expected_args = scenario.get("expected_args", {}) or {}
    expected_error = scenario.get("expected_error_substring")

    requires_judge = bool(payload.get("requires_judge", False))
    evaluation = str(payload.get("evaluation", "judge" if requires_judge else "structured")).lower()
    if requires_judge:
        evaluation = "judge"
    elif evaluation not in {"structured", "judge"}:
        evaluation = "structured"

    text = str(payload.get("text", "")).strip()
    if not text:
        text = _summarize_expected_response(scenario.get("expected_tool", ""), expected_args, expect_success, expected_error)

    answers = payload.get("answers") or []
    normalised_answers = [str(answer).strip() for answer in answers if str(answer).strip()]
    if evaluation == "structured" and not normalised_answers:
        normalised_answers = [text]

    scenario["expected_response"] = {
        "text": text,
        "evaluation": evaluation,
        "answers": normalised_answers,
        "requires_judge": evaluation == "judge" or requires_judge,
    }


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
            expected_response=scenario.get("expected_response"),
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
        per_turn = (result.per_turn_results or [None])[0] or {}

        if scenario.get("expect_success", True):
            return bool(per_turn.get("tool_success")) and bool(per_turn.get("response_success"))

        return (
            not result.overall_success
            and per_turn.get("verification") == "expected_failure_diagnostic"
            and bool(per_turn.get("tool_success"))
        )

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
        default=Path("artifacts/scenarios_single_turn/scenarios_clean.jsonl"),
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

    # Generate success scenarios
    success_scenarios = []
    if args.success_count > 0:
        print(f"Generating {args.success_count} success scenarios for {args.tool}...")
        success_dataset = Dataset.from_list([
            {"tool_name": args.tool, "count": args.success_count, "success_type": "success"}
        ])
        success_result = generator(success_dataset)
        success_scenarios = list(success_result.dataset)

    # Generate failure scenarios
    failure_scenarios = []
    if args.failure_count > 0:
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
