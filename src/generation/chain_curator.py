"""Curator LLM classes for chained conversation generation.

This module implements Curator LLM classes following Bespoke patterns:
- ScenarioSelector: Selects scenarios for each turn in a segment
- ChainUtteranceGenerator: Generates user utterances for chained conversations
"""

from typing import Any, Dict, List, Mapping, Sequence

from datasets import Dataset
from pydantic import BaseModel, Field

try:
    from bespokelabs import curator
except ImportError as exc:
    raise ImportError(
        "bespokelabs-curator is required. Install via: pip install -r requirements.txt"
    ) from exc

from src.generation.curator_chain_models import (
    ScenarioSelection,
    ScenarioSelectionResponse,
    TurnUtterance,
    TurnUtteranceResponse,
)


class ScenarioSelector(curator.LLM):
    """Curator LLM for selecting scenarios per turn in a workflow segment."""

    response_format = ScenarioSelectionResponse
    batch = True

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        *,
        backend_params: Mapping[str, Any] | None = None,
        generation_params: Mapping[str, Any] | None = None,
    ) -> None:
        default_backend = {
            "max_requests_per_minute": 200,
            "max_tokens_per_minute": 60_000,
        }
        merged_backend = dict(default_backend)
        if backend_params:
            merged_backend.update(backend_params)

        default_generation = {
            "temperature": 0.3,
            "max_output_tokens": 1000,
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
        """Build prompt for scenario selection."""
        workflow_category = input["workflow_category"]
        turn_templates = input["turn_templates"]
        available_scenarios = input.get("available_scenarios", {})

        turn_info = []
        for i, turn_template in enumerate(turn_templates, start=1):
            tool_name = turn_template.get("tool_name", "")
            scenarios_for_tool = available_scenarios.get(tool_name, [])
            turn_info.append(
                f"Turn {i}: tool={tool_name}, available_scenarios={len(scenarios_for_tool)}"
            )

        return f"""Select scenarios for each turn in a {workflow_category} workflow segment.

Turns to select scenarios for:
{chr(10).join(turn_info)}

Available scenarios by tool:
{self._format_available_scenarios(available_scenarios)}

For each turn, select an appropriate scenario ID from the available scenarios.
Return selections as JSON following the ScenarioSelectionResponse schema."""

    def parse(self, input: Dict[str, Any], response: ScenarioSelectionResponse) -> List[Dict[str, Any]]:
        """Parse response into list of selection dictionaries."""
        workflow_category = input["workflow_category"]
        turn_templates = input["turn_templates"]

        results = []
        for selection in response.selections:
            results.append({
                "workflow_category": workflow_category,
                "scenario_id": selection.scenario_id,
                "tool_name": selection.tool_name,
                "turn_number": selection.turn_number,
            })
        return results

    @staticmethod
    def _format_available_scenarios(available_scenarios: Dict[str, List[str]]) -> str:
        """Format available scenarios for prompt."""
        lines = []
        for tool_name, scenario_ids in available_scenarios.items():
            lines.append(f"  {tool_name}: {', '.join(scenario_ids[:10])}")  # Limit to first 10
            if len(scenario_ids) > 10:
                lines.append(f"    ... and {len(scenario_ids) - 10} more")
        return "\n".join(lines) if lines else "  (no scenarios available)"


class ChainUtteranceGenerator(curator.LLM):
    """Curator LLM for generating user utterances in chained conversations."""

    response_format = TurnUtteranceResponse
    batch = True

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        *,
        backend_params: Mapping[str, Any] | None = None,
        generation_params: Mapping[str, Any] | None = None,
    ) -> None:
        default_backend = {
            "max_requests_per_minute": 200,
            "max_tokens_per_minute": 60_000,
        }
        merged_backend = dict(default_backend)
        if backend_params:
            merged_backend.update(backend_params)

        default_generation = {
            "temperature": 0.3,
            "max_output_tokens": 500,
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
        """Build prompt for utterance generation."""
        workflow_category = input["workflow_category"]
        turn_count = input["turn_count"]
        argument_summaries = input.get("argument_summaries", [])
        cumulative_context = input.get("cumulative_context", {})
        previous_segments = input.get("previous_segments", [])

        context_parts = [
            f"Workflow category: {workflow_category}",
            f"Number of turns: {turn_count}",
        ]

        if cumulative_context:
            context_parts.append(f"Cumulative context: {cumulative_context}")

        if previous_segments:
            context_parts.append(f"Previous segments: {len(previous_segments)} segments completed")

        if argument_summaries:
            context_parts.append("Tool arguments per turn:")
            for i, args in enumerate(argument_summaries, start=1):
                context_parts.append(f"  Turn {i}: {args}")

        return f"""Generate natural language user utterances for a CRM conversation segment.

{chr(10).join(context_parts)}

Instructions:
- Generate {turn_count} user utterances, one per turn
- Use conversational language with pronouns and implicit references
- Reference previous turns naturally (e.g., "Create an opp for them", "Update that quote")
- Avoid mentioning tools, JSON, or technical details
- Each utterance should be concise (1-2 sentences)

Return utterances as JSON following the TurnUtteranceResponse schema."""

    def parse(self, input: Dict[str, Any], response: TurnUtteranceResponse) -> List[Dict[str, Any]]:
        """Parse response into list of utterance dictionaries."""
        workflow_category = input["workflow_category"]
        turn_count = input["turn_count"]

        results = []
        for i, utterance in enumerate(response.utterances[:turn_count], start=1):
            results.append({
                "workflow_category": workflow_category,
                "turn_number": i,
                "user_utterance": utterance.user_utterance,
            })
        return results

