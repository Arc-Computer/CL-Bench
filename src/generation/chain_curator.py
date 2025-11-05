"""Curator LLM classes for chained conversation generation.

This module implements Curator LLM classes following Bespoke patterns:
- ScenarioSelector: Selects scenarios for each turn in a segment
- ChainUtteranceGenerator: Generates user utterances for chained conversations
"""

from __future__ import annotations

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
    TurnMetadata,
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

        default_generation = _default_generation_params(model_name, 1000)
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
        turn_template_payload: Sequence[Mapping[str, Any]] = input["turn_templates"]
        turn_templates: List[TurnMetadata] = [
            TurnMetadata.model_validate(item) for item in turn_template_payload
        ]
        available_scenarios = input.get("available_scenarios", {})
        workflow_description = input.get("workflow_description", "")

        turn_info: List[str] = []
        for turn in turn_templates:
            turn_number = turn.turn_number
            tool_name = turn.tool_name
            desired = turn.desired_outcome
            persona = turn.persona_hint or "default persona"
            stage_hint = turn.stage_hint or "general stage"
            dependencies = ", ".join(turn.handoff_dependencies or []) or "none"
            key = f"turn_{turn_number}:{tool_name}"
            scenarios_for_tool = available_scenarios.get(key, [])
            turn_info.append(
                f"Turn {turn_number}: tool={tool_name}, desired={desired}, persona={persona}, "
                f"stage_hint={stage_hint}, handoff_dependencies={dependencies}, "
                f"available_scenarios={len(scenarios_for_tool)}"
            )

        tags_summary = self._format_scenario_tags(input.get("scenario_tags", {}))

        description_block = f"Workflow description: {workflow_description}\n" if workflow_description else ""

        return (
            f"Select scenarios for each turn in a {workflow_category} workflow segment.\n"
            f"{description_block}\n"
            "Turns to select scenarios for:\n"
            f"{chr(10).join(turn_info)}\n\n"
            "Available scenarios by tool:\n"
            f"{self._format_available_scenarios(available_scenarios)}\n\n"
            "Scenario tags:\n"
            f"{tags_summary}\n\n"
            "For each turn, select an appropriate scenario ID from the available scenarios.\n"
            "Return selections as JSON following the ScenarioSelectionResponse schema."
        )

    def parse(self, input: Dict[str, Any], response: ScenarioSelectionResponse) -> List[Dict[str, Any]]:
        """Parse response into list of selection dictionaries."""
        workflow_category = input["workflow_category"]

        results = []
        for selection in response.selections:
            results.append(
                {
                    "workflow_category": workflow_category,
                    "scenario_id": selection.scenario_id,
                    "tool_name": selection.tool_name,
                    "turn_number": selection.turn_number,
                    "justification": selection.justification,
                    "handoff_actions": dict(selection.handoff_actions),
                }
            )
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

    @staticmethod
    def _format_scenario_tags(tags: Mapping[str, Mapping[str, Any]]) -> str:
        if not tags:
            return "  (no tag metadata provided)"
        lines: List[str] = []
        max_lines = 20
        for idx, (scenario_id, metadata) in enumerate(sorted(tags.items())):
            if idx >= max_lines:
                remaining = len(tags) - max_lines
                lines.append(f"  ... and {remaining} more tagged scenarios")
                break
            formatted = ", ".join(f"{key}={value}" for key, value in sorted(metadata.items()))
            if len(formatted) > 200:
                formatted = formatted[:197] + "..."
            lines.append(f"  {scenario_id}: {formatted or 'no tags'}")
        return "\n".join(lines)


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

        default_generation = _default_generation_params(model_name, 500)
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
        turn_metadata_payload: Sequence[Mapping[str, Any]] = input.get("turn_metadata", [])
        turn_metadata: List[TurnMetadata] = [
            TurnMetadata.model_validate(item) for item in turn_metadata_payload
        ]

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

        if turn_metadata:
            context_parts.append("Turn metadata:")
            for turn in turn_metadata:
                dependencies = ", ".join(turn.handoff_dependencies or []) or "none"
                persona = turn.persona_hint or "default persona"
                stage_hint = turn.stage_hint or "general stage"
                context_parts.append(
                    f"  Turn {turn.turn_number}: tool={turn.tool_name}, "
                    f"desired={turn.desired_outcome}, persona={persona}, "
                    f"stage_hint={stage_hint}, handoff_dependencies={dependencies}"
                )

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
            results.append(
                {
                    "workflow_category": workflow_category,
                    "turn_number": utterance.turn_number or i,
                    "user_utterance": utterance.user_utterance,
                    "persona_hint": utterance.persona_hint,
                    "stage_focus": utterance.stage_focus,
                    "referenced_entities": list(utterance.referenced_entities or []),
                    "handoff_summary": utterance.handoff_summary,
                }
            )
        return results


def _default_generation_params(model_name: str, max_tokens: int) -> Dict[str, Any]:
    """Build generation params compatible with the selected model provider."""
    normalized = model_name.lower()
    params: Dict[str, Any] = {}
    if not normalized.startswith("gpt-5"):
        params["temperature"] = 0.3

    if "gemini" in normalized or "claude" in normalized or "haiku" in normalized:
        params["max_output_tokens"] = max_tokens
    elif any(prefix in normalized for prefix in ("gpt-4.1", "gpt-4o", "gpt-5", "o1", "o3")):
        params["max_completion_tokens"] = max_tokens
    elif "gpt" in normalized:
        params["max_tokens"] = max_tokens
    else:
        params["max_output_tokens"] = max_tokens
    return params
