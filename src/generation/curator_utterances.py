"""Minimal Curator LLM wrapper that only generates natural-language utterances."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence

from pydantic import BaseModel, Field

try:
    from bespokelabs import curator
except ImportError as exc:  # pragma: no cover - exercised at runtime
    raise ImportError(
        "bespokelabs-curator is required to run the conversation generator. "
        "Install dependencies via `pip install -r requirements.txt`."
    ) from exc


class UtteranceResponse(BaseModel):
    """Structured output model requiring a single user utterance string."""

    user_utterance: str = Field(
        ...,
        description="Natural-language user request for the CRM assistant. "
        "Use concise, conversational language and avoid tool or JSON references.",
    )


class CuratorUtteranceGenerator(curator.LLM):
    """Thin Curator wrapper that prompts the LLM for utterances only."""

    response_format = UtteranceResponse
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
            "max_output_tokens": 120,
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

    # ------------------------------------------------------------------
    # Core Curator hooks
    # ------------------------------------------------------------------
    def prompt(self, input: Dict[str, Any]) -> str:
        """Build a prompt instructing the model to emit a single utterance."""
        workflow = input["workflow_category"]
        turn_number = input["turn_number"]
        tool_name = input["tool_name"]
        argument_summary: str = input.get("argument_summary", "")
        entity_context: Sequence[str] = input.get("entity_context", [])
        history: Sequence[Mapping[str, str]] = input.get("history", [])

        history_lines = self._format_history(history)
        entity_lines = self._format_entities(entity_context)

        prompt_sections = [
            "You script the USER side of a CRM assistant conversation.",
            f"Workflow category: {workflow}",
            f"Current turn: {turn_number}",
            f"Assistant tool that will run: {tool_name}",
        ]
        if argument_summary:
            prompt_sections.append(f"Tool arguments provided by the template: {argument_summary}")
        if entity_lines:
            prompt_sections.append("Relevant CRM entities:\n" + entity_lines)
        if history_lines:
            prompt_sections.append("Conversation so far:\n" + history_lines)

        prompt_sections.append(
            "Write the user's next message in plain natural language. "
            "Do not mention tools, JSON, or instructions. "
            "Return a short sentence or two that would cause the assistant to execute the tool above."
        )
        prompt_sections.append('Respond with JSON: {"user_utterance": "<message>"}')

        return "\n\n".join(prompt_sections)

    def parse(self, input: Dict[str, Any], response: Any) -> Dict[str, Any]:
        """Extract the utterance from the model response, with fallbacks."""
        if isinstance(response, UtteranceResponse):
            utterance = response.user_utterance.strip()
        elif isinstance(response, str):
            utterance = self._parse_string_response(response)
        elif isinstance(response, Mapping):
            utterance = str(response.get("user_utterance", "")).strip()
        else:
            utterance = ""

        return {
            "conversation_id": input.get("conversation_id"),
            "turn_number": input["turn_number"],
            "user_utterance": utterance,
        }

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_history(history: Sequence[Mapping[str, str]]) -> str:
        lines: List[str] = []
        for item in history:
            turn = item.get("turn")
            speaker = item.get("speaker", "User")
            content = item.get("content", "")
            if turn is not None:
                lines.append(f"Turn {turn} – {speaker}: {content}")
            else:
                lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _format_entities(entity_context: Sequence[str]) -> str:
        return "\n".join(f"- {line}" for line in entity_context if line)

    @staticmethod
    def _parse_string_response(response: str) -> str:
        stripped = response.strip()
        if not stripped:
            return ""

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, Mapping):
                return str(parsed.get("user_utterance", "")).strip()
        except json.JSONDecodeError:
            pass

        # Attempt to locate JSON embedded in markdown
        import re

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, Mapping):
                    return str(parsed.get("user_utterance", "")).strip()
            except json.JSONDecodeError:
                return ""

        return stripped
