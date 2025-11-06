"""Agent abstractions for CRM baseline evaluation.

This module defines the interface used by :mod:`src.evaluation.conversation_harness`
to interact with different agent implementations.  The harness only requires a
single method – ``tool_call`` – and receives a structured response describing the
tool invocation the agent wishes to execute.  Concrete agents can be implemented
for live LLM providers, deterministic mocks, or scripted behaviours.
"""

from __future__ import annotations

import csv
import inspect
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.crm_sandbox import MockCrmApi

from src.conversation_schema import Conversation, ConversationTurn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SCHEMA_PATH = _REPO_ROOT / "data" / "fake_crm_tables_schema.json"
_DEFAULT_TASKS_PATH = _REPO_ROOT / "data" / "Agent_tasks.csv"


@lru_cache(maxsize=None)
def _load_schema(path: str) -> Dict[str, Any]:
    schema_path = Path(path)
    if not schema_path.exists():
        return {}
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=None)
def _load_tasks(path: str) -> List[Dict[str, Any]]:
    tasks_path = Path(path)
    if not tasks_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with tasks_path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


@lru_cache(maxsize=1)
def _tool_catalog() -> List[str]:
    entries: List[str] = []
    for name, func in inspect.getmembers(MockCrmApi, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        signature = inspect.signature(func)
        params = [
            param_name
            for param_name, param in signature.parameters.items()
            if param_name != "self"
        ]
        doc = inspect.getdoc(func) or ""
        summary = doc.strip().split("\n")[0] if doc else "No description provided."
        entries.append(f"{name}({', '.join(params)}) – {summary}")
    entries.sort()
    return entries


def _summarise_schema(schema: Mapping[str, Any], *, limit: int = 12) -> List[str]:
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    summaries: List[str] = []
    for entity_name, entity_schema in properties.items():
        if len(summaries) >= limit:
            break
        entity_props = entity_schema.get("properties", {}) if isinstance(entity_schema, Mapping) else {}
        required = entity_schema.get("required", []) if isinstance(entity_schema, Mapping) else []
        enums = []
        for field_name, field_schema in entity_props.items():
            if not isinstance(field_schema, Mapping):
                continue
            enum_values = field_schema.get("enum")
            if enum_values:
                enums.append(f"{field_name}: {', '.join(map(str, enum_values[:5]))}")
        required_str = ", ".join(required) if required else "none"
        enum_str = "; ".join(enums) if enums else "no enumerations"
        summaries.append(f"{entity_name}: required [{required_str}]; enums -> {enum_str}")
    return summaries


def _top_task_summaries(tasks: Sequence[Mapping[str, Any]], *, limit: int = 5) -> List[str]:
    ranked: List[tuple[int, str]] = []
    for row in tasks:
        name = str(row.get("Task Description") or row.get("task") or "").strip()
        if not name:
            continue
        count_value = row.get("Count") or row.get("count") or row.get("Frequency")
        try:
            count = int(float(count_value)) if count_value not in (None, "") else 0
        except (TypeError, ValueError):
            count = 0
        ranked.append((count, name))
    ranked.sort(key=lambda item: item[0], reverse=True)
    summaries: List[str] = []
    for count, name in ranked[:limit]:
        if count > 0:
            summaries.append(f"{name} ({count})")
        else:
            summaries.append(name)
    return summaries

try:  # pragma: no cover - optional dependency handled at runtime
    from litellm import completion
except ImportError:  # pragma: no cover - optional dependency handled in class initialisation
    completion = None  # type: ignore


class AgentError(Exception):
    """Base exception for agent-related failures."""


class AgentInvocationError(AgentError):
    """Raised when an agent cannot produce a tool call."""


class AgentResponseFormatError(AgentError):
    """Raised when an agent returns an unexpected payload."""


@dataclass
class AgentToolCall:
    """Structured representation of a tool invocation proposed by an agent."""

    tool_name: str
    arguments: Dict[str, Any]
    raw_response: Optional[str] = None
    token_usage: Dict[str, int] = field(default_factory=dict)
    reasoning: Optional[str] = None


@dataclass
class AgentTurnContext:
    """Context passed to agents when generating a tool call."""

    conversation: Conversation
    turn: ConversationTurn
    prior_turns: Sequence[ConversationTurn]
    previous_results: Mapping[int, Dict[str, Any]]
    expected_arguments: Dict[str, Any]


class ConversationAgent:
    """Abstract base class for conversation-aware agents."""

    provider_name: str = "mock"
    model_name: str = "ground_truth"

    def tool_call(self, context: AgentTurnContext) -> AgentToolCall:  # pragma: no cover - interface
        """Return the tool call that should be executed for ``context.turn``."""
        raise NotImplementedError


class MockAgent(ConversationAgent):
    """Agent that simply replays the ground-truth tool calls from the dataset."""

    def tool_call(self, context: AgentTurnContext) -> AgentToolCall:
        turn = context.turn
        return AgentToolCall(
            tool_name=turn.expected_tool,
            arguments=context.expected_arguments,
        )


class LiteLLMChatAgent(ConversationAgent):
    """LiteLLM-backed agent that generates tool calls via chat completion APIs."""

    def __init__(
        self,
        *,
        model_name: str,
        provider_name: str,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
        schema_path: Optional[Path] = None,
        tasks_path: Optional[Path] = None,
        tool_catalog_limit: int = 10,
    ) -> None:
        if completion is None:  # pragma: no cover - dependency guard
            raise ImportError(
                "litellm is required to use LiteLLMChatAgent; install the 'litellm' package."
            )

        self.model_name = model_name
        self.provider_name = provider_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        schema = _load_schema(str(schema_path or _DEFAULT_SCHEMA_PATH))
        tasks = _load_tasks(str(tasks_path or _DEFAULT_TASKS_PATH))
        self._schema_summary = _summarise_schema(schema)
        self._top_tasks = _top_task_summaries(tasks)
        self._tool_catalog = _tool_catalog()[:tool_catalog_limit] or ["No tool metadata available."]
        self._system_prompt = self._build_system_prompt()

    # ------------------------------------------------------------------
    def tool_call(self, context: AgentTurnContext) -> AgentToolCall:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": json.dumps(self._build_user_payload(context), ensure_ascii=False),
            },
        ]
        response = self._invoke_model(messages)
        return self._parse_response(response)

    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        schema_section = (
            "\n".join(f"- {line}" for line in self._schema_summary)
            if self._schema_summary
            else "- No schema description available."
        )
        tasks_section = ", ".join(self._top_tasks) if self._top_tasks else "Not supplied"
        tools_section = "\n".join(f"- {entry}" for entry in self._tool_catalog)

        return dedent(
            f"""
            You are the Student agent operating inside the Arc CRM sandbox. Execute exactly one CRM tool call per turn.
            Respond strictly with a single JSON object using this shape:
            {{
              "tool_name": "<tool identifier>",
              "arguments": {{ "<parameter>": <value>, ... }},
              "reasoning": "<optional concise rationale>"
            }}
            Do not include natural-language commentary outside the JSON object.

            CRM schema summary:
            {schema_section}

            High-frequency customer tasks: {tasks_section}

            Available CRM tools:
            {tools_section}
            """
        ).strip()

    def _build_user_payload(self, context: AgentTurnContext) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        for prior_turn in context.prior_turns:
            record = context.previous_results.get(prior_turn.turn_id, {})
            history.append(
                {
                    "turn_id": prior_turn.turn_id,
                    "user_utterance": prior_turn.user_utterance,
                    "tool_name": record.get("tool_name"),
                    "arguments": record.get("arguments"),
                    "result": record.get("result"),
                    "error": record.get("error"),
                    "success": record.get("success"),
                }
            )

        payload: Dict[str, Any] = {
            "conversation_id": context.conversation.conversation_id,
            "workflow_category": context.conversation.workflow_category,
            "complexity_level": context.conversation.complexity_level,
            "success_criteria": context.conversation.success_criteria,
            "history": history,
            "current_turn": {
                "turn_id": context.turn.turn_id,
                "user_utterance": context.turn.user_utterance,
                "expected_tool_hint": context.turn.expected_tool,
                "expected_arguments_hint": context.expected_arguments,
                "expect_success": context.turn.expect_success,
            },
        }
        if context.conversation.cumulative_context:
            payload["cumulative_context"] = context.conversation.cumulative_context
        return payload

    def _invoke_model(self, messages: List[Dict[str, Any]]) -> Mapping[str, Any]:
        assert completion is not None  # for mypy
        try:
            return completion(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
        except Exception as exc:  # pragma: no cover - dependency failure
            raise AgentInvocationError(
                f"LiteLLM failed for provider '{self.provider_name}' model '{self.model_name}': {exc}"
            ) from exc

    def _parse_response(self, response: Mapping[str, Any]) -> AgentToolCall:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentResponseFormatError("Model response did not include message content.") from exc

        text_content = self._normalise_content(content)
        payload = self._extract_json_dict(text_content)

        tool_name = payload.get("tool_name") or payload.get("tool")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise AgentResponseFormatError("Model response did not include a tool_name.")

        arguments = payload.get("arguments") or payload.get("args") or {}
        if not isinstance(arguments, dict):
            raise AgentResponseFormatError("Model response 'arguments' must be a JSON object.")

        reasoning = payload.get("reasoning") or payload.get("explanation")
        reasoning_text = reasoning.strip() if isinstance(reasoning, str) else None

        token_usage = self._normalise_token_usage(response.get("usage", {}))

        return AgentToolCall(
            tool_name=tool_name.strip(),
            arguments=arguments,
            raw_response=text_content,
            token_usage=token_usage,
            reasoning=reasoning_text,
        )

    @staticmethod
    def _normalise_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            pieces: List[str] = []
            for fragment in content:
                if isinstance(fragment, Mapping):
                    text = fragment.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
            if pieces:
                return "".join(pieces).strip()
        raise AgentResponseFormatError("Unsupported message content format returned by the model.")

    @staticmethod
    def _extract_json_dict(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or start >= end:
                raise AgentResponseFormatError("Model response is not valid JSON.")
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError as exc:
                raise AgentResponseFormatError("Model response is not valid JSON.") from exc

    @staticmethod
    def _normalise_token_usage(usage: Mapping[str, Any]) -> Dict[str, int]:
        tokens: Dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key) if isinstance(usage, Mapping) else None
            if value is None:
                continue
            try:
                tokens[key] = int(value)
            except (TypeError, ValueError):
                continue
        return tokens


class LiteLLMGPT4Agent(LiteLLMChatAgent):
    """LiteLLM agent configured for OpenAI GPT-4.1 variants."""

    provider_name = "openai"

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
        schema_path: Optional[Path] = None,
        tasks_path: Optional[Path] = None,
    ) -> None:
        default_model = os.getenv("CRM_BASELINE_GPT41_MODEL", "gpt-4.1-mini")
        super().__init__(
            model_name=model_name or default_model,
            provider_name="openai",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            schema_path=schema_path,
            tasks_path=tasks_path,
        )


class LiteLLMClaudeAgent(LiteLLMChatAgent):
    """LiteLLM agent configured for Claude Sonnet variants."""

    provider_name = "anthropic"

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
        schema_path: Optional[Path] = None,
        tasks_path: Optional[Path] = None,
    ) -> None:
        default_model = os.getenv("CRM_BASELINE_CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
        super().__init__(
            model_name=model_name or default_model,
            provider_name="anthropic",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            schema_path=schema_path,
            tasks_path=tasks_path,
        )


__all__ = [
    "AgentError",
    "AgentInvocationError",
    "AgentResponseFormatError",
    "AgentToolCall",
    "AgentTurnContext",
    "ConversationAgent",
    "MockAgent",
    "LiteLLMChatAgent",
    "LiteLLMGPT4Agent",
    "LiteLLMClaudeAgent",
]
