"""Baseline harness for running CRM golden cases through LLM agents.

This module supports:
  * Running golden cases with real LLMs (Claude 4.5 Sonnet, GPT-4.1, etc.).
  * Executing cases in mock mode for deterministic testing.
  * Capturing pre/post CRM state, validator outcomes, and logging to JSONL.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

from .crm_sandbox import MockCrmApi
from .golden_cases import GOLDEN_CASES, GoldenCase
from .validators import CrmStateSnapshot, ValidationResult


# ---------------------------------------------------------------------------
# Tool call data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """Structured tool invocation returned by an agent."""

    tool_name: str
    arguments: Dict[str, Any]
    raw_response: str


@dataclass(frozen=True)
class EpisodeLog:
    """Serialized episode data written to JSONL."""

    case_id: str
    task: str
    timestamp: str
    provider: str
    model: str
    success: bool
    message: str
    tool_call: Dict[str, Any]
    agent_response: str
    validator_details: Optional[Dict[str, Any]]
    expected_tool: str
    expected_arguments: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "case_id": self.case_id,
                "task": self.task,
                "timestamp": self.timestamp,
                "provider": self.provider,
                "model": self.model,
                "success": self.success,
                "message": self.message,
                "tool_call": self.tool_call,
                "agent_response": self.agent_response,
                "validator_details": self.validator_details,
                "expected_tool": self.expected_tool,
                "expected_arguments": self.expected_arguments,
            },
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Agent protocol and implementations
# ---------------------------------------------------------------------------


class Agent(Protocol):
    """Interface for generating tool calls from golden-case prompts."""

    provider_name: str
    model_name: str

    def tool_call(self, case: GoldenCase, prompt: str) -> ToolCall:
        """Return the tool invocation suggested by the agent."""


class MockAgent:
    """Agent that returns ground-truth arguments; used for testing."""

    provider_name = "mock"
    model_name = "ground_truth"

    def tool_call(self, case: GoldenCase, prompt: str) -> ToolCall:  # noqa: D401 - signature satisfies protocol
        context: Dict[str, Any] = {}
        expected_args = case.expected_args(context)
        return ToolCall(case.expected_tool, expected_args, raw_response=json.dumps(expected_args))


class ClaudeAgent:
    """Anthropic Claude client wrapper."""

    provider_name = "anthropic"

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency required at runtime
            raise RuntimeError("anthropic package is required to use ClaudeAgent.") from exc
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def tool_call(self, case: GoldenCase, prompt: str) -> ToolCall:  # noqa: D401 - interface
        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text" and hasattr(block, "text")
        ).strip()
        if not text:
            raise ValueError("Claude response did not return textual tool instructions.")
        return _parse_tool_call(text)


class OpenAIAgent:
    """OpenAI GPT client wrapper."""

    provider_name = "openai"

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency required at runtime
            raise RuntimeError("openai package is required to use OpenAIAgent.") from exc
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def tool_call(self, case: GoldenCase, prompt: str) -> ToolCall:  # noqa: D401 - interface
        response = self._client.responses.create(
            model=self.model_name,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            input=[{"role": "user", "content": prompt}],
        )
        output = response.output_text if hasattr(response, "output_text") else response.output[0].content[0].text  # type: ignore[attr-defined]
        return _parse_tool_call(output)


# ---------------------------------------------------------------------------
# Prompt construction & parsing helpers
# ---------------------------------------------------------------------------


TASK_TOOL_BLURBS = {
    "create_new_client": (
        "Call `create_new_client(name: str, email: str, status: str, **optional_fields)` with the appropriate values."
    ),
    "create_new_opportunity": (
        "Call `create_new_opportunity(name: str, client_id: str, amount: float, stage: str, **optional_fields)`."
    ),
    "create_quote": (
        "Call `create_quote(opportunity_id: str, amount: float, status: str, **optional_fields)` to generate the quote."
    ),
    "upload_document": (
        "Call `upload_document(entity_type: str, entity_id: str, file_name: str, **optional_fields)` to attach the document."
    ),
    "modify_opportunity": (
        "Call `modify_opportunity(opportunity_id: str, updates: Dict[str, Any])`, supplying the fields to change."
    ),
}


def build_prompt(case: GoldenCase, context: Dict[str, Any]) -> str:
    """Create an instruction prompt for the agent."""
    header = (
        "You are Arc's CRM automation agent. Produce exactly one JSON object with keys "
        "`tool_name` and `arguments`. Do not call any tool other than the one specified. "
        "If data is missing, make reasonable business assumptions based on the utterance."
    )
    tool_instructions = TASK_TOOL_BLURBS.get(case.task, "")
    context_lines = _format_context(context)
    json_spec = (
        "Output format:\n"
        "```json\n"
        '{"tool_name": "<tool_name>", "arguments": {...}}\n'
        "```\n"
        "Ensure arguments strictly follow the tool signature."
    )
    prompt_parts = [
        header,
        "",
        f"User utterance:\n{case.utterance}",
        "",
        "Relevant CRM entities:",
        context_lines or "None; you may need to create or reference entities per the task.",
        "",
        f"Tool directive:\n{tool_instructions}",
        "",
        json_spec,
    ]
    return "\n".join(prompt_parts)


def _format_context(context: Dict[str, Any]) -> str:
    """Pretty-print context objects to surface IDs and metadata."""
    lines: List[str] = []
    client = context.get("client")
    if client:
        lines.append(
            f"- Client: {client.name} (client_id={client.client_id}, status={client.status}, email={client.email})"
        )
    opportunity = context.get("opportunity")
    if opportunity:
        lines.append(
            f"- Opportunity: {opportunity.name} (opportunity_id={opportunity.opportunity_id}, stage={opportunity.stage}, amount={opportunity.amount})"
        )
    quote = context.get("quote")
    if quote:
        lines.append(
            f"- Quote: {quote.quote_id} (status={quote.status}, amount={quote.amount}, opportunity_id={quote.opportunity_id})"
        )
    contract = context.get("contract")
    if contract:
        lines.append(
            f"- Contract: {contract.contract_id} (status={contract.status}, value={contract.value}, client_id={contract.client_id})"
        )
    if not lines:
        return ""
    return "\n".join(lines)


def _parse_tool_call(text: str) -> ToolCall:
    """Parse a JSON tool call from LLM output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Agent response is not valid JSON: {cleaned}") from exc
    if not isinstance(data, dict) or "tool_name" not in data or "arguments" not in data:
        raise ValueError(f"Agent response missing required keys: {data}")
    if not isinstance(data["arguments"], dict):
        raise ValueError("Agent arguments must be a JSON object.")
    return ToolCall(tool_name=data["tool_name"], arguments=data["arguments"], raw_response=text)


# ---------------------------------------------------------------------------
# Harness runner
# ---------------------------------------------------------------------------


@dataclass
class HarnessResult:
    """Aggregate results for a harness run."""

    success_count: int
    failure_count: int
    episodes: List[EpisodeLog]


class BaselineHarness:
    """Execute golden cases against a specified agent."""

    def __init__(
        self,
        agent: Agent,
        log_path: Union[str, Path],
        cases: Optional[Sequence[GoldenCase]] = None,
    ) -> None:
        self.agent = agent
        self.log_path = Path(log_path)
        self.cases: Sequence[GoldenCase] = cases or GOLDEN_CASES
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, mode: str = "agent") -> HarnessResult:
        """Run the selected cases; `mode='mock'` bypasses the agent."""
        episodes: List[EpisodeLog] = []
        successes = 0
        failures = 0

        with self.log_path.open("w", encoding="utf-8") as log_file:
            for case in self.cases:
                api = MockCrmApi()
                context = case.setup(api)

                expected_args = case.expected_args(context)
                prompt = build_prompt(case, context)

                if mode == "mock":
                    tool_call = ToolCall(case.expected_tool, expected_args, raw_response=json.dumps(expected_args))
                else:
                    tool_call = self.agent.tool_call(case, prompt)

                pre = CrmStateSnapshot.from_api(api)
                result = self._execute_tool(api, tool_call)
                post = CrmStateSnapshot.from_api(api)

                if not result.success:
                    validation = ValidationResult.fail(
                        f"Tool execution failed: {result.message}", details=result.details
                    )
                else:
                    validator_kwargs = case.validator_kwargs(context, tool_call.arguments)
                    validation = case.validator(pre, post, tool_call.arguments, **validator_kwargs)

                success = validation.success and tool_call.tool_name == case.expected_tool
                if success:
                    successes += 1
                else:
                    failures += 1

                episode = EpisodeLog(
                    case_id=case.case_id,
                    task=case.task,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    provider=self.agent.provider_name,
                    model=self.agent.model_name,
                    success=success,
                    message=validation.message if validation.success else validation.message,
                    tool_call={"tool_name": tool_call.tool_name, "arguments": tool_call.arguments},
                    agent_response=tool_call.raw_response,
                    validator_details=validation.details,
                    expected_tool=case.expected_tool,
                    expected_arguments=expected_args,
                )
                log_file.write(episode.to_json() + "\n")
                episodes.append(episode)

        return HarnessResult(success_count=successes, failure_count=failures, episodes=episodes)

    @staticmethod
    def _execute_tool(api: MockCrmApi, tool_call: ToolCall) -> ValidationResult:
        """Run the tool call against the MockCrmApi and capture immediate errors."""
        try:
            tool = getattr(api, tool_call.tool_name)
        except AttributeError:
            return ValidationResult.fail(f"Unknown tool '{tool_call.tool_name}'.")

        try:
            if tool_call.tool_name == "modify_opportunity":
                tool(tool_call.arguments["opportunity_id"], tool_call.arguments["updates"])
            else:
                tool(**tool_call.arguments)
        except Exception as exc:
            return ValidationResult.fail(str(exc))

        return ValidationResult.ok()


__all__ = ["BaselineHarness", "HarnessResult", "MockAgent", "ClaudeAgent", "OpenAIAgent", "build_prompt"]
