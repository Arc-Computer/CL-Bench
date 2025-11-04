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

from .crm_backend import DatabaseConfig, PostgresCrmBackend
from .crm_sandbox import MockCrmApi
from .golden_cases import GOLDEN_CASES, GoldenCase
from .validators import CrmStateSnapshot, ValidationResult, VerificationMode, get_task_verification_mode
from .verifier import VerifierRequest, VerifierResult, get_registered_verifier, ToolTrace as VerifierToolTrace


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
    expected_success: bool
    message: str
    tool_call: Dict[str, Any]
    agent_response: str
    validator_details: Optional[Dict[str, Any]]
    expected_tool: str
    expected_arguments: Dict[str, Any]
    verification_mode: str
    verifier_name: Optional[str]
    verifier_score: Optional[float]
    verifier_rationale: Optional[str]
    verifier_metadata: Optional[Dict[str, Any]]
    reward_breakdown: Dict[str, Any]
    learning_signals: Dict[str, Any]
    validator_metadata: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "case_id": self.case_id,
                "task": self.task,
                "timestamp": self.timestamp,
                "provider": self.provider,
                "model": self.model,
                "success": self.success,
                "expected_success": self.expected_success,
                "message": self.message,
                "tool_call": self.tool_call,
                "agent_response": self.agent_response,
                "validator_details": self.validator_details,
                "expected_tool": self.expected_tool,
                "expected_arguments": self.expected_arguments,
                "verification_mode": self.verification_mode,
                "verifier_name": self.verifier_name,
                "verifier_score": self.verifier_score,
                "verifier_rationale": self.verifier_rationale,
                "verifier_metadata": self.verifier_metadata,
                "reward_breakdown": self.reward_breakdown,
                "learning_signals": self.learning_signals,
                "validator_metadata": self.validator_metadata,
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
        "Call `create_new_client(name: str, email: str, status: str, **optional_fields)` and ensure status is exactly "
        "one of 'Active', 'Inactive', or 'Prospect'."
    ),
    "create_new_opportunity": (
        "Call `create_new_opportunity(name: str, client_id: str, amount: float, stage: str, **optional_fields)` with stage "
        "set to one of 'Prospecting', 'Qualification', 'Proposal', 'Negotiation', 'Closed-Won', or 'Closed-Lost'."
    ),
    "create_quote": (
        "Call `create_quote(opportunity_id: str, amount: float, status: str, **optional_fields)` using a status from "
        "'Draft', 'Sent', 'Approved', 'Rejected', or 'Canceled'."
    ),
    "upload_document": (
        "Call `upload_document(entity_type: str, entity_id: str, file_name: str, **optional_fields)` and pass entity_type as "
        "'Client', 'Opportunity', 'Quote', or 'Contract'; keep the file extension in the name."
    ),
    "modify_opportunity": (
        "Call `modify_opportunity(opportunity_id: str, updates: Dict[str, Any])`, using update keys such as 'stage', 'probability', "
        "'amount', 'close_date', 'notes', or 'owner', and respect the CRM enum rules for those fields."
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


def _parse_single_tool_call(data: Dict[str, Any], raw_response: str) -> ToolCall:
    """Parse a single tool call from a dictionary."""
    if not isinstance(data, dict) or "tool_name" not in data or "arguments" not in data:
        raise ValueError(f"Tool call missing required keys: {data}")
    if not isinstance(data["arguments"], dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return ToolCall(tool_name=data["tool_name"], arguments=data["arguments"], raw_response=raw_response)


def _parse_tool_calls(text: str) -> List[ToolCall]:
    """Parse one or more JSON tool calls from LLM output.
    
    Supports:
    - Single: {"tool_name": "...", "arguments": {...}}
    - Array: [{"tool_name": "...", ...}, ...]
    - Newline-separated JSON objects
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    
    # Try single JSON object
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "tool_name" in data:
            return [_parse_single_tool_call(data, text)]
        elif isinstance(data, list):
            return [_parse_single_tool_call(item, text) for item in data]
    except json.JSONDecodeError:
        pass
    
    # Try newline-separated objects
    tool_calls = []
    for line in cleaned.split('\n'):
        line = line.strip()
        if not line or line.startswith(('#', '//')):
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict) and "tool_name" in data:
                tool_calls.append(_parse_single_tool_call(data, line))
        except json.JSONDecodeError:
            continue
    
    if tool_calls:
        return tool_calls
    
    raise ValueError(f"Could not parse tool call(s): {text[:200]}...")


def _parse_tool_call(text: str) -> ToolCall:
    """Parse a single tool call (backwards compatibility wrapper)."""
    tool_calls = _parse_tool_calls(text)
    if len(tool_calls) != 1:
        raise ValueError(f"Expected single tool call but got {len(tool_calls)} calls")
    return tool_calls[0]


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
        backend: str = "mock",
        db_config: Optional[DatabaseConfig] = None,
        reset_database_each_case: Optional[bool] = None,
        enable_verifier: bool = False,
        verifier_name: Optional[str] = None,
        verifier_config: Optional[Mapping[str, Any]] = None,
        verifier_reward_weight: float = 0.0,
    ) -> None:
        self.agent = agent
        self.log_path = Path(log_path)
        self.cases: Sequence[GoldenCase] = cases or GOLDEN_CASES
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        backend_mode = backend.lower().strip()
        if backend_mode not in {"mock", "postgres"}:
            raise ValueError("backend must be either 'mock' or 'postgres'.")
        self._backend_mode = backend_mode
        if backend_mode == "postgres":
            self._db_backend = PostgresCrmBackend(db_config or DatabaseConfig.from_env())
        else:
            self._db_backend = None
        if reset_database_each_case is None:
            reset_database_each_case = backend_mode == "postgres"
        self._reset_database_each_case = bool(reset_database_each_case)
        self._verifier_enabled = bool(enable_verifier)
        self._default_verifier_name = verifier_name.strip() if verifier_name else None
        self._default_verifier_config: Dict[str, Any] = dict(verifier_config or {})
        self._verifier_reward_weight = float(max(0.0, min(1.0, verifier_reward_weight)))

    def run(self, mode: str = "agent") -> HarnessResult:
        """Run the selected cases; `mode='mock'` bypasses the agent."""
        episodes: List[EpisodeLog] = []
        successes = 0
        failures = 0

        with self.log_path.open("w", encoding="utf-8") as log_file:
            for case in self.cases:
                if self._backend_mode == "postgres":
                    if not self._db_backend:
                        raise RuntimeError("Postgres backend not initialized.")
                    backend = self._db_backend
                    backend.begin_session(reset=self._reset_database_each_case)
                else:
                    backend = MockCrmApi()

                try:
                    context = case.setup(backend)

                    expected_args = case.expected_args(context)
                    prompt = build_prompt(case, context)

                    if mode == "mock":
                        tool_call = ToolCall(case.expected_tool, expected_args, raw_response=json.dumps(expected_args))
                        tool_calls = [tool_call]
                        raw_response = json.dumps(expected_args)
                    else:
                        # Get tool call(s) from agent - parse raw response to support multi-tool
                        agent_tool_call = self.agent.tool_call(case, prompt)
                        raw_response = agent_tool_call.raw_response
                        # Parse potentially multiple tool calls from raw response
                        tool_calls = _parse_tool_calls(raw_response)
                        tool_call = tool_calls[0]  # Use first for backward compatibility with validation

                    pre = CrmStateSnapshot.from_backend(backend)
                    # Execute all tool calls sequentially (fail-fast on first error)
                    execution_result = self._execute_tools(backend, tool_calls)
                    post = CrmStateSnapshot.from_backend(backend)

                    tool_correct = tool_call.tool_name == case.expected_tool
                    validator_result_for_verifier: Optional[ValidationResult] = None

                    if case.expect_success:
                        if execution_result.success and tool_correct:
                            validator_kwargs = case.validator_kwargs(context, tool_call.arguments)
                            validation = case.validator(pre, post, tool_call.arguments, **validator_kwargs)
                            validator_result_for_verifier = validation
                            case_passed = validation.success
                            outcome_message = validation.message or "Validation succeeded."
                        else:
                            if not tool_correct:
                                failure_message = (
                                    f"Expected tool '{case.expected_tool}' but agent called '{tool_call.tool_name}'."
                                )
                                details = None
                            else:
                                failure_message = execution_result.message
                                details = execution_result.details
                            validator_result_for_verifier = ValidationResult.fail(failure_message, details)
                            case_passed = False
                            outcome_message = failure_message
                    else:
                        if not tool_correct:
                            failure_message = (
                                f"Expected tool '{case.expected_tool}' but agent called '{tool_call.tool_name}'."
                            )
                            validator_result_for_verifier = ValidationResult.fail(failure_message)
                            case_passed = False
                            outcome_message = failure_message
                        elif execution_result.success:
                            failure_message = "Expected failure but tool executed successfully."
                            validator_result_for_verifier = ValidationResult.fail(failure_message)
                            case_passed = False
                            outcome_message = failure_message
                        else:
                            substring_ok = (
                                case.expected_error_substring is None
                                or (case.expected_error_substring in execution_result.message)
                            )
                            state_unchanged = pre == post
                            details = dict(execution_result.details or {})
                            details["substring_match"] = substring_ok
                            details["state_unchanged"] = state_unchanged
                            case_passed = substring_ok and state_unchanged
                            outcome_message = execution_result.message
                            validator_result_for_verifier = ValidationResult(case_passed, execution_result.message, details)

                    if validator_result_for_verifier is None:
                        validator_result_for_verifier = ValidationResult.fail("Validator result unavailable.")
                    validator_details = validator_result_for_verifier.details

                    # Build tool traces for verifier (one per tool call)
                    tool_traces = tuple(
                        VerifierToolTrace(
                            step=idx + 1,
                            tool_name=tc.tool_name,
                            arguments=dict(tc.arguments),
                            execution_success=execution_result.success if idx == 0 else True,  # Only first can fail in fail-fast
                            validator_success=validator_result_for_verifier.success if idx == 0 else True,
                            message=validator_result_for_verifier.message if idx == 0 else "Tool executed successfully",
                        )
                        for idx, tc in enumerate(tool_calls)
                    )
                    tool_trace = tool_traces[0]  # Primary trace for backward compatibility
                    verifier_result: Optional[VerifierResult] = None
                    verifier_name_used: Optional[str] = None
                    if self._verifier_enabled:
                        selected_name = case.verifier_name or self._default_verifier_name
                        if selected_name:
                            verifier_config: Dict[str, Any] = dict(self._default_verifier_config)
                            if case.verifier_options:
                                verifier_config.update(case.verifier_options)
                            try:
                                verifier = get_registered_verifier(selected_name, verifier_config)
                            except ValueError as exc:
                                raise ValueError(
                                    f"Failed to initialize verifier '{selected_name}' for case '{case.case_id}': {exc}"
                                ) from exc
                            request = VerifierRequest(
                                case_id=case.case_id,
                                task=case.task,
                                utterance=case.utterance,
                                expect_success=case.expect_success,
                                expected_error_substring=case.expected_error_substring,
                                expected_tool=case.expected_tool,
                                expected_arguments=expected_args,
                                tool_traces=tool_traces,
                                final_response=raw_response,
                                validator_result=validator_result_for_verifier,
                                pre_state=pre,
                                post_state=post,
                            )
                            try:
                                verifier_result = verifier.evaluate(request)
                            except Exception as exc:  # pragma: no cover - defensive path
                                verifier_result = VerifierResult(
                                    score=0.0,
                                    rationale=f"Verifier '{selected_name}' raised an exception: {exc}",
                                    metadata={"exception": repr(exc)},
                                )
                            verifier_name_used = selected_name

                    if case_passed:
                        successes += 1
                    else:
                        failures += 1

                    reward_breakdown = self._build_reward_breakdown(
                        success=case_passed,
                        verifier_result=verifier_result,
                    )
                    learning_signals = self._build_learning_signals(
                        validator_result=validator_result_for_verifier,
                        verifier_result=verifier_result,
                        final_reward=reward_breakdown["final"],
                    )
                    validator_metadata = self._build_validator_metadata(
                        validator_result=validator_result_for_verifier,
                        tool_correct=tool_correct,
                        execution_success=execution_result.success,
                    )

                    verification_mode = get_task_verification_mode(case.task).value
                    episode = EpisodeLog(
                        case_id=case.case_id,
                        task=case.task,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        provider=self.agent.provider_name,
                        model=self.agent.model_name,
                        success=case_passed,
                        expected_success=case.expect_success,
                        message=outcome_message,
                        tool_call={"tool_name": tool_call.tool_name, "arguments": tool_call.arguments},
                        agent_response=raw_response,
                        validator_details=validator_details,
                        expected_tool=case.expected_tool,
                        expected_arguments=expected_args,
                        verification_mode=verification_mode,
                        verifier_name=verifier_name_used,
                        verifier_score=verifier_result.score if verifier_result else None,
                        verifier_rationale=verifier_result.rationale if verifier_result else None,
                        verifier_metadata=verifier_result.metadata if verifier_result else None,
                        reward_breakdown=reward_breakdown,
                        learning_signals=learning_signals,
                        validator_metadata=validator_metadata,
                    )
                    log_file.write(episode.to_json() + "\n")
                    episodes.append(episode)
                finally:
                    if self._backend_mode == "postgres":
                        backend.rollback_session()

        return HarnessResult(success_count=successes, failure_count=failures, episodes=episodes)

    def _build_reward_breakdown(self, *, success: bool, verifier_result: Optional[VerifierResult]) -> Dict[str, Any]:
        base_reward = 1.0 if success else 0.0
        final_reward = base_reward
        contribution = 0.0
        score = verifier_result.score if verifier_result else 0.0
        if verifier_result and self._verifier_reward_weight > 0.0:
            blended = (1.0 - self._verifier_reward_weight) * base_reward + self._verifier_reward_weight * score
            contribution = blended - base_reward
            final_reward = blended
        return {
            "base": float(base_reward),
            "shaping": {
                "enabled": False,
                "applied": False,
                "tool_match_bonus": 0.0,
                "partial_progress": 0.0,
                "delta": 0.0,
            },
            "verifier": {
                "enabled": bool(self._verifier_reward_weight > 0.0),
                "weight": self._verifier_reward_weight,
                "score": float(score),
                "contribution": float(contribution),
            },
            "final": float(final_reward),
        }

    @staticmethod
    def _build_learning_signals(
        *,
        validator_result: ValidationResult,
        verifier_result: Optional[VerifierResult],
        final_reward: float,
    ) -> Dict[str, Any]:
        student_score = 1.0 if validator_result.success else 0.0
        student_summary = validator_result.message or ""
        teacher_score = verifier_result.score if verifier_result else 0.0
        teacher_summary = verifier_result.rationale if verifier_result else ""
        reason = validator_result.message or ""
        return {
            "student": {"summary": student_summary, "score": float(student_score)},
            "teacher": {"summary": teacher_summary, "score": float(teacher_score)},
            "adapter_events": [],
            "reason": reason,
            "drift_notes": "",
            "reward_observation": float(final_reward),
        }

    @staticmethod
    def _build_validator_metadata(
        *,
        validator_result: ValidationResult,
        tool_correct: bool,
        execution_success: bool,
    ) -> Dict[str, Any]:
        if validator_result.success:
            category = "success"
        elif not tool_correct:
            category = "wrong_tool"
        elif not execution_success:
            category = "execution_failure"
        else:
            category = "validator_failure"
        return {
            "success": bool(validator_result.success),
            "message": validator_result.message or "",
            "details": dict(validator_result.details) if validator_result.details else None,
            "error_category": category,
        }

    @staticmethod
    def _execute_tool(api: Any, tool_call: ToolCall) -> ValidationResult:
        """Run a single tool call against the backend and capture immediate errors."""
        try:
            tool = getattr(api, tool_call.tool_name)
        except AttributeError:
            return ValidationResult.fail(f"Unknown tool '{tool_call.tool_name}'.")

        try:
            # All methods now support uniform calling via **kwargs
            tool(**tool_call.arguments)
        except Exception as exc:
            return ValidationResult.fail(str(exc))

        return ValidationResult.ok()

    @staticmethod
    def _execute_tools(
        backend: Union[MockCrmApi, PostgresCrmBackend],
        tool_calls: List[ToolCall]
    ) -> ValidationResult:
        """Execute multiple tool calls sequentially with fail-fast error handling.
        
        Stops on first failure and returns failure result. For research baseline data,
        we need clean pass/fail signals - continuing would mask partial failures.
        """
        if not tool_calls:
            return ValidationResult.fail("No tool calls provided")
        
        for idx, tool_call in enumerate(tool_calls):
            result = BaselineHarness._execute_tool(backend, tool_call)
            if not result.success:
                # Fail-fast: stop on first failure
                return ValidationResult.fail(
                    f"Tool '{tool_call.tool_name}' failed: {result.message}",
                    {"failed_tool": tool_call.tool_name, "tool_index": idx}
                )
        
        # All tools executed successfully
        return ValidationResult.ok("All tools executed successfully", {"tool_count": len(tool_calls)})


__all__ = ["BaselineHarness", "HarnessResult", "MockAgent", "ClaudeAgent", "OpenAIAgent", "build_prompt", "ToolCall", "_parse_tool_calls"]
