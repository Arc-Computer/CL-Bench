"""Verifier abstractions for grading agent behavior.

This module defines a common request/response protocol that downstream
components (Gym environment, harness, continual-learning telemetry) can use
to plug in different verifier strategies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Protocol, Sequence

from .validators import CrmStateSnapshot, ValidationResult


@dataclass(frozen=True)
class ToolTrace:
    """Snapshot of a tool invocation captured during an episode."""

    step: int
    tool_name: Optional[str]
    arguments: Mapping[str, Any]
    execution_success: bool
    validator_success: bool
    message: str


@dataclass(frozen=True)
class VerifierRequest:
    """Inputs provided to a verifier for scoring."""

    case_id: str
    task: str
    utterance: str
    expect_success: bool
    expected_error_substring: Optional[str]
    expected_tool: str
    expected_arguments: Mapping[str, Any]
    tool_traces: Sequence[ToolTrace]
    final_response: Optional[str] = None
    validator_result: Optional[ValidationResult] = None
    pre_state: Optional[CrmStateSnapshot] = None
    post_state: Optional[CrmStateSnapshot] = None


@dataclass(frozen=True)
class VerifierResult:
    """Score emitted by a verifier with supporting rationale."""

    score: float
    rationale: str
    metadata: Optional[Dict[str, Any]] = None


class Verifier(Protocol):
    """Protocol that all verifier implementations must satisfy."""

    name: str

    def evaluate(self, request: VerifierRequest) -> VerifierResult:
        """Return a score describing how well the agent satisfied expectations."""


_VerifierFactory = Callable[[Mapping[str, Any]], Verifier]
_VERIFIER_REGISTRY: MutableMapping[str, _VerifierFactory] = {}


def register_verifier(name: str, factory: _VerifierFactory) -> None:
    """Register a verifier factory under the provided name."""
    key = name.strip().lower()
    if not key:
        raise ValueError("Verifier name must be a non-empty string.")
    if key in _VERIFIER_REGISTRY:
        raise ValueError(f"Verifier '{name}' is already registered.")
    _VERIFIER_REGISTRY[key] = factory


def get_registered_verifier(name: str, config: Optional[Mapping[str, Any]] = None) -> Verifier:
    """Instantiate a verifier from the registry."""
    key = name.strip().lower()
    try:
        factory = _VERIFIER_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"Unknown verifier '{name}'. Available: {sorted(_VERIFIER_REGISTRY)}") from exc
    return factory(dict(config or {}))


def list_registered_verifiers() -> Sequence[str]:
    """Return the names of all registered verifiers."""
    return tuple(sorted(_VERIFIER_REGISTRY))


class StructuredVerifier:
    """Verifier that evaluates tool usage against expected structure."""

    name = "structured"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._weights = self._parse_weights(config.get("weights"))

    @staticmethod
    def _parse_weights(raw: Optional[Mapping[str, Any]]) -> Dict[str, float]:
        default = {
            "presence": 0.1,
            "tool_match": 0.35,
            "arguments": 0.35,
            "state": 0.2,
        }
        if not raw:
            return default
        weights = {}
        for key, value in default.items():
            weights[key] = float(raw.get(key, value))
        total = sum(weights.values())
        if total <= 0:
            return default
        return {key: weights[key] / total for key in weights}

    def evaluate(self, request: VerifierRequest) -> VerifierResult:
        traces = list(request.tool_traces)
        metadata: Dict[str, Any] = {"components": {}}
        reasons: list[str] = []

        if not traces:
            return VerifierResult(
                score=0.0,
                rationale="No tool invocations recorded; unable to grade structured output.",
                metadata=metadata,
            )

        final_trace = traces[-1]

        # Presence component.
        presence_score = 1.0 if final_trace.tool_name else 0.0
        metadata["components"]["presence"] = presence_score
        if presence_score < 1.0:
            reasons.append("Final step did not include a tool invocation.")

        # Tool match component.
        tool_match_score = 1.0 if final_trace.tool_name == request.expected_tool else 0.0
        metadata["components"]["tool_match"] = tool_match_score
        if tool_match_score < 1.0:
            reasons.append(
                f"Expected tool '{request.expected_tool}' but observed '{final_trace.tool_name or 'none'}'."
            )

        # Argument component.
        argument_score = self._score_arguments(request.expected_arguments, final_trace.arguments)
        metadata["components"]["arguments"] = argument_score
        if argument_score < 1.0:
            missing = self._diff_expected_arguments(request.expected_arguments, final_trace.arguments)
            if missing:
                formatted = ", ".join(f"{key}={value!r}" for key, value in missing.items())
                reasons.append(f"Argument mismatch for fields: {formatted}.")

        # State component.
        state_score, state_reason = self._score_state(request, final_trace)
        metadata["components"]["state"] = state_score
        if state_reason:
            reasons.append(state_reason)

        score = (
            presence_score * self._weights["presence"]
            + tool_match_score * self._weights["tool_match"]
            + argument_score * self._weights["arguments"]
            + state_score * self._weights["state"]
        )
        score = max(0.0, min(1.0, round(score, 4)))

        rationale = " ".join(reasons) if reasons else "Tool invocation matched expected structure."
        return VerifierResult(score=score, rationale=rationale, metadata=metadata)

    @staticmethod
    def _score_arguments(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> float:
        if not expected:
            return 1.0
        if not actual:
            return 0.0
        missing = StructuredVerifier._diff_expected_arguments(expected, actual)
        if missing:
            return 0.0
        return 1.0

    @staticmethod
    def _diff_expected_arguments(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> Dict[str, Any]:
        diff: Dict[str, Any] = {}
        for key, value in expected.items():
            if key not in actual:
                diff[key] = value
            elif actual[key] != value:
                diff[key] = value
        return diff

    def _score_state(self, request: VerifierRequest, trace: ToolTrace) -> tuple[float, Optional[str]]:
        validator = request.validator_result
        if request.expect_success:
            if validator and validator.success:
                return 1.0, None
            if validator and not validator.success:
                return 0.0, validator.message or "Validator reported failure."
            if trace.execution_success:
                return 0.5, "Tool executed but validator outcome unavailable."
            return 0.0, "Tool execution failed; expected success."
        # Negative case handling.
        pre_state = request.pre_state
        post_state = request.post_state
        no_change = (pre_state is not None and post_state is not None and pre_state == post_state)
        if validator and not validator.success and no_change:
            return 1.0, None
        if not no_change:
            return 0.0, "State changed in a negative test; expected no side effects."
        if validator and validator.success:
            return 0.0, "Validator marked the case as success but failure was expected."
        if trace.execution_success:
            return 0.0, "Tool executed successfully but failure was expected."
        return 0.5, "Observed failure without full validator details."


def _structured_factory(config: Mapping[str, Any]) -> Verifier:
    return StructuredVerifier(config)


register_verifier(StructuredVerifier.name, _structured_factory)


class LlmJudgeVerifier:
    """Verifier that prompts an LLM judge for qualitative grading."""

    name = "llm_judge"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._provider = str(config.get("provider", "anthropic")).strip().lower()
        if self._provider not in {"anthropic", "openai"}:
            raise ValueError("LlmJudgeVerifier provider must be either 'anthropic' or 'openai'.")
        self._model = str(
            config.get(
                "model",
                "claude-sonnet-4-5-20250929" if self._provider == "anthropic" else "gpt-4.1",
            )
        )
        self._temperature = float(config.get("temperature", 0.0))
        self._max_output_tokens = int(config.get("max_output_tokens", 256))
        self._client = config.get("client")
        self._api_key = config.get("api_key")
        self._prompt_preamble = config.get(
            "prompt_preamble",
            (
                "You are an impartial grading assistant. Score the agent's work between 0 and 1. "
                "0 means the agent failed or hallucinated; 1 means the agent fully satisfied the task."
            ),
        )
        self._mock_response = config.get("mock_response")

    def evaluate(self, request: VerifierRequest) -> VerifierResult:
        prompt = self._build_prompt(request)
        raw = self._obtain_response(prompt)
        parsed_score, parsed_reason = self._parse_response(raw)
        metadata = {
            "provider": self._provider,
            "model": self._model,
            "raw_response": raw,
        }
        return VerifierResult(score=parsed_score, rationale=parsed_reason, metadata=metadata)

    def _build_prompt(self, request: VerifierRequest) -> str:
        traces_lines = []
        for trace in request.tool_traces:
            arguments_json = json.dumps(trace.arguments, sort_keys=True)
            traces_lines.append(
                f"- step={trace.step} tool={trace.tool_name or 'none'} success={trace.execution_success} "
                f"validator_success={trace.validator_success} args={arguments_json} msg={trace.message}"
            )
        traces_text = "\n".join(traces_lines) or "- No tool invocations recorded."

        expected_args = json.dumps(request.expected_arguments, sort_keys=True)
        validator_summary = ""
        if request.validator_result:
            validator_summary = json.dumps(
                {
                    "success": request.validator_result.success,
                    "message": request.validator_result.message,
                    "details": request.validator_result.details,
                },
                sort_keys=True,
            )
        else:
            validator_summary = "null"

        sections = [
            self._prompt_preamble,
            "",
            f"Case ID: {request.case_id}",
            f"Task: {request.task}",
            f"User utterance: {request.utterance}",
            f"Expected success: {request.expect_success}",
            f"Expected tool: {request.expected_tool}",
            f"Expected arguments: {expected_args}",
            f"Expected error substring: {request.expected_error_substring!r}",
            "",
            "Tool history:",
            traces_text,
            "",
            f"Validator result: {validator_summary}",
            "",
            "Final agent response:",
            request.final_response or "(none provided)",
            "",
            (
                "Respond with strict JSON containing keys `score` (float between 0 and 1) "
                "and `reason` (concise explanation). Example:\n"
                '{"score": 0.75, "reason": "Partial match with missing amount field."}'
            ),
        ]
        return "\n".join(sections)

    def _obtain_response(self, prompt: str) -> str:
        if self._mock_response is not None:
            return str(self._mock_response)
        if self._provider == "anthropic":
            client = self._resolve_anthropic_client()
            response = client.messages.create(  # type: ignore[call-arg]
                model=self._model,
                max_tokens=self._max_output_tokens,
                temperature=self._temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text_blocks = [
                getattr(block, "text", "")
                for block in getattr(response, "content", [])
                if getattr(block, "type", None) == "text"
            ]
            output = "".join(text_blocks).strip()
            if not output and hasattr(response, "content"):
                output = json.dumps(getattr(response, "content"))
            return output
        client = self._resolve_openai_client()
        response = client.responses.create(
            model=self._model,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            input=[{"role": "user", "content": prompt}],
        )
        if hasattr(response, "output_text"):
            return str(response.output_text).strip()
        # Fallback for older SDK shapes.
        output = getattr(response, "output", None)
        if isinstance(output, list) and output:
            first = output[0]
            if hasattr(first, "content") and first.content:
                content_item = first.content[0]
                text = getattr(content_item, "text", "")
                if text:
                    return str(text).strip()
        return str(response)

    def _resolve_anthropic_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency required at runtime
            raise RuntimeError("anthropic package is required for LlmJudgeVerifier with provider='anthropic'.") from exc
        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY must be set for LlmJudgeVerifier.")
        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _resolve_openai_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency required at runtime
            raise RuntimeError("openai package is required for LlmJudgeVerifier with provider='openai'.") from exc
        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set for LlmJudgeVerifier.")
        self._client = OpenAI(api_key=api_key)
        return self._client

    @staticmethod
    def _parse_response(raw: str) -> tuple[float, str]:
        raw = raw.strip()
        if not raw:
            return 0.0, "LLM judge returned an empty response."
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return 0.0, f"Failed to parse judge response as JSON: {raw[:200]}"
        score = data.get("score")
        reason = str(data.get("reason") or data.get("rationale") or "").strip()
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        score_value = max(0.0, min(1.0, score_value))
        if not reason:
            reason = "LLM judge did not provide a rationale."
        return score_value, reason


def _llm_factory(config: Mapping[str, Any]) -> Verifier:
    return LlmJudgeVerifier(config)


register_verifier(LlmJudgeVerifier.name, _llm_factory)


__all__ = [
    "ToolTrace",
    "Verifier",
    "VerifierRequest",
    "VerifierResult",
    "StructuredVerifier",
    "LlmJudgeVerifier",
    "get_registered_verifier",
    "list_registered_verifiers",
    "register_verifier",
]
