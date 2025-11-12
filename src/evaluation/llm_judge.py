"""LLM-based judge used to evaluate alternate agent approaches during baseline runs."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

try:  # pragma: no cover - optional dependency
    from litellm import completion as _litellm_completion
except ImportError:  # pragma: no cover - degraded mode handled in class initialisation
    _litellm_completion = None


class LLMJudge:
    """Wrapper around a chat-completion model that can judge conversation turns."""

    def __init__(
        self,
        *,
        model: str = "gpt-4.1",
        temperature: float = 0.0,
        max_tokens: int = 500,
        completion_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        """
        Args:
            model: Model identifier supplied to LiteLLM.
            temperature: Sampling temperature for the completion.
            max_tokens: Maximum completion tokens.
            completion_fn: Optional override for dependency injection (used in tests).
        """
        if completion_fn is not None:
            self._completion = completion_fn
        elif _litellm_completion is None:
            raise ImportError("litellm required for LLM judge; install the 'litellm' package.")
        else:
            self._completion = _litellm_completion

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _serialize_for_json(obj: Any) -> Any:
        """Serialize object for JSON, handling UUIDs in keys and values."""
        if isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, dict):
            return {str(k): LLMJudge._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [LLMJudge._serialize_for_json(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    # ------------------------------------------------------------------
    def judge_turn(
        self,
        *,
        user_utterance: str,
        agent_tool: str,
        agent_arguments: Dict[str, Any],
        tool_result: Optional[Any],
        tool_error: Optional[str],
        expected_tool: str,
        expected_arguments: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return the LLM's judgement for a single turn."""
        prompt = f"""You are evaluating a CRM agent's performance on a single conversational turn.

USER REQUEST:
{user_utterance}

EXPECTED APPROACH (ground truth from dataset):
Tool: {expected_tool}
Arguments: {json.dumps(self._serialize_for_json(expected_arguments), indent=2)}

AGENT'S ACTUAL APPROACH:
Tool: {agent_tool}
Arguments: {json.dumps(self._serialize_for_json(agent_arguments), indent=2)}
Result: {json.dumps(self._serialize_for_json(tool_result), indent=2) if tool_result is not None else "N/A"}
Error: {tool_error or "None"}

RECENT CONVERSATION HISTORY:
{json.dumps(self._serialize_for_json(conversation_history), indent=2)}

EVALUATION CRITERIA (prioritize goal achievement over exact matching):
- PRIMARY: Did the agent accomplish the user's goal? If the tool executed successfully and returned the expected information, this is a PASS.
- Did the agent understand the user's intent?
- Is the agent's approach functionally equivalent to the expected approach? (e.g., opportunity_details vs view_opportunity_details are equivalent)
- If the agent used a different but valid tool that accomplishes the same goal, that's acceptable.
- Only fail if the agent misunderstood intent, used an invalid tool, or the execution failed.

Grade the agent's performance:
- Pass (score ≥ 0.7): Agent accomplished user intent. Tool name/argument differences are acceptable if execution succeeded and goal was achieved.
- Fail (score < 0.7): Agent misunderstood intent, used an invalid tool, or execution failed.

Respond in JSON format:
{{
  "pass": true or false,
  "score": 0.0 to 1.0,
  "rationale": "Brief explanation (1-2 sentences) focusing on whether the user's goal was accomplished"
}}"""

        try:
            response = self._completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)

            return {
                "pass": bool(parsed.get("pass", False)),
                "score": float(parsed.get("score", 0.0)),
                "rationale": str(parsed.get("rationale", "")),
                "token_usage": {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                },
            }

        except Exception as exc:  # pragma: no cover - defensive guard
            # Fall back to a conservative failure
            return {
                "pass": False,
                "score": 0.0,
                "rationale": f"Judge error: {exc}",
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

    # ------------------------------------------------------------------
    def judge_response(
        self,
        *,
        user_utterance: str,
        expected_response: Dict[str, Any],
        agent_response: str,
        tool_result: Optional[Any],
        conversation_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate the assistant's natural-language reply for a turn."""
        expected_text = expected_response.get("text") or ""
        answers = expected_response.get("answers") or []
        requires_judge = expected_response.get("requires_judge", False)

        try:
            tool_result_json = json.dumps(self._serialize_for_json(tool_result), indent=2) if tool_result is not None else "null"
        except (TypeError, ValueError):
            tool_result_json = str(tool_result)

        prompt = f"""You are grading the assistant's natural-language reply for a CRM workflow.

USER REQUEST:
{user_utterance}

RECENT CONVERSATION HISTORY:
{json.dumps(self._serialize_for_json(conversation_history), indent=2)}

GROUND-TRUTH TOOL RESULT:
{tool_result_json}

REFERENCE RESPONSE (authoritative guidance):
{expected_text}

ACCEPTABLE ANSWERS:
{json.dumps(self._serialize_for_json(answers), indent=2)}

AGENT RESPONSE TO GRADE:
{agent_response or '[empty response]'}

EVALUATION CRITERIA (prioritize goal achievement):
- PRIMARY: Did the tool execute successfully and return the information the user requested?
- If the tool result contains the requested information, the agent has accomplished the goal, even if the response text is brief or incomplete.
- Pass if: Tool succeeded AND tool result contains the requested information (response text quality is secondary)
- Fail if: Tool failed OR tool result doesn't contain requested information OR agent misunderstood the request

Instructions:
- Mark as Pass if the tool executed successfully and the tool result contains the information the user requested, even if the agent's response text is brief or incomplete.
- Allow brief responses like "Searching..." if the tool result is available and contains the requested information.
- Mark as Fail only if the tool failed, the tool result doesn't contain the requested information, or the agent misunderstood the request.
- If the response is empty but tool succeeded and result contains requested info, mark as Pass (goal achieved).

Respond in JSON:
{{
  "pass": true or false,
  "score": 0.0 to 1.0,
  "rationale": "Brief justification focusing on whether the user's goal was accomplished via tool result"
}}"""

        try:
            response = self._completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            return {
                "pass": bool(parsed.get("pass", False)),
                "score": float(parsed.get("score", 0.0)),
                "rationale": str(parsed.get("rationale", "")),
                "token_usage": {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                },
                "requires_judge": requires_judge,
            }
        except Exception as exc:  # pragma: no cover - defensive guard
            return {
                "pass": False,
                "score": 0.0,
                "rationale": f"Judge error: {exc}",
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "requires_judge": requires_judge,
            }
