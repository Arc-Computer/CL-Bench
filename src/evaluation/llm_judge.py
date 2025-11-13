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

EVALUATION CRITERIA (balance goal achievement with process correctness):
- PRIMARY (70% weight): Did the agent accomplish the user's goal? If the tool executed successfully and returned the expected information, this contributes positively.
- SECONDARY (30% weight): Process correctness matters for production systems:
  * Tool selection: Should match expected tool or be a close functional equivalent (e.g., opportunity_details vs view_opportunity_details are equivalent, but client_search vs opportunity_search are not)
  * Argument accuracy: Arguments should be substantially correct (>70% match expected arguments). Minor differences in optional fields are acceptable, but required fields must match.
  * Approach appropriateness: The agent's approach should follow the expected workflow pattern. Significantly different approaches that achieve the goal may still pass but with lower scores.
- Fail conditions:
  * Agent misunderstood user intent
  * Used an invalid tool or wrong tool category
  * Execution failed
  * Significant argument mismatches (>50% of required arguments incorrect)
  * Tool selection is functionally different (e.g., search tool when create was expected)

Grade the agent's performance:
- Pass (score ≥ 0.8): Agent accomplished user intent AND used appropriate process. Tool name/argument differences are acceptable if they are minor variations (e.g., functionally equivalent tools, optional field differences) and execution succeeded.
- Fail (score < 0.8): Agent misunderstood intent, used an invalid/wrong tool, execution failed, or had significant process errors even if goal was achieved.

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

EVALUATION CRITERIA (balance goal achievement with response quality):
- PRIMARY (70% weight): Did the tool execute successfully and return the information the user requested?
- SECONDARY (30% weight): Response quality matters for user experience:
  * Response should address the user's question directly
  * Response should include key information from the tool result
  * Response should be complete and informative (not just placeholder text like "Searching...")
  * Empty responses are acceptable only if the tool result is self-explanatory and the user's question is fully answered by the result structure
- Pass if: Tool succeeded AND tool result contains requested information AND response adequately addresses user's question
- Fail if: Tool failed OR tool result doesn't contain requested information OR agent misunderstood request OR response is incomplete/missing key information

Instructions:
- Mark as Pass if the tool executed successfully, the tool result contains the requested information, AND the agent's response adequately communicates this information to the user.
- Brief responses are acceptable if they include the key information requested (e.g., "Found opportunity X with stage Y" is acceptable, but just "Searching..." is not).
- Mark as Fail if the tool failed, the tool result doesn't contain the requested information, the agent misunderstood the request, OR the response is incomplete/doesn't address the user's question.
- Empty responses should only pass if the tool result structure itself fully answers the user's question without additional explanation.

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
