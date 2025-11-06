"""LLM-based judge used to evaluate alternate agent approaches during baseline runs."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    from litellm import completion as _litellm_completion
except ImportError:  # pragma: no cover - degraded mode handled in class initialisation
    _litellm_completion = None


class LLMJudge:
    """Wrapper around a chat-completion model that can judge conversation turns."""

    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
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
Arguments: {json.dumps(expected_arguments, indent=2)}

AGENT'S ACTUAL APPROACH:
Tool: {agent_tool}
Arguments: {json.dumps(agent_arguments, indent=2)}
Result: {json.dumps(tool_result) if tool_result is not None else "N/A"}
Error: {tool_error or "None"}

RECENT CONVERSATION HISTORY:
{json.dumps(conversation_history, indent=2)}

EVALUATION CRITERIA:
- Did the agent understand the user's intent?
- Did the agent's approach accomplish the task (even if using different tools/arguments)?
- Is the agent's approach semantically equivalent to the expected approach?
- If the agent encountered an error, was it due to reasonable assumptions about CRM state?

Grade the agent's performance:
- Pass (score ≥ 0.7): Agent accomplished user intent, possibly via alternate valid path
- Fail (score < 0.7): Agent misunderstood intent or made an invalid tool choice

Respond in JSON format:
{{
  "pass": true or false,
  "score": 0.0 to 1.0,
  "rationale": "Brief explanation (1-2 sentences) of why this approach does/doesn't accomplish user intent"
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
