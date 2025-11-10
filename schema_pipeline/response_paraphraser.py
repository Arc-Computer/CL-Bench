from __future__ import annotations

import json
from typing import Any, Mapping

from litellm import completion


class ResponseParaphraser:
    """Lightweight wrapper that rewrites assistant turns using actual tool outcomes."""

    def __init__(self, *, model_name: str, generation_params: Mapping[str, Any] | None = None) -> None:
        self._model = model_name
        self._generation_params = dict(generation_params or {})

    def paraphrase(
        self,
        *,
        persona: Mapping[str, Any],
        user_utterance: str,
        turn_result: Mapping[str, Any],
    ) -> str:
        system_prompt = (
            f"You are a {persona.get('persona', 'CRM assistant')} with a {persona.get('formality', 'neutral')} "
            f"style and a {persona.get('tone', 'balanced')} tone. Always cite factual values exactly as provided."
        )

        tool_name = turn_result.get("expected_tool") or turn_result.get("tool_name") or "the tool"
        tool_args = json.dumps(_coerce(turn_result.get("expected_arguments") or turn_result.get("arguments") or {}), ensure_ascii=False)
        tool_result = json.dumps(_coerce(turn_result.get("result")), ensure_ascii=False)
        error = turn_result.get("error")
        user_text = user_utterance or "(no additional user prompt)"

        instructions = [
            "Rewrite the assistant reply based solely on these facts:",
            f"Tool executed: {tool_name} with arguments {tool_args}.",
            f"Tool result payload: {tool_result}.",
            "If the result is empty/null, state that plainly.",
            "If there was an error, acknowledge it and avoid pretending it succeeded.",
            "Do not invent IDs, names, or dates. Mention only the provided values.",
            "Respond with a single assistant message (no additional metadata).",
        ]
        if error:
            instructions.append(f"Error message: {error}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User said: '{user_text}'"},
            {"role": "user", "content": "\n".join(instructions)},
        ]

        response = completion(
            model=self._model,
            messages=messages,
            temperature=self._generation_params.get("temperature", 0.2),
            max_tokens=self._generation_params.get("max_tokens", 200),
        )
        content = response.choices[0].message.content if response.choices else ""
        return content.strip()


def _coerce(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except TypeError:
        return str(value)
