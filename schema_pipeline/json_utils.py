from __future__ import annotations

import json
from typing import Any, Dict


def ensure_dict(response: Any) -> Dict[str, Any]:
    """Ensure the model response is parsed into a dictionary."""
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        text = response.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines()]
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Expected JSON response but received: {text[:200]}") from exc
    raise TypeError(f"Unsupported response type {type(response)}; expected str or dict")
