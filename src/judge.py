"""Golden-case review utilities using LLM-based feedback."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .crm_sandbox import MockCrmApi
from .golden_cases import GOLDEN_CASES, GoldenCase, summary as golden_summary


def _load_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _redact_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Replace identifier-like values with placeholders for readability."""
    redacted: Dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, dict):
            redacted[key] = _redact_arguments(value)
        elif isinstance(value, list):
            redacted[key] = [
                _redact_arguments(item) if isinstance(item, dict) else _mask_if_identifier(key, item) for item in value
            ]
        else:
            redacted[key] = _mask_if_identifier(key, value)
    return redacted


def _mask_if_identifier(key: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    identifier_keys = {"client_id", "contact_id", "opportunity_id", "quote_id", "contract_id", "entity_id"}
    if key in identifier_keys or key.endswith("_id"):
        return "<derived_id>"
    return value


def build_case_digest() -> str:
    """Create a human-readable summary of the golden cases."""
    lines: List[str] = []
    counts = golden_summary()
    lines.append("Golden-Case Coverage")
    for task, count in counts.items():
        lines.append(f"- {task}: {count} cases")
    lines.append("")

    for case in GOLDEN_CASES:
        api = MockCrmApi()
        context = case.setup(api)
        expected_args = _redact_arguments(case.expected_args(context))
        lines.append(f"{case.case_id} | {case.task}")
        lines.append(f"  Description: {case.description}")
        lines.append(f"  Expected tool: {case.expected_tool}")
        lines.append(f"  Expected arguments: {json.dumps(expected_args, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


def run_case_review(output_path: str = "artifacts/golden_case_review_gpt4.1.txt") -> str:
    """Request an external LLM to critique coverage and highlight gaps."""
    _load_env()
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai package is required to run the case review.") from exc

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    system_prompt = (
        "You are an expert evaluator supporting Arc's Workstream 2 for a synthetic CRM benchmark. "
        "Arc built a Pydantic-backed sandbox (Workstream 1) and a 50-scenario golden-case suite for the top five CRM tasks. "
        "Our goal now is to baseline Claude 4.5 Sonnet and GPT-4.1, capture failure modes, and ensure the cases truly represent "
        "customer pain points (state-modifying actions, relationship enforcement, enum validation, etc.). "
        "Review the provided golden-case digest and identify:\n"
        "  • coverage gaps or missing edge cases,\n"
        "  • ambiguous or brittle assumptions (e.g., casing, date formats, probability bounds),\n"
        "  • opportunities to mirror the customer's deterministic checks more faithfully.\n"
        "Return actionable feedback prioritized by impact. Suggest new scenarios or modifications when helpful."
    )
    user_prompt = build_case_digest()

    response = client.responses.create(
        model="gpt-4.1",
        temperature=0.2,
        max_output_tokens=1200,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    feedback = response.output_text if hasattr(response, "output_text") else response.output[0].content[0].text  # type: ignore[attr-defined]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(feedback, encoding="utf-8")
    return feedback


__all__ = ["build_case_digest", "run_case_review"]

