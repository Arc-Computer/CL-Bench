from __future__ import annotations

from typing import Any, Dict, List

import json

from bespokelabs import curator
from pydantic import BaseModel, Field

from .json_utils import ensure_dict
from .schema_utils import build_schema_context

ALLOWED_TOOLS = [
    "create_new_client",
    "client_search",
    "modify_client",
    "create_new_contact",
    "contact_search",
    "modify_contact",
    "create_new_opportunity",
    "opportunity_search",
    "modify_opportunity",
    "create_quote",
    "quote_search",
    "modify_quote",
    "create_contract",
    "contract_search",
    "upload_document",
    "add_note",
    "summarize_opportunities",
]

TOOL_ALIAS_MAP = {
    "clientsearch": "client_search",
    "opportunity_create": "create_new_opportunity",
    "opportunity_new": "create_new_opportunity",
    "opportunitysearch": "opportunity_search",
    "quote_create": "create_quote",
    "quote_new": "create_quote",
    "quotesearch": "quote_search",
    "contract_create": "create_contract",
    "contract_new": "create_contract",
    "contractsearch": "contract_search",
    "modifyopportunity": "modify_opportunity",
    "modifyclient": "modify_client",
}


def _normalize_tool_name(raw: str) -> str:
    slug = (raw or "").strip().lower().replace(" ", "_")
    return TOOL_ALIAS_MAP.get(slug, slug)


WORKFLOW_JSON_EXAMPLE = json.dumps(
    {
        "task": "CREATE NEW CLIENT",
        "rationale": "Ensure the client does not already exist, then create it, then verify persistence.",
        "success_path": [
            {
                "tool_name": "client_search",
                "arguments_needed": ["name"],
                "description": "Check whether the client already exists.",
                "validation_hints": "Returns an empty list when the client does not exist."
            },
            {
                "tool_name": "create_new_client",
                "arguments_needed": ["name", "email", "status"],
                "description": "Create the new client with the provided metadata.",
                "validation_hints": "Status must be one of Active/Prospect/Inactive."
            },
            {
                "tool_name": "client_search",
                "arguments_needed": ["name"],
                "description": "Confirm the newly created client is discoverable.",
                "validation_hints": "Should return the client_id produced earlier."
            },
        ],
        "failure_checks": [
            {
                "tool_name": "create_new_client",
                "arguments_needed": ["email"],
                "description": "Handle duplicate email constraint violations.",
                "validation_hints": "If the email already exists, surface a descriptive error."
            }
        ],
    },
    indent=2,
)


class WorkflowStep(BaseModel):
    """Single tool call within a workflow."""

    tool_name: str = Field(..., description="CRM API method, e.g. create_new_client.")
    arguments_needed: List[str] = Field(..., description="Ordered list of required arguments.")
    description: str = Field(..., description="Natural language summary of what the tool accomplishes.")
    validation_hints: str = Field(..., description="Key schema constraints or business rules.")


class WorkflowPlan(BaseModel):
    """Structured workflow plan (happy + guarded path)."""

    task: str = Field(..., description="High-level CRM task (e.g., CREATE NEW OPPORTUNITY).")
    rationale: str = Field(..., description="Reason about selected sequence and schema considerations.")
    success_path: List[WorkflowStep] = Field(..., description="Ordered tool calls that should succeed.")
    failure_checks: List[WorkflowStep] = Field(..., description="Optional guardrails to validate data quality.")


class WorkflowSequenceGenerator(curator.LLM):
    """Curator block that expands weighted tasks into concrete tool sequences."""

    response_format = None

    def __init__(
        self,
        *,
        model_name: str,
        backend: str,
        backend_params: Dict | None = None,
        generation_params: Dict | None = None,
        system_prompt: str | None = None,
    ):
        prompt = system_prompt or (
            "You are a staff-level CRM systems architect. Produce reliable tool sequences that obey the schema "
            "and enforce data integrity. Always reason about why each tool is required."
        )
        super().__init__(
            model_name=model_name,
            backend=backend,
            backend_params=backend_params,
            generation_params=generation_params,
            system_prompt=prompt,
        )

    def prompt(self, input: Dict) -> List[Dict[str, str]]:
        schema_context = input.get("schema_context") or build_schema_context()
        task_name = input["task_name"]
        intent = input.get("intent", "")
        verification_mode = input.get("verification_mode", "")
        phrasings = input.get("sample_phrasings", "")
        sub_actions = ", ".join(input.get("sub_actions", []))

        retry_note = ""
        if input.get("prompt_variant", 0) > 0:
            retry_note = "\nPrevious response was invalid. Return ONLY the JSON schema below."

        allowed_tools_text = ", ".join(sorted(ALLOWED_TOOLS))

        user_prompt = f"""
Task: {task_name}
Intent: {intent}
Verification mode: {verification_mode}
Schema context:
{schema_context}

Typical customer phrasing:
{phrasings}

Related sub-actions: {sub_actions}

Instructions:
- Return 2-4 steps in the success path, each referencing a CRM tool from this allowlist: {allowed_tools_text} (use exact snake_case names).
- Add at least one failure check to catch schema violations (missing FK, invalid enum, etc.).
- Prefer minimum-token descriptions that still capture the constraint.
{retry_note}
Respond ONLY with valid JSON shaped like:
{{
  "task": "...",
  "rationale": "...",
  "success_path": [
    {{"tool_name": "...", "arguments_needed": ["field"], "description": "...", "validation_hints": "..."}}
  ],
  "failure_checks": [
    {{"tool_name": "...", "arguments_needed": ["field"], "description": "...", "validation_hints": "..."}}
  ]
}}

Example JSON:
{WORKFLOW_JSON_EXAMPLE}
""".strip()

        return [
            {"role": "user", "content": user_prompt},
        ]

    def parse(self, input: Dict, response) -> Dict:
        payload = ensure_dict(response)

        def _normalize_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            normalized: List[Dict[str, Any]] = []
            for step in steps:
                raw_name = step.get("tool_name", "")
                normalized_name = _normalize_tool_name(raw_name)
                if normalized_name not in ALLOWED_TOOLS:
                    raise ValueError(f"Unsupported tool '{raw_name}' in workflow plan. Allowed tools: {sorted(ALLOWED_TOOLS)}")
                normalized.append({**step, "tool_name": normalized_name})
            return normalized

        payload["success_path"] = _normalize_steps(payload.get("success_path") or [])
        payload["failure_checks"] = _normalize_steps(payload.get("failure_checks") or [])
        style_profile = input.get("style_profile")
        if isinstance(style_profile, dict):
            style_profile = json.dumps(style_profile)
        return {
            "sample_id": input["sample_id"],
            "task_name": input["task_name"],
            "intent": input.get("intent"),
            "verification_mode": input.get("verification_mode"),
            "schema_context": input.get("schema_context"),
            "sample_phrasings": input.get("sample_phrasings"),
            "sub_actions": input.get("sub_actions"),
            "style_profile": style_profile,
            "workflow_plan": json.dumps(payload),
        }
