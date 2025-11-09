from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

import json

from bespokelabs import curator
from pydantic import BaseModel, Field

from .generated import TABLE_SCHEMAS
from .json_utils import ensure_dict
from .schema_utils import build_schema_context, summarize_table

TOOL_ENTITY_MAP = {
    "create_new_client": "Client",
    "create_new_opportunity": "Opportunity",
    "create_quote": "Quote",
    "create_contract": "Contract",
    "create_new_contact": "Contact",
    "modify_opportunity": "Opportunity",
    "upload_document": "Document",
    "add_note": "Note",
}

AUTO_GENERATED_FIELDS = {
    "client_id",
    "contact_id",
    "opportunity_id",
    "quote_id",
    "contract_id",
    "document_id",
    "note_id",
}


class ToolArgument(BaseModel):
    tool_name: str = Field(..., description="Name of the CRM tool being invoked.")
    arguments: Dict[str, str] = Field(..., description="Argument payload keyed by parameter.")
    references: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from argument -> provenance (e.g., lead persona, CRM entity).",
    )
    validation_notes: List[str] = Field(
        default_factory=list, description="Schema checks already satisfied by these arguments."
    )


class WorkflowArguments(BaseModel):
    tool_arguments: List[ToolArgument]


class SchemaCompliantArgumentGenerator(curator.LLM):
    """Generate schema-aligned arguments for each workflow step."""

    response_format = None

    def __init__(
        self,
        *,
        model_name: str,
        backend: str,
        backend_params: Dict | None = None,
        generation_params: Dict | None = None,
        system_prompt: Optional[str] = None,
    ):
        prompt = system_prompt or (
            "You are a senior CRM integration engineer. Given a workflow plan, generate arguments that satisfy "
            "all schema constraints. Use realistic values (US phone, corporate emails, ISO dates)."
        )
        super().__init__(
            model_name=model_name,
            backend=backend,
            backend_params=backend_params,
            generation_params=generation_params,
            system_prompt=prompt,
        )

    def prompt(self, input: Dict) -> List[Dict[str, str]]:
        workflow_plan = input["workflow_plan"]
        if isinstance(workflow_plan, str):
            workflow_plan = json.loads(workflow_plan)
        workflow_plan_steps = workflow_plan["success_path"]
        instructions = [
            "You must cover every tool in the plan. Use the schema to respect enums and required fields.",
            "Respond with JSON: {\"tool_arguments\": [{\"tool_name\": ..., \"arguments\": {...}, \"references\": {...}, \"validation_notes\": [\"...\"]}]}",
        ]
        if input.get("prompt_variant", 0) > 0:
            instructions.append("Previous response was invalid. Return strict JSON only; no prose.")
        step_blocks = []
        for index, step in enumerate(workflow_plan_steps, start=1):
            tool = step["tool_name"]
            entity = TOOL_ENTITY_MAP.get(tool)
            schema_excerpt = summarize_table(entity) if entity else build_schema_context()
            step_blocks.append(
                f"Step {index}: {tool}\n"
                f"Description: {step['description']}\n"
                f"Arguments needed: {', '.join(step['arguments_needed'])}\n"
                f"Schema excerpt:\n{schema_excerpt}\n"
            )
        user_prompt = "\n\n".join(step_blocks + instructions)
        return [{"role": "user", "content": user_prompt}]

    def parse(self, input: Dict, response: WorkflowArguments) -> Dict:
        payload = ensure_dict(response)
        tool_arguments = payload.get("tool_arguments") or []
        self._validate_against_schema(tool_arguments)
        self._ensure_unique_values(tool_arguments, input.get("sample_id", uuid4().hex))
        style_profile = input.get("style_profile")
        if isinstance(style_profile, dict):
            style_profile = json.dumps(style_profile)
        return {
            "sample_id": input["sample_id"],
            "task_name": input["task_name"],
            "intent": input.get("intent"),
            "verification_mode": input.get("verification_mode"),
            "schema_context": input.get("schema_context"),
            "workflow_plan": input["workflow_plan"],
            "style_profile": style_profile,
            "arguments": json.dumps(tool_arguments),
        }

    def _validate_against_schema(self, tool_arguments: List[Dict[str, Any]]) -> None:
        for entry in tool_arguments:
            tool = entry["tool_name"]
            entity = TOOL_ENTITY_MAP.get(tool)
            if not entity:
                continue
            schema = TABLE_SCHEMAS[entity]
            is_creation = tool.startswith("create")
            if not is_creation:
                continue
            required_fields = [
                field for field in schema.get("required", []) if field not in AUTO_GENERATED_FIELDS
            ]
            missing = [field for field in required_fields if field not in entry["arguments"]]
            if missing:
                raise ValueError(f"{tool} is missing required fields: {missing}")

    def _ensure_unique_values(self, tool_arguments: List[Dict[str, Any]], sample_id: str) -> None:
        for index, entry in enumerate(tool_arguments):
            tool = entry["tool_name"]
            if tool != "create_new_client":
                continue
            args = entry.get("arguments", {})
            email = args.get("email")
            if not isinstance(email, str) or not email:
                unique_email = f"client_{sample_id[:8]}_{index}@example.com"
            else:
                if "@" in email:
                    local, domain = email.split("@", 1)
                    unique_email = f"{local}+{sample_id[:8]}_{index}@{domain}"
                else:
                    unique_email = f"{email}_{sample_id[:8]}_{index}@example.com"
            args["email"] = unique_email.lower()
