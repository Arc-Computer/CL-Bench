from __future__ import annotations

from typing import Dict, List, Optional

import json

from bespokelabs import curator
from pydantic import BaseModel, Field

from .json_utils import ensure_dict


class ConversationTurn(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Natural language utterance.")
    tool_call: Optional[Dict[str, str]] = Field(default=None, description="Optional structured tool call.")


class GeneratedConversation(BaseModel):
    task_name: str
    turns: List[ConversationTurn]
    summary: str = Field(..., description="Short recap useful for retrieval.")


class NaturalUtteranceGenerator(curator.LLM):
    """Generate natural conversations that execute the workflow using tool arguments."""

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
            "You are a seasoned CRM agent. Produce lively but concise dialogues that walk through the workflow. "
            "Keep the assistant grounded in the provided tool arguments."
        )
        super().__init__(
            model_name=model_name,
            backend=backend,
            backend_params=backend_params,
            generation_params=generation_params,
            system_prompt=prompt,
        )

    def prompt(self, input: Dict) -> List[Dict[str, str]]:
        task = input["task_name"]
        workflow_plan = input["workflow_plan"]
        if isinstance(workflow_plan, str):
            workflow_plan = json.loads(workflow_plan)
        workflow_steps = workflow_plan["success_path"]
        arguments = input["arguments"]
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = []
        raw_style = input.get("style_profile") or {}
        if isinstance(raw_style, str):
            try:
                style_profile = json.loads(raw_style)
            except json.JSONDecodeError:
                style_profile = {}
        else:
            style_profile = raw_style
        persona = style_profile.get("persona", "CRM project lead")
        formality = style_profile.get("formality", "neutral")
        tone = style_profile.get("tone", "balanced")
        step_count = max(len(workflow_steps), 2)
        tool_lines = []
        for arg in arguments:
            tool_lines.append(
                f"- {arg['tool_name']}: {arg['arguments']} (checks: {', '.join(arg.get('validation_notes', []))})"
            )
        retry_note = ""
        if input.get("prompt_variant", 0) > 0:
            retry_note = "\nReturn ONLY the JSON payload described below. Do not include extra narration."
        user_prompt = f"""
Task: {task}
Workflow steps: {workflow_steps}
Tool arguments:
{chr(10).join(tool_lines)}

Persona: You are a {persona}. Maintain a {formality} register and a {tone} tone throughout.

Write a conversation with exactly {step_count * 2} turns (user then assistant per step). Each assistant turn must explicitly mention the tool it is invoking and describe the reasoning before the call. When the assistant performs a tool call,
embed it in the turn as JSON like TOOL:{{"name": "...", "arguments": {{...}}}}. Ensure the tool order matches the workflow steps above.
Return JSON payload: {{"turns": [{{"role": "user|assistant", "content": "...", "tool_call": {{...}} }}], "task_name": "...", "summary": "..."}}.{retry_note}
""".strip()
        return [{"role": "user", "content": user_prompt}]

    def parse(self, input: Dict, response) -> Dict:
        payload = ensure_dict(response)
        return {
            "sample_id": input["sample_id"],
            "task_name": input["task_name"],
            "intent": input.get("intent"),
            "verification_mode": input.get("verification_mode"),
            "schema_context": input.get("schema_context"),
            "workflow_plan": input["workflow_plan"],
            "arguments": input["arguments"],
            "style_profile": input.get("style_profile"),
            "conversation": json.dumps(payload),
        }
