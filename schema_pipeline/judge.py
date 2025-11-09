from __future__ import annotations

from typing import Dict, List

import json

from bespokelabs import curator
from pydantic import BaseModel, Field

from .json_utils import ensure_dict
from .schema_utils import build_schema_context


class JudgeVerdict(BaseModel):
    score: float = Field(..., ge=0, le=5, description="Overall 0-5 quality score.")
    verdict: str = Field(..., description="pass/fail string")
    naturalness_notes: str = Field(..., description="Commentary on linguistic quality.")
    schema_alignment_notes: str = Field(..., description="Commentary on tool/schema correctness.")
    blocking_issues: List[str] = Field(default_factory=list)


class ConversationQualityJudge(curator.LLM):
    """LLM judge that acts like a CRM subject-matter expert."""

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
            "You are the final reviewer for a CRM conversation dataset. Score how well the conversation "
            "matches the workflow, adheres to schema rules, and sounds natural."
        )
        super().__init__(
            model_name=model_name,
            backend=backend,
            backend_params=backend_params,
            generation_params=generation_params,
            system_prompt=prompt,
        )

    def prompt(self, input: Dict) -> List[Dict[str, str]]:
        schema_context = build_schema_context(["Client", "Opportunity", "Quote", "Contract"])
        convo = input["conversation"]
        workflow_plan = input["workflow_plan"]
        arguments = input["arguments"]
        user_prompt = f"""
Schema snippet:
{schema_context}

Workflow plan: {workflow_plan}
Arguments: {arguments}
Conversation transcript:
{convo}

Score criteria (0-5):
1. Tool coverage + schema compliance
2. Natural conversational flow
3. Clear rationales / error messaging

Return pass if score >=4 AND no blocking issues.
Respond with JSON: {{"score": 0-5, "verdict": "pass|fail", "naturalness_notes": "...", "schema_alignment_notes": "...", "blocking_issues": ["..."]}}
""".strip()
        return [{"role": "user", "content": user_prompt}]

    def parse(self, input: Dict, response) -> Dict:
        verdict = ensure_dict(response)
        score = float(verdict.get("score", 0))
        blocking = verdict.get("blocking_issues") or []
        verdict["passes"] = score >= 4.0 and not blocking
        verdict["sample_id"] = input["sample_id"]
        verdict["task_name"] = input.get("task_name")
        return verdict
