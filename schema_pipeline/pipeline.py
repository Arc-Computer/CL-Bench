from __future__ import annotations

import json
import random
from uuid import uuid4
from typing import Any, Dict, List, Optional

from bespokelabs.curator.client import Client
from bespokelabs.curator.types.curator_response import CuratorResponse

from .arguments import SchemaCompliantArgumentGenerator
from .config import PipelineConfig
from .judge import ConversationQualityJudge
from .schema_utils import build_schema_context
from .task_sampler import TaskSampler
from .utterances import NaturalUtteranceGenerator
from .workflow import WorkflowSequenceGenerator

STYLE_PRESETS = [
    {"persona": "enterprise account executive", "formality": "formal", "tone": "concise"},
    {"persona": "startup customer success lead", "formality": "casual", "tone": "supportive"},
    {"persona": "solutions consultant", "formality": "consultative", "tone": "analytical"},
    {"persona": "implementation project manager", "formality": "formal", "tone": "detailed"},
    {"persona": "sales development rep", "formality": "casual", "tone": "energetic"},
]


class SchemaFirstPipeline:
    """End-to-end orchestration for schema-first conversation generation."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.task_sampler = TaskSampler(self.config.tasks_csv)
        self.schema_context = build_schema_context()
        self.viewer_client = Client(hosted=self.config.gemini.viewer_enabled)
        self.viewer_client.create_session(
            {
                "project": "schema-first-crm",
                "notes": "Option B pipeline",
            },
            session_id=self.config.viewer_session_id,
        )

        gemini = self.config.gemini
        self.workflow_generator = WorkflowSequenceGenerator(
            model_name=gemini.workflow_model,
            backend=gemini.backend,
            backend_params=gemini.backend_params,
            generation_params={**gemini.generation_params, "temperature": 0.2},
        )
        self.argument_generator = SchemaCompliantArgumentGenerator(
            model_name=gemini.argument_model,
            backend=gemini.backend,
            backend_params=gemini.backend_params,
            generation_params=gemini.generation_params,
        )
        self.utterance_generator = NaturalUtteranceGenerator(
            model_name=gemini.utterance_model,
            backend=gemini.backend,
            backend_params=gemini.backend_params,
            generation_params=gemini.generation_params,
        )
        self.judge = ConversationQualityJudge(
            model_name=gemini.judge_model,
            backend=gemini.backend,
            backend_params=gemini.backend_params,
            generation_params={**gemini.generation_params, "temperature": 0.2},
        )

    def _workflow_inputs(self, batch_size: int) -> List[Dict]:
        rows = []
        for descriptor in self.task_sampler.sample(batch_size):
            style_profile = random.choice(STYLE_PRESETS)
            rows.append(
                {
                    "sample_id": uuid4().hex,
                    "task_name": descriptor.name,
                    "intent": descriptor.intent,
                    "verification_mode": descriptor.verification_mode,
                    "sample_phrasings": descriptor.typical_user_phrasing,
                    "sub_actions": descriptor.normalized_actions,
                    "schema_context": self.schema_context,
                    "style_profile": style_profile,
                    "prompt_variant": 0,
                }
            )
        return rows

    def _serialize_row_for_llm(self, row: Dict[str, Any]) -> Dict[str, Any]:
        serialized = dict(row)
        style_profile = serialized.get("style_profile")
        if isinstance(style_profile, dict):
            serialized["style_profile"] = json.dumps(style_profile)
        workflow_plan = serialized.get("workflow_plan")
        if isinstance(workflow_plan, dict):
            serialized["workflow_plan"] = json.dumps(workflow_plan)
        arguments = serialized.get("arguments")
        if isinstance(arguments, (list, dict)):
            serialized["arguments"] = json.dumps(arguments)
        return serialized

    def _run_llm_with_retry(
        self,
        llm: curator.LLM,
        rows: List[Dict[str, Any]],
        *,
        max_attempts: int = 3,
        use_prompt_variant: bool = False,
    ) -> List[Dict[str, Any]]:
        pending = [dict(row) for row in rows]
        completed: List[Dict[str, Any]] = []
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            if not pending:
                break
            next_pending: List[Dict[str, Any]] = []
            for row in pending:
                serialized = self._serialize_row_for_llm(row)
                if use_prompt_variant:
                    serialized["prompt_variant"] = attempt
                try:
                    response: CuratorResponse = llm([serialized])
                except Exception as exc:
                    last_error = exc
                    next_pending.append(row)
                    continue
                batch_results = response.dataset.to_list()
                if not batch_results:
                    next_pending.append(row)
                    continue
                completed.extend(batch_results)
            pending = next_pending

        if pending:
            missing = ", ".join(row["sample_id"] for row in pending)
            if last_error:
                raise ValueError(f"Generation failed for sample_ids: {missing}") from last_error
            raise ValueError(f"Generation failed for sample_ids: {missing}")
        return completed

    def generate_batch(self, batch_size: int) -> List[Dict]:
        """Generate and judge a batch of CRM conversations."""

        workflow_inputs = self._workflow_inputs(batch_size)
        metadata_by_sample = {row["sample_id"]: row for row in workflow_inputs}
        workflow_results = self._run_llm_with_retry(self.workflow_generator, workflow_inputs, use_prompt_variant=True)
        workflow_rows = []
        for row in workflow_results:
            sample_id = row["sample_id"]
            meta = metadata_by_sample[sample_id]
            style_profile = row.get("style_profile")
            if not style_profile:
                style_profile = json.dumps(meta.get("style_profile") or random.choice(STYLE_PRESETS))
            workflow_rows.append(
                {
                    **row,
                    "workflow_plan": row["workflow_plan"],
                    "style_profile": style_profile,
                    "intent": meta["intent"],
                    "verification_mode": meta["verification_mode"],
                    "schema_context": meta["schema_context"],
                    "sample_phrasings": meta["sample_phrasings"],
                    "sub_actions": meta["sub_actions"],
                }
            )

        argument_results = self._run_llm_with_retry(self.argument_generator, workflow_rows, use_prompt_variant=True)
        argument_rows = []
        for row in argument_results:
            sample_id = row["sample_id"]
            meta = metadata_by_sample[sample_id]
            existing_row = next((w for w in workflow_rows if w["sample_id"] == sample_id), None)
            style_value = row.get("style_profile")
            if not style_value and existing_row:
                style_value = existing_row.get("style_profile")
            if not style_value:
                style_value = json.dumps(meta.get("style_profile") or random.choice(STYLE_PRESETS))
            argument_rows.append(
                {
                    **row,
                    "workflow_plan": row["workflow_plan"],
                    "arguments": row["arguments"],
                    "style_profile": style_value,
                    "intent": meta["intent"],
                    "verification_mode": meta["verification_mode"],
                    "schema_context": meta["schema_context"],
                    "sample_phrasings": meta["sample_phrasings"],
                    "sub_actions": meta["sub_actions"],
                }
            )

        utterance_results = self._run_llm_with_retry(self.utterance_generator, argument_rows, use_prompt_variant=True)
        conversation_records = []
        judge_inputs: List[Dict] = []
        for row in utterance_results:
            sample_id = row["sample_id"]
            meta = metadata_by_sample[sample_id]
            style_profile_obj = meta.get("style_profile") or random.choice(STYLE_PRESETS)
            style_value = row.get("style_profile") or json.dumps(style_profile_obj)
            try:
                style_profile = json.loads(style_value)
            except json.JSONDecodeError:
                style_profile = style_profile_obj
            conversation_obj = json.loads(row["conversation"])
            arguments_serialized = row["arguments"]
            try:
                arguments_parsed = json.loads(arguments_serialized) if isinstance(arguments_serialized, str) else arguments_serialized
            except json.JSONDecodeError:
                arguments_parsed = arguments_serialized
            workflow_plan_obj = row["workflow_plan"]
            try:
                workflow_plan_obj = json.loads(workflow_plan_obj) if isinstance(workflow_plan_obj, str) else workflow_plan_obj
            except json.JSONDecodeError:
                workflow_plan_obj = workflow_plan_obj

            conversation_records.append(
                {
                    **row,
                    "conversation": conversation_obj,
                    "arguments": arguments_parsed,
                    "workflow_plan": workflow_plan_obj,
                    "style_profile": style_profile,
                    "intent": meta["intent"],
                    "verification_mode": meta["verification_mode"],
                }
            )
            judge_inputs.append(
                {
                    **row,
                    "conversation": row["conversation"],
                    "style_profile": json.dumps(style_profile),
                }
            )

        judge_response: CuratorResponse = self.judge(judge_inputs)
        judgement_rows = judge_response.dataset.to_list()
        judgement_map = {row["sample_id"]: row for row in judgement_rows}

        records: List[Dict] = []
        for conversation_row in conversation_records:
            sample_id = conversation_row["sample_id"]
            records.append(
                {
                    "sample_id": sample_id,
                    "task_name": conversation_row["task_name"],
                    "intent": conversation_row.get("intent"),
                    "verification_mode": conversation_row.get("verification_mode"),
                    "style_profile": conversation_row.get("style_profile"),
                    "workflow_plan": conversation_row["workflow_plan"],
                    "arguments": conversation_row["arguments"],
                    "conversation": conversation_row["conversation"],
                    "judgement": judgement_map.get(sample_id),
                }
            )
        return records

    def save_batch(self, records: List[Dict], suffix: str) -> None:
        """Persist combined artifacts as JSONL for later analysis."""

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"batch_{suffix}.jsonl"
        with target.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
