from __future__ import annotations

import json
import logging
import os
import random
import re
from uuid import uuid4
from typing import Any, Dict, List, Optional

from bespokelabs.curator.client import Client
from bespokelabs.curator.types.curator_response import CuratorResponse

from src.evaluation.conversation_harness import ConversationResult

from .arguments import SchemaCompliantArgumentGenerator
from .config import PipelineConfig
from .judge import ConversationQualityJudge
from .schema_utils import build_schema_context
from .task_sampler import TaskSampler
from .utterances import NaturalUtteranceGenerator
from .response_paraphraser import ResponseParaphraser
from .workflow import WorkflowSequenceGenerator

os.environ.setdefault("CURATOR_DISABLE_CACHE", "1")

STYLE_PRESETS = [
    {"persona": "enterprise account executive", "formality": "formal", "tone": "concise"},
    {"persona": "startup customer success lead", "formality": "casual", "tone": "supportive"},
    {"persona": "solutions consultant", "formality": "consultative", "tone": "analytical"},
    {"persona": "implementation project manager", "formality": "formal", "tone": "detailed"},
    {"persona": "sales development rep", "formality": "casual", "tone": "energetic"},
]

logger = logging.getLogger(__name__)

_REFERENCE_PATTERN = re.compile(
    r"step(?P<step>\d+)\.(?:input|output)\.(?P<field>[a-z0-9_]+)",
    re.IGNORECASE,
)


def _parse_reference_spec(spec: Any) -> Dict[str, int | str] | None:
    if isinstance(spec, dict):
        try:
            step = int(spec.get("from_step") or spec.get("step"))
        except (TypeError, ValueError):
            return None
        field = spec.get("output_field") or spec.get("field")
        if not field:
            return None
        return {"from_step": step, "output_field": str(field)}
    if isinstance(spec, str):
        match = _REFERENCE_PATTERN.fullmatch(spec.strip())
        if not match:
            return None
        return {
            "from_step": int(match.group("step")),
            "output_field": match.group("field"),
        }
    return None


def _normalize_argument_entries(entries: List[Dict[str, Any]]) -> None:
    for entry in entries:
        references = entry.get("references") or {}
        if not isinstance(references, dict):
            continue
        normalized_refs: Dict[str, Dict[str, int | str]] = {}
        for field, raw_spec in references.items():
            parsed = _parse_reference_spec(raw_spec)
            if not parsed:
                continue
            placeholder = f"{{{{turn_{parsed['from_step']}.{parsed['output_field']}}}}}"
            entry.setdefault("arguments", {})
            entry["arguments"][field] = placeholder
            normalized_refs[field] = parsed
        entry["references"] = normalized_refs


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
        self.response_paraphraser = ResponseParaphraser(
            model_name=gemini.utterance_model,
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

    def _generate_argument_rows(self, workflow_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the argument generator with parity validation between plan steps and argument stanzas."""

        if not workflow_rows:
            return []

        workflow_lookup = {row["sample_id"]: row for row in workflow_rows}
        remaining = list(workflow_lookup.keys())
        stored: Dict[str, Dict[str, Any]] = {}
        max_attempts = max(1, self.config.argument_regeneration_attempts)
        attempts = 0

        while remaining:
            attempts += 1
            batch_rows: List[Dict[str, Any]] = []
            for sample_id in remaining:
                base_row = dict(workflow_lookup[sample_id])
                if attempts > 1:
                    base_row["prompt_variant"] = int(base_row.get("prompt_variant", 0)) + attempts - 1
                    base_row["_cache_bust"] = uuid4().hex
                batch_rows.append(base_row)
            batch_results = self._run_llm_with_retry(self.argument_generator, batch_rows, use_prompt_variant=True)
            retry_ids: List[str] = []
            returned_ids = {row["sample_id"] for row in batch_results}
            missing_ids = [sample_id for sample_id in remaining if sample_id not in returned_ids]
            if missing_ids:
                logger.warning("Argument generator skipped sample_ids: %s", ", ".join(missing_ids))
                retry_ids.extend(missing_ids)
            for row in batch_results:
                sample_id = row["sample_id"]
                workflow_plan_obj = row.get("workflow_plan")
                try:
                    workflow_plan_payload = (
                        json.loads(workflow_plan_obj)
                        if isinstance(workflow_plan_obj, str)
                        else dict(workflow_plan_obj)
                    )
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Invalid workflow payload for %s; retrying.", sample_id)
                    retry_ids.append(sample_id)
                    continue

                arguments_payload = row.get("arguments")
                try:
                    arguments_list = (
                        json.loads(arguments_payload)
                        if isinstance(arguments_payload, str)
                        else list(arguments_payload)
                    )
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Invalid argument payload for %s; retrying.", sample_id)
                    retry_ids.append(sample_id)
                    continue

                success_steps = workflow_plan_payload.get("success_path") or []
                if len(success_steps) != len(arguments_list):
                    logger.warning(
                        "Sample %s produced %s plan steps but %s argument blocks; regenerating.",
                        sample_id,
                        len(success_steps),
                        len(arguments_list),
                    )
                    retry_ids.append(sample_id)
                    continue

                stored[sample_id] = row

            if not retry_ids:
                break

            if attempts >= max_attempts:
                raise ValueError(
                    "Argument generation failed to align plan/argument counts for samples: "
                    + ", ".join(sorted(set(retry_ids)))
                )

            remaining = sorted(set(retry_ids))

        return [stored[row["sample_id"]] for row in workflow_rows]

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

        argument_results = self._generate_argument_rows(workflow_rows)
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
            arguments_payload = row.get("arguments")
            try:
                argument_entries = (
                    json.loads(arguments_payload)
                    if isinstance(arguments_payload, str)
                    else list(arguments_payload or [])
                )
            except (json.JSONDecodeError, TypeError):
                argument_entries = []
            _normalize_argument_entries(argument_entries)
            normalized_arguments = json.dumps(argument_entries)
            argument_rows.append(
                {
                    **row,
                    "workflow_plan": row["workflow_plan"],
                    "arguments": normalized_arguments,
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

            workflow_plan_data = row["workflow_plan"]
            try:
                workflow_plan_parsed = json.loads(workflow_plan_data) if isinstance(workflow_plan_data, str) else dict(workflow_plan_data)
            except (json.JSONDecodeError, TypeError):
                workflow_plan_parsed = workflow_plan_data

            scenario_metadata = {
                "contains_failure": bool(row.get("contains_failure") or conversation_obj.get("contains_failure")),
                "failure_turn": row.get("failure_turn") or conversation_obj.get("failure_turn"),
                "chain_id": row.get("chain_id") or conversation_obj.get("chain_id"),
                "segment_boundaries": row.get("segment_boundaries") or conversation_obj.get("segment_boundaries"),
                "complexity_hint": len((workflow_plan_parsed or {}).get("success_path") or []),
            }

            conversation_records.append(
                {
                    **row,
                    "conversation": conversation_obj,
                    "arguments": arguments_parsed,
                    "workflow_plan": workflow_plan_obj,
                    "style_profile": style_profile,
                    "intent": meta["intent"],
                    "verification_mode": meta["verification_mode"],
                    "scenario_metadata": scenario_metadata,
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
                    "scenario_metadata": conversation_row.get("scenario_metadata", {}),
                }
            )
        return records

    def rewrite_conversations(
        self,
        records: List[Dict[str, Any]],
        harness_results: List[ConversationResult],
    ) -> None:
        """Paraphrase assistant turns using actual tool outputs from harness results."""

        result_map = {result.conversation_id: result for result in harness_results}
        for record in records:
            conversation_id = record.get("sample_id")
            harness_result = result_map.get(conversation_id)
            if not harness_result:
                continue

            conversation_payload = record.get("conversation")
            if isinstance(conversation_payload, str):
                try:
                    conversation_payload = json.loads(conversation_payload)
                except json.JSONDecodeError:
                    continue

            turns = conversation_payload.get("turns")
            if not isinstance(turns, list):
                continue

            style_profile = record.get("style_profile") or {}
            if isinstance(style_profile, str):
                try:
                    style_profile = json.loads(style_profile)
                except json.JSONDecodeError:
                    style_profile = {}

            per_turn = harness_result.per_turn_results
            assistant_index = 0
            for idx, turn in enumerate(turns):
                if turn.get("role") != "assistant":
                    continue
                if assistant_index >= len(per_turn):
                    break

                turn_result = per_turn[assistant_index]
                assistant_index += 1

                user_prompt = ""
                if idx > 0:
                    user_prompt = turns[idx - 1].get("content", "")

                turn_number = assistant_index

                rendered, template = self.response_paraphraser.paraphrase(
                    persona=style_profile,
                    user_utterance=user_prompt,
                    turn_result=turn_result,
                    turn_number=turn_number,
                    generate_template=True,
                )
                if rendered:
                    turn["content"] = rendered
                if template:
                    turn["response_template"] = template

                tool_call = turn.get("tool_call") or {}
                tool_call.setdefault("name", turn_result.get("expected_tool") or turn_result.get("tool_name"))
                args = turn_result.get("expected_arguments") or turn_result.get("arguments") or {}
                tool_call["arguments"] = args
                turn["tool_call"] = tool_call

            record["conversation"] = conversation_payload

    def save_batch(self, records: List[Dict], suffix: str) -> None:
        """Persist combined artifacts as JSONL for later analysis."""

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"batch_{suffix}.jsonl"
        with target.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
