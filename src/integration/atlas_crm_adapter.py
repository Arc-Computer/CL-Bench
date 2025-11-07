"""Atlas BYOA adapter that routes CRM conversations through the harness."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.conversation_schema import Conversation, ConversationResult, ConversationTurn
from src.crm_sandbox import MockCrmApi
from src.evaluation.conversation_harness import (
    ConversationHarness,
    build_chain_conversation_result,
    build_regular_conversation_result,
)

from .atlas_common import build_conversation_agent, conversation_from_payload

logger = logging.getLogger(__name__)


def _to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return _to_primitive(asdict(value))
    if isinstance(value, dict):
        return {k: _to_primitive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_primitive(item) for item in value]
    return value


def _build_segment_meta_lookup(conversation: Conversation) -> Dict[int, Mapping[str, Any]]:
    meta_lookup: Dict[int, Mapping[str, Any]] = {}
    segment_summaries = conversation.cumulative_context.get("segment_summaries", [])
    for item in segment_summaries:
        if not isinstance(item, Mapping):
            continue
        number = item.get("segment_number") or item.get("segment_id")
        if number is None:
            continue
        meta_lookup[int(number)] = item
    return meta_lookup


@dataclass
class ConversationSession:
    """Stateful harness runner for a single conversation/task."""

    session_id: str
    conversation: Conversation
    harness: ConversationHarness
    api: MockCrmApi
    dataset_revision: Optional[str]
    agent_config: Dict[str, Any]
    use_llm_judge: bool
    previous_turn_outputs: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    executed_turns: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    per_turn_records: List[Dict[str, Any]] = field(default_factory=list)
    first_failed_turn: Optional[int] = None
    completed: bool = False
    segment_boundaries: Sequence[int] = field(default_factory=list)
    segment_meta_lookup: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    conversation_result: Optional[ConversationResult] = None

    def __post_init__(self) -> None:
        ConversationHarness._seed_backend(self.api, self.conversation.initial_entities)
        self._agent_provider = self.harness._agent.provider_name  # type: ignore[attr-defined]
        self._agent_model = self.harness._agent.model_name  # type: ignore[attr-defined]
        self._judge_enabled = self.harness._judge is not None  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    def plan_steps(self) -> Dict[str, Any]:
        steps: List[Dict[str, Any]] = []
        for turn in self.conversation.turns:
            dependencies = set(int(dep) for dep in turn.references_previous_turns or [])
            if turn.turn_id > 1:
                dependencies.add(turn.turn_id - 1)
            steps.append(
                {
                    "id": turn.turn_id,
                    "description": turn.user_utterance,
                    "depends_on": sorted(dependencies),
                }
            )
        return {"steps": steps, "execution_mode": "stepwise"}

    # ------------------------------------------------------------------
    def execute_turn(self, turn_id: int) -> Dict[str, Any]:
        if turn_id <= len(self.per_turn_records):
            # Atlas may re-request a completed step (e.g., teacher retries). Return cached record.
            return self.per_turn_records[turn_id - 1]

        expected_turn = len(self.per_turn_records) + 1
        if turn_id != expected_turn:
            raise ValueError(
                f"Adapter requested turn {turn_id} out of order (expected {expected_turn}) "
                f"for conversation {self.conversation.conversation_id}."
            )
        turn = self._turn_by_id(turn_id)
        segment_number = self._segment_number(turn_id)
        outcome = self.harness._process_turn_with_judge(  # type: ignore[attr-defined]
            conversation=self.conversation,
            api=self.api,
            turn=turn,
            previous_turn_outputs=self.previous_turn_outputs,
            executed_turns=self.executed_turns,
            conversation_history=self.per_turn_records,
            segment_number=segment_number,
        )
        self.per_turn_records.append(outcome.record)

        if not outcome.success and self.first_failed_turn is None and turn.expect_success:
            self.first_failed_turn = turn.turn_id

        if len(self.per_turn_records) == len(self.conversation.turns):
            self.completed = True
        return outcome.record

    # ------------------------------------------------------------------
    def finalize(self) -> ConversationResult:
        if self.conversation_result is not None:
            return self.conversation_result

        if self.conversation.chain_id and self.segment_boundaries:
            result = build_chain_conversation_result(
                conversation=self.conversation,
                per_turn=self.per_turn_records,
                first_failed_turn=self.first_failed_turn,
                segment_boundaries=self.segment_boundaries,
                segment_meta_lookup=self.segment_meta_lookup,
                judge_enabled=self._judge_enabled,
                agent_provider=self._agent_provider,
                agent_model=self._agent_model,
            )
        else:
            result = build_regular_conversation_result(
                conversation=self.conversation,
                per_turn=self.per_turn_records,
                first_failed_turn=self.first_failed_turn,
                judge_enabled=self._judge_enabled,
                agent_provider=self._agent_provider,
                agent_model=self._agent_model,
            )

        self.conversation_result = result
        return result

    def run_full_conversation(self) -> ConversationResult:
        while len(self.per_turn_records) < len(self.conversation.turns):
            self.execute_turn(len(self.per_turn_records) + 1)
        return self.finalize()

    # ------------------------------------------------------------------
    def _segment_number(self, turn_id: int) -> Optional[int]:
        if not self.segment_boundaries:
            return None
        for index, boundary in enumerate(self.segment_boundaries):
            if turn_id <= boundary:
                return index + 1
        return len(self.segment_boundaries)

    def _turn_by_id(self, turn_id: int) -> ConversationTurn:
        try:
            return self.conversation.turns[turn_id - 1]
        except IndexError as exc:
            raise ValueError(
                f"Conversation {self.conversation.conversation_id} "
                f"does not define turn {turn_id} (total turns={len(self.conversation.turns)})."
            ) from exc


_SESSION_CACHE: Dict[str, ConversationSession] = {}


def _build_session(session_id: str, task_payload: Mapping[str, Any]) -> ConversationSession:
    conversation_payload = task_payload.get("conversation")
    if not isinstance(conversation_payload, Mapping):
        raise ValueError("Task payload is missing the serialized conversation.")

    conversation = conversation_from_payload(conversation_payload)
    agent_config = dict(task_payload.get("agent_config") or {})
    agent = build_conversation_agent(agent_config)
    use_llm_judge = bool(task_payload.get("use_llm_judge", True))
    harness = ConversationHarness([conversation], agent=agent, use_llm_judge=use_llm_judge)
    api = MockCrmApi()

    return ConversationSession(
        session_id=session_id,
        conversation=conversation,
        harness=harness,
        api=api,
        dataset_revision=task_payload.get("dataset_revision"),
        agent_config=agent_config,
        use_llm_judge=use_llm_judge,
        segment_boundaries=conversation.segment_boundaries or [],
        segment_meta_lookup=_build_segment_meta_lookup(conversation),
    )


def _get_session(session_id: str, task_payload: Mapping[str, Any]) -> ConversationSession:
    session = _SESSION_CACHE.get(session_id)
    if session is None:
        session = _build_session(session_id, task_payload)
        _SESSION_CACHE[session_id] = session
    return session


def _reset_session(session_id: str, task_payload: Mapping[str, Any]) -> ConversationSession:
    session = _build_session(session_id, task_payload)
    _SESSION_CACHE[session_id] = session
    return session


def _conversation_id_from_payload(task_payload: Mapping[str, Any]) -> str:
    conversation_payload = task_payload.get("conversation")
    if isinstance(conversation_payload, Mapping):
        conv_id = conversation_payload.get("conversation_id")
        if conv_id:
            return str(conv_id)
    return "unknown"


def _response(**payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def create_conversation_adapter(
    prompt: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> str:
    """Entry point for the Atlas Python adapter."""

    metadata = metadata or {}
    mode = metadata.get("mode")
    execution_mode = metadata.get("execution_mode")
    single_shot_mode = execution_mode == "single_shot"
    step_payload = metadata.get("step_payload")
    raw_task_payload = metadata.get("task_payload")

    if raw_task_payload is None:
        if prompt is None:
            return _response(status="ready", message="Atlas CRM adapter expects structured payloads.")
        return _response(status="error", error="metadata.task_payload is required for Atlas CRM adapter.")

    try:
        task_payload = json.loads(raw_task_payload)
    except json.JSONDecodeError as exc:
        return _response(status="error", error=f"Invalid task_payload JSON: {exc}")

    session_id = str(task_payload.get("task_id") or _conversation_id_from_payload(task_payload))
    try:
        if mode == "planning":
            session = _reset_session(session_id, task_payload)
            plan = session.plan_steps()
            return json.dumps(plan, ensure_ascii=False)

        single_shot_step = False
        if step_payload:
            description = step_payload.get("description")
            if isinstance(description, str) and description.startswith("Produce the complete answer"):
                single_shot_step = True

        if step_payload and not (single_shot_mode or single_shot_step):
            session = _get_session(session_id, task_payload)
            turn_id_raw = step_payload.get("step_id") or step_payload.get("id")
            if turn_id_raw is None:
                raise ValueError("step_payload.step_id is required for execution.")
            turn_id = int(turn_id_raw)
            turn_record = session.execute_turn(turn_id)
            response: Dict[str, Any] = {
                "status": "ok",
                "conversation_id": session.conversation.conversation_id,
                "dataset_revision": session.dataset_revision,
                "agent_config": session.agent_config,
                "use_llm_judge": session.use_llm_judge,
                "turn_id": turn_id,
                "tool_name": turn_record.get("tool_name"),
                "arguments": turn_record.get("arguments"),
                "turn_result": turn_record,
                "completed_turns": len(session.per_turn_records),
                "total_turns": len(session.conversation.turns),
            }
            if session.completed:
                result = session.finalize()
                response["conversation_result"] = _to_primitive(result)
                _SESSION_CACHE.pop(session_id, None)
            return _response(**response)

        session = _reset_session(session_id, task_payload)
        session.run_full_conversation()
        result = session.finalize()
        response = {
            "status": "ok",
            "conversation_id": session.conversation.conversation_id,
            "dataset_revision": session.dataset_revision,
            "agent_config": session.agent_config,
            "use_llm_judge": session.use_llm_judge,
            "turn_results": list(session.per_turn_records),
            "conversation_result": _to_primitive(result),
            "execution_mode": "single_shot",
        }
        _SESSION_CACHE.pop(session_id, None)
        return _response(**response)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Atlas CRM adapter error")
        return _response(
            status="error",
            error=str(exc),
            conversation_id=_conversation_id_from_payload(task_payload),
        )


__all__ = ["create_conversation_adapter"]
