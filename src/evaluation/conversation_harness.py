"""Lean harness for executing CRM conversations against the mock backend."""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Union
from uuid import UUID

from src.conversation_schema import Conversation, ConversationResult, ConversationTurn
from src.crm_backend import DatabaseConfig, PostgresCrmBackend
from src.crm_sandbox import MockCrmApi
from src.evaluation.agents import AgentTurnContext, ConversationAgent, MockAgent
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.verification import VerificationMode
from src.generation.conversation_generator import (
    API_STORE_ATTR,
    ENTITY_TYPE_ORDER,
    _extract_reference_payload,
    _get_entity_builder,
)
from enum import Enum
from src.reference_resolver import TemplateResolutionError, resolve_template
from src.tool_schema import canonicalize_tool_arguments, prepare_execution_arguments

logger = logging.getLogger(__name__)

POSTGRES_ENTITY_COLUMN_MAP: Dict[str, tuple[str, tuple[str, ...]]] = {
    "Client": (
        "clients",
        ("client_id", "name", "email", "status", "industry", "phone", "address", "owner"),
    ),
    "Contact": (
        "contacts",
        ("contact_id", "first_name", "last_name", "title", "email", "phone", "client_id", "notes"),
    ),
    "Opportunity": (
        "opportunities",
        ("opportunity_id", "client_id", "name", "stage", "amount", "close_date", "owner", "probability", "notes"),
    ),
    "Quote": (
        "quotes",
        ("quote_id", "opportunity_id", "version", "amount", "status", "valid_until", "quote_prefix"),
    ),
    "Contract": (
        "contracts",
        ("contract_id", "client_id", "opportunity_id", "start_date", "end_date", "value", "status", "document_url"),
    ),
    "Document": (
        "documents",
        ("document_id", "entity_type", "entity_id", "file_name", "uploaded_by", "file_url"),
    ),
    "Note": (
        "notes",
        ("note_id", "entity_type", "entity_id", "content", "created_by"),
    ),
}


@dataclass
class TurnProcessOutcome:
    """Internal helper capturing the result of executing a single turn."""

    record: Dict[str, Any]
    success: bool
    expected_failure: bool = False
    error_message: Optional[str] = None


def _to_primitive(value: Any) -> Any:
    """Convert Python objects to JSON-serializable primitives.
    
    Recursively converts datetime, UUID, Decimal, dataclass instances, and
    Pydantic models to JSON-safe types (strings, floats, dicts).
    """
    if value is None:
        return None
    
    # Handle Pydantic models (must check before dataclass since Pydantic models can be dataclasses)
    try:
        from pydantic import BaseModel
        if isinstance(value, BaseModel):
            return _to_primitive(value.model_dump(mode='json'))
    except ImportError:
        pass
    
    if is_dataclass(value):
        return _to_primitive(asdict(value))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        # Convert keys to strings if they're UUIDs or other non-string types
        result = {}
        for k, v in value.items():
            key = str(k) if not isinstance(k, (str, int, float, bool)) or k is None else k
            result[key] = _to_primitive(v)
        return result
    if isinstance(value, list):
        return [_to_primitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_primitive(item) for item in value)
    
    # Fallback: try to convert unknown types to string
    try:
        # Check if it's a Pydantic AnyUrl or similar
        if hasattr(value, '__str__'):
            return str(value)
    except Exception:
        pass
    
    return value


def _arguments_match_semantically(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    _path: str = "",
) -> tuple[bool, Optional[str]]:
    """Recursively compare arguments allowing for common LLM variations."""
    for key, expected_val in expected.items():
        if key not in actual:
            return False, f"Missing required field: {_path}{key}"

        actual_val = actual[key]
        current_path = f"{_path}{key}."

        if isinstance(expected_val, Mapping) and isinstance(actual_val, Mapping):
            match, reason = _arguments_match_semantically(actual_val, expected_val, current_path)
            if not match:
                return False, reason
        elif isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float, str)):
            try:
                if float(actual_val) != float(expected_val):
                    return False, f"Field {_path}{key}: {actual_val} != {expected_val}"
            except (TypeError, ValueError):
                return False, f"Field {_path}{key}: {actual_val} not numeric"
        elif isinstance(expected_val, str) and isinstance(actual_val, str):
            if actual_val.strip() != expected_val.strip():
                return False, f"Field {_path}{key}: '{actual_val}' != '{expected_val}'"
        elif isinstance(expected_val, list) and isinstance(actual_val, list):
            if len(actual_val) != len(expected_val):
                return False, f"Field {_path}{key}: list length {len(actual_val)} != {len(expected_val)}"
            for index, (actual_item, expected_item) in enumerate(zip(actual_val, expected_val)):
                if isinstance(expected_item, Mapping) and isinstance(actual_item, Mapping):
                    match, reason = _arguments_match_semantically(
                        actual_item,
                        expected_item,
                        f"{current_path}[{index}].",
                    )
                    if not match:
                        return False, reason
                elif actual_item != expected_item:
                    return False, f"Field {_path}{key}[{index}]: {actual_item} != {expected_item}"
        elif actual_val != expected_val:
            return False, f"Field {_path}{key}: {actual_val} != {expected_val}"

    return True, None


def _normalise_response_text(text: Any) -> str:
    """Normalize response strings for structured comparison."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.split()).strip().lower()


class ConversationHarness:
    """Execute conversations against the mock CRM backend."""

    def __init__(
        self,
        conversations: Sequence[Conversation],
        *,
        output_path: Optional[Path] = None,
        agent: Optional[ConversationAgent] = None,
        use_llm_judge: bool = True,
        backend: Literal["mock", "postgres"] = "mock",
        db_config: Optional[DatabaseConfig] = None,
    ) -> None:
        self._conversations = list(conversations)
        self._agent: ConversationAgent = agent or MockAgent()
        self._output_path = output_path
        self._backend_mode = backend
        self._db_config = db_config
        self._judge: Optional[LLMJudge] = None
        if use_llm_judge:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                warnings.warn("LLM judge disabled: OPENAI_API_KEY not set")
            else:
                try:
                    self._judge = LLMJudge()
                except ImportError as exc:
                    warnings.warn(f"LLM judge disabled: {exc}")

    # ------------------------------------------------------------------
    def run(self) -> List[ConversationResult]:
        """Execute all conversations and return their results.
        
        Supports resume functionality: if output_path exists, loads existing results
        and skips already-processed conversations. Writes results incrementally
        after each conversation to enable crash recovery.
        """
        # Load existing results if output file exists (resume support)
        existing_results: Dict[str, ConversationResult] = {}
        if self._output_path and Path(self._output_path).exists():
            try:
                existing_results = self._load_existing_results()
                logger.info(
                    "Found %d existing results in %s, will resume from remaining conversations",
                    len(existing_results),
                    self._output_path,
                )
            except Exception as exc:
                logger.warning("Failed to load existing results, starting fresh: %s", exc)

        results: List[ConversationResult] = []
        total = len(self._conversations)
        if total == 0:
            logger.info("ConversationHarness received zero conversations; nothing to run.")
            return results
        
        # Filter out already-processed conversations
        remaining_conversations = [
            conv for conv in self._conversations
            if conv.conversation_id not in existing_results
        ]
        
        if not remaining_conversations:
            logger.info("All conversations already processed, returning existing results")
            return list(existing_results.values())
        
        logger.info(
            "Processing %d conversations (%d already completed, %d remaining)",
            total,
            len(existing_results),
            len(remaining_conversations),
        )
        
        # Prepare output file for incremental writing
        output_file = None
        if self._output_path:
            output_path = Path(self._output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Open in append mode if file exists, otherwise create new
            output_file = output_path.open("a" if output_path.exists() else "w", encoding="utf-8")
        
        start_time = time.time()
        processed_count = len(existing_results)
        
        try:
            for index, conversation in enumerate(remaining_conversations, start=1):
                try:
                    result = self._run_single(conversation)
                except Exception as exc:  # pragma: no cover - logging path
                    logger.exception("Conversation %s failed", conversation.conversation_id)
                    result = ConversationResult(
                        conversation_id=conversation.conversation_id,
                        overall_success=False,
                        turns_executed=1,
                        failed_at_turn=1,
                        per_turn_results=[
                            {
                                "turn_id": 1,
                                "success": False,
                                "error": str(exc),
                                "verification": "pre_execution_error",
                                "tool_success": False,
                                "response_success": False,
                            }
                        ],
                        reward_signal=0.0,
                        error_message=str(exc),
                        metadata={
                            "verification_mode": conversation.verification_mode.value,
                            "agent": {
                                "provider": self._agent.provider_name,
                                "model": self._agent.model_name,
                            },
                        },
                    )
                
                results.append(result)
                
                # Write result immediately (incremental writing for crash recovery)
                if output_file:
                    output_file.write(json.dumps(_to_primitive(result.__dict__), ensure_ascii=False) + "\n")
                    output_file.flush()  # Ensure data is written to disk
                
                # Calculate running success rate (including existing results)
                all_results = list(existing_results.values()) + results
                successes = sum(1 for r in all_results if r.overall_success)
                current_index = processed_count + index
                success_rate = (successes / current_index * 100.0) if current_index > 0 else 0.0
                
                # Log every conversation with running success rate
                elapsed = time.time() - start_time
                rate = index / elapsed if elapsed else 0.0
                eta = (len(remaining_conversations) - index) / rate if rate else float("inf")
                eta_text = (
                    "N/A"
                    if eta == float("inf")
                    else time.strftime("%H:%M:%S", time.gmtime(int(max(0.0, eta))))
                )
                logger.info(
                    "[%d/%d] Conversation: %s | Success: %s | Running Success Rate: %.1f%% (%d/%d) | ETA: %s",
                    current_index,
                    total,
                    conversation.conversation_id,
                    "✓" if result.overall_success else "✗",
                    success_rate,
                    successes,
                    current_index,
                    eta_text,
                )
        finally:
            if output_file:
                output_file.close()
        
        # Return all results (existing + newly processed)
        return list(existing_results.values()) + results

    def _run_single(self, conversation: Conversation) -> ConversationResult:
        """Run a single conversation, handling both regular and chained conversations."""
        if conversation.chain_id and conversation.segment_boundaries:
            return self._run_single_chain(conversation)
        return self._run_single_regular(conversation)

    # ------------------------------------------------------------------
    def _run_single_regular(self, conversation: Conversation) -> ConversationResult:
        """Execute a regular (non-chained) conversation without failing fast."""
        api = self._create_backend()
        try:
            self._seed_backend(api, conversation.initial_entities)

            per_turn: List[Dict[str, Any]] = []
            previous_turn_outputs: Dict[int, Dict[str, Any]] = {}
            executed_turns: Dict[int, Dict[str, Any]] = {}
            first_failed_turn: Optional[int] = None

            for turn in conversation.turns:
                outcome = self._process_turn_with_judge(
                    conversation=conversation,
                    api=api,
                    turn=turn,
                    segment_number=1,
                    previous_turn_outputs=previous_turn_outputs,
                    executed_turns=executed_turns,
                    conversation_history=per_turn,
                )
                per_turn.append(outcome.record)

                if not outcome.success and first_failed_turn is None and turn.expect_success:
                    first_failed_turn = turn.turn_id

            return build_regular_conversation_result(
                conversation=conversation,
                per_turn=per_turn,
                first_failed_turn=first_failed_turn,
                judge_enabled=self._judge is not None,
                agent_provider=self._agent.provider_name,
                agent_model=self._agent.model_name,
            )
        finally:
            self._teardown_backend(api)

    # ------------------------------------------------------------------
    def _run_single_chain(self, conversation: Conversation) -> ConversationResult:
        """Execute a chained conversation while collecting per-segment metrics."""
        api = self._create_backend()
        try:
            self._seed_backend(api, conversation.initial_entities)

            segment_boundaries = conversation.segment_boundaries or []
            if not segment_boundaries:
                raise ValueError(
                    f"Chained conversation {conversation.conversation_id} is missing segment boundaries."
                )

            segment_meta_lookup: Dict[int, Mapping[str, Any]] = {}
            for item in conversation.cumulative_context.get("segment_summaries", []):
                if not isinstance(item, Mapping):
                    continue
                number = item.get("segment_number") or item.get("segment_id")
                if number is None:
                    continue
                segment_meta_lookup[int(number)] = item

            per_turn: List[Dict[str, Any]] = []
            previous_turn_outputs: Dict[int, Dict[str, Any]] = {}
            executed_turns: Dict[int, Dict[str, Any]] = {}
            first_failed_turn: Optional[int] = None

            def _segment_for_turn(turn_id: int) -> int:
                for index, boundary in enumerate(segment_boundaries):
                    if turn_id <= boundary:
                        return index + 1
                return len(segment_boundaries)

            for turn in conversation.turns:
                segment_number = _segment_for_turn(turn.turn_id)
                outcome = self._process_turn_with_judge(
                    conversation=conversation,
                    api=api,
                    turn=turn,
                    segment_number=segment_number,
                    previous_turn_outputs=previous_turn_outputs,
                    executed_turns=executed_turns,
                    conversation_history=per_turn,
                )
                per_turn.append(outcome.record)

                if not outcome.success and first_failed_turn is None and turn.expect_success:
                    first_failed_turn = turn.turn_id

            return build_chain_conversation_result(
                conversation=conversation,
                per_turn=per_turn,
                first_failed_turn=first_failed_turn,
                segment_boundaries=segment_boundaries,
                segment_meta_lookup=segment_meta_lookup,
                judge_enabled=self._judge is not None,
                agent_provider=self._agent.provider_name,
                agent_model=self._agent.model_name,
            )
        finally:
            self._teardown_backend(api)

    # ------------------------------------------------------------------
    def _process_turn_with_judge(
        self,
        *,
        conversation: Conversation,
        api: Union[MockCrmApi, PostgresCrmBackend],
        turn: ConversationTurn,
        previous_turn_outputs: Dict[int, Dict[str, Any]],
        executed_turns: Dict[int, Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
        segment_number: Optional[int] = None,
        guidance: Optional[List[str]] = None,
    ) -> TurnProcessOutcome:
        """Execute a single turn using semantic validation with an optional LLM judge."""
        try:
            return self._process_turn_with_judge_impl(
                conversation=conversation,
                api=api,
                turn=turn,
                previous_turn_outputs=previous_turn_outputs,
                executed_turns=executed_turns,
                conversation_history=conversation_history,
                segment_number=segment_number,
                guidance=guidance,
            )
        except Exception as exc:
            # If using Postgres backend and transaction is aborted, rollback and restart
            if isinstance(api, PostgresCrmBackend):
                try:
                    api.rollback_session()
                    api.begin_session(reset=False)
                    logger.warning(
                        "Rolled back and restarted Postgres transaction after error in turn %d of %s: %s",
                        turn.turn_id,
                        conversation.conversation_id,
                        exc,
                    )
                except Exception as cleanup_exc:
                    logger.error(
                        "Failed to cleanup Postgres transaction after error: %s",
                        cleanup_exc,
                        exc_info=True,
                    )
            # Re-raise the original exception
            raise

    def _process_turn_with_judge_impl(
        self,
        *,
        conversation: Conversation,
        api: Union[MockCrmApi, PostgresCrmBackend],
        turn: ConversationTurn,
        previous_turn_outputs: Dict[int, Dict[str, Any]],
        executed_turns: Dict[int, Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
        segment_number: Optional[int] = None,
        guidance: Optional[List[str]] = None,
    ) -> TurnProcessOutcome:
        """Execute a single turn using semantic validation with an optional LLM judge."""
        # Resolve templates with strict=False if previous turns failed
        # This allows the turn to proceed even if templates can't be resolved
        # (e.g., when a previous turn failed and its results aren't available)
        strict_resolution = True
        for prev_turn_id in range(1, turn.turn_id):
            if prev_turn_id not in previous_turn_outputs:
                # A previous turn failed, so we can't resolve templates strictly
                strict_resolution = False
                break
        
        try:
            resolved_expected_args = resolve_template(
                turn.expected_args,
                previous_turn_outputs,
                turn.turn_id,
                strict=strict_resolution,
            )
        except TemplateResolutionError as exc:  # pragma: no cover - defensive guard
            # If strict resolution failed, try non-strict to allow the turn to proceed
            if strict_resolution:
                logger.warning(
                    "Strict template resolution failed for turn %d in %s: %s. "
                    "Attempting non-strict resolution.",
                    turn.turn_id,
                    conversation.conversation_id,
                    exc,
                )
                resolved_expected_args = resolve_template(
                    turn.expected_args,
                    previous_turn_outputs,
                    turn.turn_id,
                    strict=False,
                )
            else:
                raise RuntimeError(
                    f"Failed to resolve arguments for turn {turn.turn_id} "
                    f"in {conversation.conversation_id}: {exc}"
                ) from exc

        resolved_expected_args = canonicalize_tool_arguments(turn.expected_tool, resolved_expected_args)

        context = AgentTurnContext(
            conversation=conversation,
            turn=turn,
            prior_turns=conversation.turns[: turn.turn_id - 1],
            previous_results=executed_turns,
            expected_arguments=resolved_expected_args,
        )

        try:
            agent_call = self._agent.tool_call(context, guidance=guidance)
        except Exception as exc:  # pragma: no cover - defensive guard
            record: Dict[str, Any] = {
                "turn_id": turn.turn_id,
                "user_utterance": turn.user_utterance,
                "expected_tool": turn.expected_tool,
                "expected_arguments": resolved_expected_args,
                "expect_success": turn.expect_success,
                "success": False,
                "error": f"Agent error: {exc}",
                "verification": "agent_error",
                "tool_success": False,
                "response_success": False,
            }
            if segment_number is not None:
                record["segment_number"] = segment_number
            return TurnProcessOutcome(
                record=_to_primitive(record),
                success=False,
                error_message=str(exc),
            )

        tool_name = agent_call.tool_name or turn.expected_tool
        if not tool_name:
            raise RuntimeError(
                f"Agent did not provide a tool name for turn {turn.turn_id} "
                f"of {conversation.conversation_id}."
            )

        tool = getattr(api, tool_name, None)
        if tool is None:
            error_message = (
                f"MockCrmApi does not implement tool '{tool_name}' "
                f"(turn {turn.turn_id} of {conversation.conversation_id})."
            )
            record = {
                "turn_id": turn.turn_id,
                "user_utterance": turn.user_utterance,
                "expected_tool": turn.expected_tool,
                "expected_arguments": resolved_expected_args,
                "tool_name": tool_name,
                "arguments": agent_call.arguments if isinstance(agent_call.arguments, dict) else {},
                "expect_success": turn.expect_success,
                "matches_expected_tool": False,
                "token_usage": dict(agent_call.token_usage),
                "success": False,
                "error": error_message,
                "verification": "tool_not_found",
                "judge_used": False,
                "tool_success": False,
                "response_success": False,
            }
            if agent_call.response_text:
                record["agent_response_text"] = agent_call.response_text
            if segment_number is not None:
                record["segment_number"] = segment_number
            return TurnProcessOutcome(
                record=_to_primitive(record),
                success=False,
                error_message=error_message,
            )

        if not isinstance(agent_call.arguments, dict):
            raise RuntimeError(
                f"Agent returned non-dictionary arguments for turn {turn.turn_id} "
                f"of {conversation.conversation_id}: {agent_call.arguments!r}"
            )

        arguments = canonicalize_tool_arguments(tool_name, agent_call.arguments)
        execution_arguments = prepare_execution_arguments(tool_name, arguments)
        record: Dict[str, Any] = {
            "turn_id": turn.turn_id,
            "user_utterance": turn.user_utterance,
            "tool_name": tool_name,
            "arguments": arguments,
            "expected_tool": turn.expected_tool,
            "expected_arguments": resolved_expected_args,
            "expect_success": turn.expect_success,
            "matches_expected_tool": tool_name == turn.expected_tool,
            "token_usage": dict(agent_call.token_usage),
        }
        if segment_number is not None:
            record["segment_number"] = segment_number
        if agent_call.reasoning:
            record["reasoning"] = agent_call.reasoning
        if agent_call.raw_response:
            record["raw_agent_response"] = agent_call.raw_response
        if agent_call.response_text:
            record["agent_response_text"] = agent_call.response_text
        if turn.expected_response:
            record["expected_response"] = turn.expected_response.to_dict()
            record["response_evaluation"] = turn.expected_response.evaluation
        else:
            record["response_evaluation"] = "structured"
        record["tool_success"] = False
        record["response_success"] = False
        record["judge_used"] = False
        record["response_judge_used"] = False
        record["response_judge_pass"] = False

        expected_args_canonical = resolved_expected_args
        record["expected_arguments"] = expected_args_canonical

        if isinstance(expected_args_canonical, Mapping):
            arg_match, mismatch_reason = _arguments_match_semantically(arguments, expected_args_canonical)
        else:
            arg_match = arguments == expected_args_canonical
            mismatch_reason = None if arg_match else "Arguments did not exactly match expected payload"
        record["matches_expected_arguments"] = arg_match
        if not arg_match and mismatch_reason:
            record["argument_mismatch_reason"] = mismatch_reason

        try:
            result = tool(**execution_arguments)
            execution_success = True
            record["result"] = _extract_reference_payload(result)
        except Exception as exc:
            execution_success = False
            error_message = str(exc)
            record["error"] = error_message

            # If using Postgres backend, rollback and restart transaction to allow subsequent turns
            if isinstance(api, PostgresCrmBackend):
                try:
                    api.rollback_session()
                    api.begin_session(reset=False)
                    logger.debug(
                        "Rolled back and restarted Postgres transaction after tool error in turn %d of %s",
                        turn.turn_id,
                        conversation.conversation_id,
                    )
                except Exception as cleanup_exc:
                    logger.warning(
                        "Failed to cleanup Postgres transaction after tool error: %s",
                        cleanup_exc,
                        exc_info=True,
                    )

            if not turn.expect_success:
                executed_turns[turn.turn_id] = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "error": error_message,
                    "success": False,
                }

                expected_substring = turn.expected_error_substring
                if expected_substring:
                    expected_lower = expected_substring.lower()
                    error_lower = error_message.lower()
                    if expected_lower not in error_lower and expected_lower != "validation error":
                        record["expected_error_mismatch"] = expected_substring

                record["verification"] = "expected_failure_diagnostic"
                record["tool_success"] = True
                record["response_success"] = False
                record["success"] = False
                return TurnProcessOutcome(
                    record=_to_primitive(record),
                    success=False,
                    expected_failure=True,
                    error_message=error_message,
                )

        if not turn.expect_success:
            executed_turns[turn.turn_id] = {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": record.get("result"),
                "success": True,
            }
            previous_turn_outputs[turn.turn_id] = record.get("result", {})
            record["verification"] = "unexpected_success"
            record["tool_success"] = False
            record["response_success"] = False
            record["success"] = False
            record["error"] = (
                f"Tool '{tool_name}' succeeded but failure was expected for "
                f"turn {turn.turn_id} of {conversation.conversation_id}."
            )
            record["expected_error_mismatch"] = turn.expected_error_substring
            return TurnProcessOutcome(
                record=_to_primitive(record),
                success=False,
                error_message=record["error"],
            )

        if execution_success:
            executed_turns[turn.turn_id] = {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": record.get("result"),
                "success": True,
            }
            previous_turn_outputs[turn.turn_id] = record.get("result", {})
        else:
            executed_turns[turn.turn_id] = {
                "tool_name": tool_name,
                "arguments": arguments,
                "error": record.get("error"),
                "success": False,
            }

        tool_exact_match = record["matches_expected_tool"] and arg_match and execution_success
        tool_success = bool(tool_exact_match)
        record["verification"] = "exact_match" if tool_exact_match else "failed"

        # For quality evaluation focused on goal achievement:
        # - If exact match fails but execution succeeded, use judge to evaluate goal achievement
        # - This allows alternate valid paths (e.g., opportunity_details vs view_opportunity_details)
        #   to be counted as success if they accomplish the user's intent
        if not tool_success and self._judge:
            # Only use judge if execution succeeded (goal-focused evaluation)
            # If execution failed, exact match failure is appropriate
            if execution_success:
                judge_result = self._judge.judge_turn(
                    user_utterance=turn.user_utterance,
                    agent_tool=tool_name,
                    agent_arguments=arguments,
                    tool_result=record.get("result"),
                    tool_error=record.get("error"),
                    expected_tool=turn.expected_tool,
                    expected_arguments=resolved_expected_args,
                    conversation_history=conversation_history[-3:],
                )

                record["verification"] = "llm_judge"
                record["judge_used"] = True
                record["judge_score"] = judge_result["score"]
                record["judge_rationale"] = judge_result["rationale"]
                record["judge_pass"] = judge_result["pass"]
                record["token_usage"]["judge"] = judge_result["token_usage"]
                tool_success = bool(judge_result["pass"])
            # If execution failed, tool_success remains False (exact match failed + execution failed)

        record["tool_success"] = tool_success

        expected_response = turn.expected_response
        agent_response_text = (agent_call.response_text or "").strip()
        if agent_response_text:
            agent_response_text = _resolve_response_text(
                agent_response_text,
                previous_turn_outputs,
                record.get("result"),
                turn.turn_id,
            )
            record["agent_response_text"] = agent_response_text

        response_success = True
        if expected_response:
            record["response_evaluation"] = expected_response.evaluation
            record["response_verification"] = expected_response.evaluation
            if not expected_response.answers and not expected_response.text:
                response_success = tool_success
                record["response_verification"] = "unspecified"
            elif expected_response.requires_judge or expected_response.evaluation == "judge":
                if self._judge:
                    judge_response = self._judge.judge_response(
                        user_utterance=turn.user_utterance,
                        expected_response=expected_response.to_dict(),
                        agent_response=agent_response_text,
                        tool_result=record.get("result"),
                        conversation_history=conversation_history[-3:],
                    )
                    record["response_judge_used"] = True
                    record["response_judge_score"] = judge_response["score"]
                    record["response_judge_rationale"] = judge_response["rationale"]
                    record["response_judge_pass"] = judge_response["pass"]
                    record["token_usage"]["judge_response"] = judge_response["token_usage"]
                    response_success = bool(judge_response["pass"])
                else:
                    record["response_judge_used"] = False
                    record["response_judge_reason"] = "judge_unavailable"
                    response_success = tool_success
                    record["response_judge_pass"] = response_success
            else:
                # Try exact match first
                allowed = { _normalise_response_text(answer) for answer in expected_response.answers }
                if expected_response.text:
                    allowed.add(_normalise_response_text(expected_response.text))
                normalized_agent = _normalise_response_text(agent_response_text)
                exact_match = bool(agent_response_text) and normalized_agent in allowed
                
                # If exact match fails but we have a judge and agent provided response_text,
                # use judge for more lenient evaluation (quality evaluation setup)
                if not exact_match and self._judge and agent_response_text:
                    judge_response = self._judge.judge_response(
                        user_utterance=turn.user_utterance,
                        expected_response=expected_response.to_dict(),
                        agent_response=agent_response_text,
                        tool_result=record.get("result"),
                        conversation_history=conversation_history[-3:],
                    )
                    record["response_judge_used"] = True
                    record["response_judge_score"] = judge_response["score"]
                    record["response_judge_rationale"] = judge_response["rationale"]
                    record["response_judge_pass"] = judge_response["pass"]
                    record["token_usage"]["judge_response"] = judge_response["token_usage"]
                    record["response_verification"] = "llm_judge"
                    response_success = bool(judge_response["pass"])
                else:
                    response_success = exact_match
                    record["response_expected_answers"] = list(expected_response.answers or [])
        else:
            record["response_verification"] = "structured"
            response_success = tool_success

        record["response_success"] = response_success

        combined_success = bool(tool_success and response_success)
        record["success"] = combined_success
        if not combined_success and record.get("verification") == "exact_match":
            record["verification"] = "failed"
        return TurnProcessOutcome(record=_to_primitive(record), success=combined_success)

    # ------------------------------------------------------------------
    def _create_backend(self) -> Union[MockCrmApi, PostgresCrmBackend]:
        if self._backend_mode == "postgres":
            config = self._db_config or DatabaseConfig.from_env()
            backend = PostgresCrmBackend(config)
            backend.begin_session(reset=True)
            return backend
        return MockCrmApi()

    @staticmethod
    def _teardown_backend(api: Union[MockCrmApi, PostgresCrmBackend]) -> None:
        if isinstance(api, PostgresCrmBackend):
            api.rollback_session()
            api.close()

    @staticmethod
    def _seed_backend(api: Union[MockCrmApi, PostgresCrmBackend], initial_entities: Mapping[str, Any]) -> None:
        seed_data = initial_entities.get("seed_data") if isinstance(initial_entities, Mapping) else None
        if not seed_data and not isinstance(api, PostgresCrmBackend):
            return
        if isinstance(api, PostgresCrmBackend):
            ConversationHarness._seed_postgres_backend(api, seed_data or {})
            return

        first_client_id: Optional[str] = None
        client_pool = seed_data.get("Client")
        if client_pool:
            first_client_id = next(iter(client_pool.keys()), None)

        for entity_type in ENTITY_TYPE_ORDER:
            entity_records = seed_data.get(entity_type, {})
            if not entity_records:
                continue
            builder = _get_entity_builder(entity_type)
            if builder is None:
                continue
            store = getattr(api, API_STORE_ATTR[entity_type])
            for entity_id, metadata in entity_records.items():
                fallback_client = metadata.get("client_id") or first_client_id
                model = builder(entity_id, metadata, fallback_client)
                store[entity_id] = model

    @staticmethod
    def _seed_postgres_backend(api: PostgresCrmBackend, seed_data: Mapping[str, Any]) -> None:
        if not seed_data:
            return
        first_client_id: Optional[str] = None
        for entity_type in ENTITY_TYPE_ORDER:
            entity_records = seed_data.get(entity_type, {})
            if not entity_records:
                continue
            builder = _get_entity_builder(entity_type)
            if builder is None:
                continue
            for entity_id, metadata in entity_records.items():
                fallback_client = metadata.get("client_id") or first_client_id
                model = builder(entity_id, metadata, fallback_client)
                ConversationHarness._insert_entity_postgres(api, entity_type, model)
                if entity_type == "Client" and first_client_id is None:
                    first_client_id = entity_id

    @staticmethod
    def _insert_entity_postgres(api: PostgresCrmBackend, entity_type: str, model: Any) -> None:
        mapping = POSTGRES_ENTITY_COLUMN_MAP.get(entity_type)
        if not mapping:
            return
        table, columns = mapping
        payload = model.model_dump()
        params: Dict[str, Any] = {}
        for column in columns:
            value = payload.get(column)
            if isinstance(value, Enum):
                value = value.value
            params[column] = value
        if entity_type == "Opportunity" and params.get("probability") is not None:
            try:
                params["probability"] = int(float(params["probability"]))
            except (TypeError, ValueError):
                params["probability"] = None
        query = (
            f"INSERT INTO {table} ({', '.join(columns)}) "
            f"VALUES ({', '.join('%({})s'.format(col) for col in columns)}) "
            f"ON CONFLICT ({columns[0]}) DO NOTHING;"
        )
        api._execute(query, params)

    def _load_existing_results(self) -> Dict[str, ConversationResult]:
        """Load existing results from output file for resume support."""
        existing_results: Dict[str, ConversationResult] = {}
        output_path = Path(self._output_path)
        if not output_path.exists():
            return existing_results
        
        try:
            with output_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Reconstruct ConversationResult from dict
                        result = ConversationResult(**data)
                        existing_results[result.conversation_id] = result
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        logger.warning("Failed to parse result line, skipping: %s", exc)
                        continue
        except Exception as exc:
            logger.warning("Failed to load existing results: %s", exc)
        
        return existing_results

    def _write_results(self, results: Iterable[ConversationResult]) -> None:
        """Write results to file (legacy method, now handled incrementally in run())."""
        # This method is kept for backward compatibility but results are now
        # written incrementally in run() for crash recovery support
        output_path = Path(self._output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(_to_primitive(result.__dict__), ensure_ascii=False) + "\n")


def _aggregate_token_usage(per_turn: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Aggregate token usage metrics from per-turn records."""
    totals: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    observed = False
    for record in per_turn:
        usage = record.get("token_usage") or {}
        for key in totals:
            value = usage.get(key)
            if value is None:
                continue
            try:
                totals[key] += int(value)
                observed = True
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                logger.debug("Ignoring non-numeric token usage entry %r for key %s", value, key)
    return totals if observed else {}


def _resolve_response_text(
    template: str,
    previous_turns: Mapping[int, Mapping[str, Any]],
    current_result: Mapping[str, Any] | None,
    turn_id: int,
) -> str:
    if "{{turn_" not in template:
        return template
    context = dict(previous_turns)
    if isinstance(current_result, Mapping):
        context[turn_id] = current_result
    resolved = resolve_template(
        {"text": template},
        context,
        turn_number=turn_id,
        strict=False,
        allow_current_turn=True,
    )
    value = resolved.get("text") if isinstance(resolved, dict) else resolved
    return value if isinstance(value, str) else template


def build_regular_conversation_result(
    *,
    conversation: Conversation,
    per_turn: Sequence[Mapping[str, Any]],
    first_failed_turn: Optional[int],
    judge_enabled: bool,
    agent_provider: str,
    agent_model: str,
) -> ConversationResult:
    """Construct a ConversationResult for non-chained conversations."""
    success_path_turns = [t for t in per_turn if t.get("expect_success", True)]
    expected_failure_turns = [t for t in per_turn if not t.get("expect_success", True)]
    successful_success_turns = sum(1 for t in success_path_turns if t.get("success", False))
    tool_successful_turns = sum(1 for t in success_path_turns if t.get("tool_success", False))
    response_successful_turns = sum(1 for t in success_path_turns if t.get("response_success", False))
    total_success_path = len(success_path_turns)
    overall_success = (total_success_path == 0) or (successful_success_turns == total_success_path)
    reward_signal = (successful_success_turns / total_success_path) if total_success_path > 0 else 1.0
    tool_success_rate = (tool_successful_turns / total_success_path) if total_success_path > 0 else 1.0
    response_success_rate = (response_successful_turns / total_success_path) if total_success_path > 0 else 1.0
    observed_expected_failure = any(
        (not turn.get("expect_success", True)) and (not turn.get("success", True)) for turn in per_turn
    )
    expected_failure_failure_turn = next(
        (
            turn["turn_id"]
            for turn in per_turn
            if (not turn.get("expect_success", True)) and (not turn.get("success", True))
        ),
        None,
    )
    if observed_expected_failure:
        overall_success = False
        if first_failed_turn is None and expected_failure_failure_turn is not None:
            first_failed_turn = expected_failure_failure_turn

    metadata = {
        "verification_mode": "hybrid_semantic_llm_judge"
        if judge_enabled
        else conversation.verification_mode.value,
        "success_path_turns": total_success_path,
        "success_path_succeeded": successful_success_turns,
        "task_success_rate": reward_signal,
        "tool_success_rate": tool_success_rate,
        "response_success_rate": response_success_rate,
        "combined_success_rate": reward_signal,
        "expected_failure_turns": len(expected_failure_turns),
        "total_turns": len(per_turn),
        "exact_match_count": sum(1 for t in per_turn if t.get("verification") == "exact_match"),
        "judge_evaluated_count": sum(1 for t in per_turn if t.get("judge_used", False)),
        "judge_approved_count": sum(1 for t in per_turn if t.get("judge_pass", False)),
        "response_judge_evaluated_count": sum(1 for t in per_turn if t.get("response_judge_used", False)),
        "response_judge_approved_count": sum(1 for t in per_turn if t.get("response_judge_pass", False)),
        "agent": {
            "provider": agent_provider,
            "model": agent_model,
        },
        "expected_failure": observed_expected_failure,
    }

    token_totals = _aggregate_token_usage(per_turn)
    if token_totals:
        metadata["agent"]["token_usage"] = token_totals

    overall_failed_turn = None if overall_success else first_failed_turn
    return ConversationResult(
        conversation_id=conversation.conversation_id,
        overall_success=overall_success,
        turns_executed=len(per_turn),
        failed_at_turn=overall_failed_turn,
        per_turn_results=list(per_turn),
        reward_signal=reward_signal,
        metadata=metadata,
    )


def build_chain_conversation_result(
    *,
    conversation: Conversation,
    per_turn: Sequence[Mapping[str, Any]],
    first_failed_turn: Optional[int],
    segment_boundaries: Sequence[int],
    segment_meta_lookup: Optional[Mapping[int, Mapping[str, Any]]] = None,
    judge_enabled: bool,
    agent_provider: str,
    agent_model: str,
) -> ConversationResult:
    """Construct a ConversationResult for chained conversations."""
    if not segment_boundaries:
        raise ValueError(
            f"Chained conversation {conversation.conversation_id} is missing segment boundaries."
        )

    per_segment: List[Dict[str, Any]] = []
    start_turn = 1
    meta_lookup = segment_meta_lookup or {}
    for index, end_turn in enumerate(segment_boundaries):
        segment_turns = [pt for pt in per_turn if start_turn <= pt["turn_id"] <= end_turn]
        success_path_turns = [pt for pt in segment_turns if pt.get("expect_success", True)]
        expected_failure_turns_segment = [pt for pt in segment_turns if not pt.get("expect_success", True)]

        success_path_failure_turn = next(
            (pt["turn_id"] for pt in success_path_turns if not pt.get("success", False)),
            None,
        )
        success_path_failure_error = next(
            (pt.get("error") for pt in success_path_turns if not pt.get("success", False)),
            None,
        )
        observed_failure_turn = next(
            (pt["turn_id"] for pt in expected_failure_turns_segment if not pt.get("success", True)),
            None,
        )
        observed_failure_error = next(
            (pt.get("error") for pt in expected_failure_turns_segment if not pt.get("success", True)),
            None,
        )

        if expected_failure_turns_segment:
            segment_success = observed_failure_turn is None
            failure_turn = observed_failure_turn
            failure_error = observed_failure_error
        else:
            segment_success = all(pt.get("success", False) for pt in success_path_turns)
            failure_turn = success_path_failure_turn
            failure_error = success_path_failure_error

        actual_outcome = "success" if segment_success else "failure"

        expected_meta = meta_lookup.get(index + 1, {})
        expected_outcome = str(expected_meta.get("expected_outcome", "success")).lower()
        if expected_outcome not in ("success", "failure"):
            expected_outcome = "success"

        if expected_outcome != actual_outcome:
            location = f" at turn {failure_turn}" if failure_turn is not None else ""
            warnings.warn(
                f"Segment {index + 1} in {conversation.conversation_id} expected '{expected_outcome}' "
                f"but observed '{actual_outcome}'{location}. Continuing evaluation."
            )

        per_segment.append(
            {
                "segment_number": index + 1,
                "start_turn": start_turn,
                "end_turn": end_turn,
                "turns_attempted": len(segment_turns),
                "successful_turns": sum(1 for pt in success_path_turns if pt.get("success", False)),
                "success": segment_success,
                "actual_outcome": actual_outcome,
                "expected_outcome": expected_outcome,
                "expected_failure": expected_outcome == "failure",
                "failed_at_turn": failure_turn,
                "error": failure_error,
                "turn_ids": [pt["turn_id"] for pt in segment_turns],
                "expected_metadata": dict(expected_meta),
            }
        )
        start_turn = end_turn + 1

    success_path_turns = [t for t in per_turn if t.get("expect_success", True)]
    expected_failure_turns = [t for t in per_turn if not t.get("expect_success", True)]
    successful_success_turns = sum(1 for t in success_path_turns if t.get("success", False))
    tool_successful_turns = sum(1 for t in success_path_turns if t.get("tool_success", False))
    response_successful_turns = sum(1 for t in success_path_turns if t.get("response_success", False))
    total_success_path = len(success_path_turns)
    overall_success = (total_success_path == 0) or (successful_success_turns == total_success_path)
    reward_signal = (successful_success_turns / total_success_path) if total_success_path > 0 else 1.0
    tool_success_rate = (tool_successful_turns / total_success_path) if total_success_path > 0 else 1.0
    response_success_rate = (response_successful_turns / total_success_path) if total_success_path > 0 else 1.0
    chain_success = all(segment["actual_outcome"] == segment["expected_outcome"] for segment in per_segment)
    observed_expected_failure = any(
        (not turn.get("expect_success", True)) and (not turn.get("success", True)) for turn in per_turn
    )
    expected_failure_failure_turn = next(
        (
            turn["turn_id"]
            for turn in per_turn
            if (not turn.get("expect_success", True)) and (not turn.get("success", True))
        ),
        None,
    )
    if observed_expected_failure:
        overall_success = False
        if first_failed_turn is None and expected_failure_failure_turn is not None:
            first_failed_turn = expected_failure_failure_turn
    chain_success = chain_success and not observed_expected_failure

    metadata = {
        "verification_mode": "hybrid_semantic_llm_judge"
        if judge_enabled
        else conversation.verification_mode.value,
        "chain_id": conversation.chain_id,
        "success_path_turns": total_success_path,
        "success_path_succeeded": successful_success_turns,
        "task_success_rate": reward_signal,
        "tool_success_rate": tool_success_rate,
        "response_success_rate": response_success_rate,
        "combined_success_rate": reward_signal,
        "expected_failure_turns": len(expected_failure_turns),
        "total_turns": len(per_turn),
        "segments": len(segment_boundaries),
        "exact_match_count": sum(1 for t in per_turn if t.get("verification") == "exact_match"),
        "judge_evaluated_count": sum(1 for t in per_turn if t.get("judge_used", False)),
        "judge_approved_count": sum(1 for t in per_turn if t.get("judge_pass", False)),
        "response_judge_evaluated_count": sum(1 for t in per_turn if t.get("response_judge_used", False)),
        "response_judge_approved_count": sum(1 for t in per_turn if t.get("response_judge_pass", False)),
        "agent": {
            "provider": agent_provider,
            "model": agent_model,
        },
        "expected_failure": observed_expected_failure,
    }
    token_totals = _aggregate_token_usage(per_turn)
    if token_totals:
        metadata["agent"]["token_usage"] = token_totals

    overall_failed_turn = None if overall_success else first_failed_turn
    return ConversationResult(
        conversation_id=conversation.conversation_id,
        overall_success=overall_success,
        turns_executed=len(per_turn),
        failed_at_turn=overall_failed_turn,
        per_turn_results=list(per_turn),
        reward_signal=reward_signal,
        metadata=metadata,
        per_segment_results=per_segment,
        chain_success=chain_success,
    )


def load_conversations_from_jsonl(path: Path) -> List[Conversation]:
    conversations: List[Conversation] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            verification_str = payload.get("verification_mode", VerificationMode.DATABASE.value)
            turns = [
                ConversationTurn(
                    turn_id=turn["turn_id"],
                    user_utterance=turn.get("user_utterance", ""),
                    expected_tool=turn["expected_tool"],
                    expected_args=turn.get("expected_args", {}),
                    references_previous_turns=turn.get("references_previous_turns", []),
                    expect_success=turn.get("expect_success", True),
                    expected_error_substring=turn.get("expected_error_substring"),
                    failure_category=turn.get("failure_category"),
                    expected_response=turn.get("expected_response"),
                )
                for turn in payload.get("turns", [])
            ]
            conversation = Conversation(
                conversation_id=payload["conversation_id"],
                workflow_category=payload.get("workflow_category", "Unknown"),
                complexity_level=payload.get("complexity_level", "simple"),
                turns=turns,
                initial_entities=payload.get("initial_entities", {}),
                final_expected_state=payload.get("final_expected_state", {}),
                success_criteria=payload.get("success_criteria", "all_turns"),
                contains_failure=payload.get("contains_failure", False),
                failure_turn=payload.get("failure_turn"),
                verification_mode=VerificationMode(verification_str),
                chain_id=payload.get("chain_id"),
                segment_number=payload.get("segment_number"),
                segment_boundaries=payload.get("segment_boundaries"),
                expected_outcome=payload.get("expected_outcome"),
                cumulative_context=payload.get("cumulative_context", {}),
            )
            conversations.append(conversation)
    return conversations
