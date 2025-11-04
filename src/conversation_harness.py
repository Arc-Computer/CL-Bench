"""Baseline harness for running multi-turn CRM conversations through LLM agents.

This module extends the baseline harness to support multi-turn Conversation objects
with conversation history preservation, template resolution, and fail-fast behavior.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from .crm_backend import DatabaseConfig, PostgresCrmBackend
from .crm_sandbox import MockCrmApi
from .conversation_schema import Conversation, ConversationTurn, ConversationResult
from .harness import Agent, ClaudeAgent, OpenAIAgent, MockAgent, ToolCall, EpisodeLog, _parse_tool_calls, _generate_tool_catalog
from .reference_resolver import resolve_template, TemplateResolutionError
from .validators import CrmStateSnapshot, ValidationResult, VerificationMode, get_task_verification_mode
from .verifier import VerifierRequest, VerifierResult, get_registered_verifier, ToolTrace as VerifierToolTrace

logger = logging.getLogger(__name__)


@dataclass
class TurnExecutionResult:
    """Result from executing a single turn in a conversation.
    
    Attributes:
        turn_id: Turn number that was executed
        tool_call: Tool call made by agent
        execution_result: Result from executing the tool
        validation_result: Result from validating against expected args
        pre_state: CRM state before turn execution
        post_state: CRM state after turn execution
        verifier_result: Optional verifier result if enabled
    """
    turn_id: int
    tool_call: ToolCall
    execution_result: ValidationResult
    validation_result: ValidationResult
    pre_state: CrmStateSnapshot
    post_state: CrmStateSnapshot
    verifier_result: Optional[VerifierResult] = None


class ConversationMockAgent:
    """Mock agent that returns ground-truth for ConversationTurn objects."""

    provider_name = "mock"
    model_name = "ground_truth"

    def tool_call(self, turn: ConversationTurn, prompt: str) -> ToolCall:
        """Return ground-truth tool call from conversation turn."""
        return ToolCall(
            tool_name=turn.expected_tool,
            arguments=turn.expected_args,
            raw_response=json.dumps({"tool_name": turn.expected_tool, "arguments": turn.expected_args})
        )


class ConversationHarness:
    """Execute multi-turn conversations against specified agent.
    
    Handles:
    - Sequential turn execution with history preservation
    - Template resolution ({{turn_N.field}} references)
    - Fail-fast behavior based on success_criteria
    - Per-turn telemetry generation for Atlas
    """

    def __init__(
        self,
        conversations: Sequence[Conversation],
        agent: Union[Agent, ConversationMockAgent],
        log_path: Union[str, Path],
        backend: Literal["mock", "postgres"] = "mock",
        db_config: Optional[DatabaseConfig] = None,
        reset_database_each_case: bool = True,
        enable_verifier: bool = False,
        verifier_name: str = "structured",
        verifier_reward_weight: float = 0.3,
    ):
        self.conversations = conversations
        self.agent = agent
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._backend_mode = backend
        self._db_backend = None
        if backend == "postgres":
            self._db_backend = PostgresCrmBackend(db_config or DatabaseConfig.from_env())
        self._reset_database_each_case = reset_database_each_case

        self._verifier_enabled = enable_verifier
        self._verifier_name = verifier_name
        self._verifier_reward_weight = float(max(0.0, min(1.0, verifier_reward_weight)))

    def _setup_initial_entities(
        self,
        conversation: Conversation,
        backend: Union[MockCrmApi, PostgresCrmBackend]
    ) -> None:
        """Create entities from conversation.initial_entities in the backend."""
        initial = conversation.initial_entities

        # Create client if needed
        if "client_id" in initial:
            client_id = initial["client_id"]
            if hasattr(backend, "clients") and client_id not in backend.clients:
                backend.create_new_client(
                    client_id=client_id,
                    name=initial.get("client_name", f"Client {client_id[:8]}"),
                    industry=initial.get("industry", "Technology"),
                    status=initial.get("client_status", "Active"),
                    email=initial.get("client_email"),
                    phone=initial.get("client_phone"),
                    address=initial.get("client_address"),
                    owner=initial.get("client_owner", "sales@example.com"),
                )

        # Create opportunity if needed
        if "opportunity_id" in initial:
            opp_id = initial["opportunity_id"]
            if hasattr(backend, "opportunities") and opp_id not in backend.opportunities:
                backend.create_new_opportunity(
                    opportunity_id=opp_id,
                    name=initial.get("opportunity_name", f"Opportunity {opp_id[:8]}"),
                    client_id=initial.get("client_id"),
                    stage=initial.get("opportunity_stage", "Prospecting"),
                    amount=initial.get("opportunity_amount", 100000.0),
                    owner=initial.get("opportunity_owner", "sales@example.com"),
                    probability=initial.get("probability", 50),
                )

        # Create quote if needed
        if "quote_id" in initial:
            quote_id = initial["quote_id"]
            if hasattr(backend, "quotes") and quote_id not in backend.quotes:
                backend.create_quote(
                    quote_id=quote_id,
                    opportunity_id=initial.get("opportunity_id"),
                    amount=initial.get("quote_amount", 100000.0),
                    status=initial.get("quote_status", "Draft"),
                )

        # Create contact if needed
        if "contact_id" in initial:
            contact_id = initial["contact_id"]
            if hasattr(backend, "contacts") and contact_id not in backend.contacts:
                backend.create_new_contact(
                    contact_id=contact_id,
                    first_name=initial.get("first_name", "John"),
                    last_name=initial.get("last_name", "Doe"),
                    client_id=initial.get("client_id"),
                    title=initial.get("title"),
                    email=initial.get("contact_email"),
                    phone=initial.get("contact_phone"),
                )

    def _build_conversation_prompt(
        self,
        turn: ConversationTurn,
        turn_history: List[Tuple[ConversationTurn, TurnExecutionResult]],
        backend: Union[MockCrmApi, PostgresCrmBackend]
    ) -> str:
        """Build prompt with conversation history."""
        # Format conversation history
        history_parts = []
        for prev_turn, prev_result in turn_history:
            history_parts.append(f"User: {prev_turn.user_utterance}")
            history_parts.append(
                f"Assistant: {prev_result.tool_call.tool_name}("
                f"{json.dumps(prev_result.tool_call.arguments)})"
            )

        # Get entity context (similar to scenario harness)
        context_parts = []
        if hasattr(backend, "clients") and backend.clients:
            client_count = len(backend.clients) if isinstance(backend.clients, dict) else len(list(backend.clients))
            context_parts.append(f"- Clients: {client_count} available")
        if hasattr(backend, "opportunities") and backend.opportunities:
            opp_count = len(backend.opportunities) if isinstance(backend.opportunities, dict) else len(list(backend.opportunities))
            context_parts.append(f"- Opportunities: {opp_count} available")
        if hasattr(backend, "quotes") and backend.quotes:
            quote_count = len(backend.quotes) if isinstance(backend.quotes, dict) else len(list(backend.quotes))
            context_parts.append(f"- Quotes: {quote_count} available")

        context_str = "\n".join(context_parts) if context_parts else "No entities available."

        # Generate complete tool catalog
        catalog = _generate_tool_catalog()

        # Build full prompt
        prompt_parts = [
            "# CRM Assistant - Multi-Turn Conversation",
            "",
            "## Conversation History:",
            "\n".join(history_parts) if history_parts else "(No previous turns)",
            "",
            f"## Current User Request:",
            turn.user_utterance,
            "",
            "## Available CRM Entities:",
            context_str,
            "",
            "## Complete API Reference:",
            catalog,
            "",
            "## Response Format:",
            "Return a JSON object with tool_name and arguments fields.",
        ]

        return "\n".join(prompt_parts)

    def _execute_tool(
        self,
        backend: Union[MockCrmApi, PostgresCrmBackend],
        tool_call: ToolCall
    ) -> ValidationResult:
        """Execute a single tool call against backend and return result."""
        try:
            if not hasattr(backend, tool_call.tool_name):
                return ValidationResult.fail(f"Tool '{tool_call.tool_name}' not found in backend")
            
            method = getattr(backend, tool_call.tool_name)
            result = method(**tool_call.arguments)
            return ValidationResult.ok("Tool executed successfully", {"result": result})
        except Exception as e:
            return ValidationResult.fail(str(e), {"exception": repr(e)})

    def _execute_tools(
        self,
        backend: Union[MockCrmApi, PostgresCrmBackend],
        tool_calls: List[ToolCall]
    ) -> Tuple[ValidationResult, Optional[Any]]:
        """Execute multiple tool calls sequentially with fail-fast error handling.
        
        Returns:
            Tuple of (ValidationResult, last_tool_result)
            last_tool_result is the actual return value from the last tool call (if successful)
        """
        if not tool_calls:
            return ValidationResult.fail("No tool calls provided"), None
        
        last_result = None
        for idx, tool_call in enumerate(tool_calls):
            result = self._execute_tool(backend, tool_call)
            if not result.success:
                return ValidationResult.fail(
                    f"Tool '{tool_call.tool_name}' failed: {result.message}",
                    {"failed_tool": tool_call.tool_name, "tool_index": idx}
                ), None
            
            # Extract the actual result from the ValidationResult
            if result.details and "result" in result.details:
                last_result = result.details["result"]
        
        return ValidationResult.ok("All tools executed successfully", {"tool_count": len(tool_calls), "result": last_result}), last_result

    def _validate_turn_result(
        self,
        turn: ConversationTurn,
        tool_call: ToolCall,
        execution_result: ValidationResult,
        resolved_args: Dict[str, Any],
        pre: CrmStateSnapshot,
        post: CrmStateSnapshot,
    ) -> ValidationResult:
        """Validate turn result based on expected success/failure."""
        tool_correct = tool_call.tool_name == turn.expected_tool

        if turn.expect_success:
            if not tool_correct:
                return ValidationResult.fail(
                    f"Expected tool '{turn.expected_tool}' but agent called '{tool_call.tool_name}'"
                )
            if not execution_result.success:
                return ValidationResult.fail(f"Tool execution failed: {execution_result.message}")
            return ValidationResult.ok("Turn passed - tool executed successfully")
        else:
            # Expect failure
            if not tool_correct:
                return ValidationResult.fail(
                    f"Expected tool '{turn.expected_tool}' but agent called '{tool_call.tool_name}'"
                )
            if execution_result.success:
                return ValidationResult.fail("Expected failure but tool executed successfully")

            # Check expected error substring
            if turn.expected_error_substring:
                if turn.expected_error_substring not in execution_result.message:
                    return ValidationResult.fail(
                        f"Expected error substring '{turn.expected_error_substring}' not found in: {execution_result.message}"
                    )

            return ValidationResult.ok(f"Turn passed - expected failure occurred: {execution_result.message}")

    def _extract_result_fields(
        self,
        tool_call: ToolCall,
        execution_result: ValidationResult,
        backend: Union[MockCrmApi, PostgresCrmBackend],
        tool_result: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Extract entity IDs and fields from tool execution result for template resolution.
        
        Args:
            tool_call: The tool call that was executed
            execution_result: ValidationResult from execution
            backend: Backend instance (for fallback extraction)
            tool_result: Optional direct result from tool execution (preferred over parsing from ValidationResult)
        
        Returns:
            Dictionary with fields that can be referenced by {{turn_N.field}} templates.
        """
        result = {}
        
        # Extract IDs from tool call arguments (for create operations)
        for key in ["client_id", "contact_id", "opportunity_id", "quote_id", "contract_id"]:
            if key in tool_call.arguments:
                result[key] = tool_call.arguments[key]

        # Extract from tool_result directly if provided (most reliable)
        if tool_result is not None:
            result.update(self._extract_ids_from_result(tool_result))
        # Fall back to execution_result.details if tool_result not available
        elif execution_result.success and execution_result.details:
            exec_result = execution_result.details.get("result")
            if exec_result:
                result.update(self._extract_ids_from_result(exec_result))

        # For search operations, if no results found yet, try to get from backend state
        if tool_call.tool_name.endswith("_search") and not result:
            # If search returned empty, try to find matching entity in backend
            if tool_call.tool_name == "client_search" and hasattr(backend, "clients"):
                # Try to find client by name or ID from search criteria
                search_name = tool_call.arguments.get("name")
                search_client_id = tool_call.arguments.get("client_id")
                
                if backend.clients:
                    # First, try exact ID match if provided
                    if search_client_id:
                        clients_dict = backend.clients if isinstance(backend.clients, dict) else {c.client_id: c for c in backend.clients}
                        if search_client_id in clients_dict:
                            result["client_id"] = search_client_id
                    
                    # If no ID match, try name match
                    if "client_id" not in result and search_name:
                        for client in (backend.clients.values() if isinstance(backend.clients, dict) else backend.clients):
                            if hasattr(client, "name") and search_name.lower() in client.name.lower():
                                result["client_id"] = client.client_id
                                break
                    
                    # If still no match, use first available client
                    if "client_id" not in result:
                        first_client = next(iter(backend.clients.values() if isinstance(backend.clients, dict) else backend.clients), None)
                        if first_client:
                            result["client_id"] = first_client.client_id
                            
            elif tool_call.tool_name == "opportunity_search" and hasattr(backend, "opportunities"):
                client_id_filter = tool_call.arguments.get("client_id")
                search_name = tool_call.arguments.get("name")
                
                if backend.opportunities:
                    # Try to find by client_id filter
                    if client_id_filter:
                        for opp in (backend.opportunities.values() if isinstance(backend.opportunities, dict) else backend.opportunities):
                            if hasattr(opp, "client_id") and opp.client_id == client_id_filter:
                                result["opportunity_id"] = opp.opportunity_id
                                break
                    
                    # Try name match
                    if "opportunity_id" not in result and search_name:
                        for opp in (backend.opportunities.values() if isinstance(backend.opportunities, dict) else backend.opportunities):
                            if hasattr(opp, "name") and search_name.lower() in opp.name.lower():
                                result["opportunity_id"] = opp.opportunity_id
                                break
                    
                    # Fallback to first opportunity
                    if "opportunity_id" not in result:
                        first_opp = next(iter(backend.opportunities.values() if isinstance(backend.opportunities, dict) else backend.opportunities), None)
                        if first_opp:
                            result["opportunity_id"] = first_opp.opportunity_id
                            
            elif tool_call.tool_name == "quote_search" and hasattr(backend, "quotes"):
                opportunity_id_filter = tool_call.arguments.get("opportunity_id")
                
                if backend.quotes:
                    if opportunity_id_filter:
                        for quote in (backend.quotes.values() if isinstance(backend.quotes, dict) else backend.quotes):
                            if hasattr(quote, "opportunity_id") and quote.opportunity_id == opportunity_id_filter:
                                result["quote_id"] = quote.quote_id
                                break
                    
                    if "quote_id" not in result:
                        first_quote = next(iter(backend.quotes.values() if isinstance(backend.quotes, dict) else backend.quotes), None)
                        if first_quote:
                            result["quote_id"] = first_quote.quote_id

        return result

    def _extract_ids_from_result(self, exec_result: Any) -> Dict[str, Any]:
        """Extract entity IDs from a tool execution result.
        
        Handles various return types: dict, list, Pydantic models, dataclasses.
        """
        result = {}
        
        if isinstance(exec_result, dict):
            for key in ["client_id", "contact_id", "opportunity_id", "quote_id", "contract_id"]:
                if key in exec_result:
                    result[key] = exec_result[key]
        elif hasattr(exec_result, "__dict__"):
            # Pydantic model or dataclass
            for key in ["client_id", "contact_id", "opportunity_id", "quote_id", "contract_id"]:
                if hasattr(exec_result, key):
                    result[key] = getattr(exec_result, key)
        elif isinstance(exec_result, list) and len(exec_result) > 0:
            # Search results return a list - extract ID from first result
            first_result = exec_result[0]
            if isinstance(first_result, dict):
                for key in ["client_id", "contact_id", "opportunity_id", "quote_id", "contract_id"]:
                    if key in first_result:
                        result[key] = first_result[key]
            elif hasattr(first_result, "__dict__"):
                # Pydantic model or dataclass
                for key in ["client_id", "contact_id", "opportunity_id", "quote_id", "contract_id"]:
                    if hasattr(first_result, key):
                        result[key] = getattr(first_result, key)
            # Also check if it's a Client/Opportunity/etc object directly
            elif hasattr(first_result, "client_id"):
                result["client_id"] = first_result.client_id
            elif hasattr(first_result, "opportunity_id"):
                result["opportunity_id"] = first_result.opportunity_id
            elif hasattr(first_result, "contact_id"):
                result["contact_id"] = first_result.contact_id
            elif hasattr(first_result, "quote_id"):
                result["quote_id"] = first_result.quote_id
            elif hasattr(first_result, "contract_id"):
                result["contract_id"] = first_result.contract_id
        
        return result

    def _validate_conversation(
        self,
        conversation: Conversation,
        turn_history: List[Tuple[ConversationTurn, TurnExecutionResult]]
    ) -> ConversationResult:
        """Validate conversation-level success based on success_criteria."""
        all_turns_succeeded = all(turn_result.validation_result.success for _, turn_result in turn_history)
        turns_executed = len(turn_history)
        failed_at_turn = None

        if not all_turns_succeeded:
            # Find first failed turn
            for turn, turn_result in turn_history:
                if not turn_result.validation_result.success:
                    failed_at_turn = turn.turn_id
                    break

        # Determine overall success based on success_criteria
        if conversation.success_criteria == "all_turns":
            overall_success = all_turns_succeeded and turns_executed == len(conversation.turns)
        elif conversation.success_criteria == "final_state":
            # Only final state matters - check if last turn succeeded
            overall_success = turn_history[-1][1].validation_result.success if turn_history else False
        else:  # "both"
            overall_success = all_turns_succeeded and turns_executed == len(conversation.turns)

        # Calculate reward signal (0.0-1.0)
        if overall_success:
            reward_signal = 1.0
        elif turns_executed > 0:
            # Partial credit based on turns completed
            reward_signal = turns_executed / len(conversation.turns)
        else:
            reward_signal = 0.0

        error_message = None
        if failed_at_turn:
            failed_turn_result = next(turn_result for _, turn_result in turn_history if turn_result.turn_id == failed_at_turn)
            error_message = failed_turn_result.validation_result.message

        per_turn_results = [
            {
                "turn_id": turn_result.turn_id,
                "success": turn_result.validation_result.success,
                "message": turn_result.validation_result.message,
            }
            for _, turn_result in turn_history
        ]

        # Adjust turns_executed if failed_at_turn is set
        # turns_executed should be the number of turns that completed successfully
        # If failed_at_turn is N, then turns_executed should be N-1
        if failed_at_turn is not None:
            turns_executed = failed_at_turn - 1

        return ConversationResult(
            conversation_id=conversation.conversation_id,
            overall_success=overall_success,
            turns_executed=turns_executed,
            failed_at_turn=failed_at_turn,
            per_turn_results=per_turn_results,
            reward_signal=reward_signal,
            error_message=error_message,
        )

    def _write_conversation_telemetry(
        self,
        conversation: Conversation,
        turn_history: List[Tuple[ConversationTurn, TurnExecutionResult]],
        conversation_result: ConversationResult,
        log_file: Any,
        mode: str,
    ) -> None:
        """Generate EpisodeLogs for each turn and write to log file."""
        timestamp = datetime.now(timezone.utc).isoformat()
        provider = self.agent.provider_name if hasattr(self.agent, "provider_name") else "unknown"
        model = self.agent.model_name if hasattr(self.agent, "model_name") else "unknown"

        for turn, turn_result in turn_history:
            
            # Build reward breakdown
            base_reward = 1.0 if turn_result.validation_result.success else 0.0
            verifier_contrib = 0.0
            if turn_result.verifier_result:
                verifier_contrib = self._verifier_reward_weight * turn_result.verifier_result.score
            final_reward = (1 - self._verifier_reward_weight) * base_reward + verifier_contrib

            reward_breakdown = {
                "base_reward": base_reward,
                "verifier_contribution": verifier_contrib,
                "final_reward": final_reward,
            }

            episode_log = EpisodeLog(
                case_id=f"{conversation.conversation_id}-turn-{turn_result.turn_id}",
                task=conversation.workflow_category,
                timestamp=timestamp,
                provider=provider,
                model=model,
                success=turn_result.validation_result.success,
                expected_success=turn.expect_success,
                message=turn_result.validation_result.message,
                tool_call={
                    "tool_name": turn_result.tool_call.tool_name,
                    "arguments": turn_result.tool_call.arguments,
                },
                agent_response=turn_result.tool_call.raw_response,
                validator_details=turn_result.validation_result.details,
                expected_tool=turn.expected_tool,
                expected_arguments=turn.expected_args,
                verification_mode=conversation.verification_mode.value,
                verifier_name=turn_result.verifier_result.name if turn_result.verifier_result else None,
                verifier_score=turn_result.verifier_result.score if turn_result.verifier_result else None,
                verifier_rationale=turn_result.verifier_result.rationale if turn_result.verifier_result else None,
                verifier_metadata=turn_result.verifier_result.metadata if turn_result.verifier_result else None,
                reward_breakdown=reward_breakdown,
                learning_signals={
                    "conversation_id": conversation.conversation_id,
                    "turn_id": turn_result.turn_id,
                    "conversation_success": conversation_result.overall_success,
                },
                validator_metadata={
                    "conversation_id": conversation.conversation_id,
                    "turn_id": turn_result.turn_id,
                    "complexity_level": conversation.complexity_level,
                },
            )

            log_file.write(episode_log.to_json() + "\n")

    def run(self, mode: Literal["agent", "mock"] = "agent") -> Dict[str, Any]:
        """Execute all conversations and return results.
        
        Args:
            mode: Execution mode ("agent" for real LLM, "mock" for ground truth)
            
        Returns:
            Dictionary with execution results and statistics
        """
        episodes: List[EpisodeLog] = []
        successes = 0
        failures = 0

        with self.log_path.open("w", encoding="utf-8") as log_file:
            for conversation in self.conversations:
                # Initialize backend
                if self._backend_mode == "postgres":
                    if not self._db_backend:
                        raise RuntimeError("Postgres backend not initialized")
                    backend = self._db_backend
                    backend.begin_session(reset=self._reset_database_each_case)
                else:
                    backend = MockCrmApi()

                try:
                    # Setup initial entities
                    self._setup_initial_entities(conversation, backend)

                    # Track turn results for template resolution
                    turn_results: Dict[int, Dict[str, Any]] = {}
                    turn_history: List[Tuple[ConversationTurn, TurnExecutionResult]] = []

                    # Execute turns sequentially
                    for turn in conversation.turns:
                        # 1. Resolve {{turn_N.field}} templates
                        try:
                            resolved_args = resolve_template(
                                turn.expected_args,
                                turn_results,
                                turn.turn_id,
                                strict=True
                            )
                            
                            # Validate resolved args don't have invalid placeholder values
                            # This catches data generation issues where placeholders weren't populated
                            if "amount" in resolved_args and resolved_args["amount"] == 0.0:
                                # Check if this is a create operation that requires amount
                                if turn.expected_tool in ["create_new_opportunity", "create_quote", "create_contract"]:
                                    logger.warning(
                                        f"Invalid amount 0.0 in {conversation.conversation_id} turn {turn.turn_id} "
                                        f"for {turn.expected_tool}. This is a data generation issue. "
                                        f"Using fallback amount from user utterance or default."
                                    )
                                    # Try to extract amount from user utterance (basic heuristic)
                                    # If utterance contains "$XXXk" or "$XXX,XXX", extract it
                                    amount_match = re.search(r'\$(\d+(?:\.\d+)?)[kK]', turn.user_utterance)
                                    if amount_match:
                                        resolved_args["amount"] = float(amount_match.group(1)) * 1000
                                    else:
                                        amount_match = re.search(r'\$(\d{1,3}(?:,\d{3})+)', turn.user_utterance.replace(',', ''))
                                        if amount_match:
                                            resolved_args["amount"] = float(amount_match.group(1).replace(',', ''))
                                        else:
                                            # Default fallback
                                            resolved_args["amount"] = 100000.0
                            
                            # Clean up resolved_args: remove None values for optional fields
                            # Create methods auto-generate IDs, so None IDs should be removed
                            cleaned_args = {}
                            for k, v in resolved_args.items():
                                if v is None:
                                    # For create operations, remove None IDs - they'll be auto-generated
                                    if turn.expected_tool.startswith("create_"):
                                        if k in ["opportunity_id", "quote_id", "contract_id", "contact_id", "client_id"]:
                                            continue  # Skip None IDs for create operations
                                    # Keep None for operations that might use it (e.g., updates dict)
                                    if k in ["updates"]:
                                        cleaned_args[k] = v
                                        continue
                                    # Skip other None values
                                    continue
                                cleaned_args[k] = v
                            
                            resolved_args = cleaned_args
                            
                        except TemplateResolutionError as e:
                            logger.error(f"Template resolution failed for {conversation.conversation_id} turn {turn.turn_id}: {e}")
                            # Mark conversation as failed
                            conversation_result = ConversationResult(
                                conversation_id=conversation.conversation_id,
                                overall_success=False,
                                turns_executed=len(turn_history),
                                failed_at_turn=turn.turn_id,
                                error_message=f"Template resolution failed: {e}",
                            )
                            failures += 1
                            break

                        # 2. Build prompt with conversation history
                        prompt = self._build_conversation_prompt(turn, turn_history, backend)

                        # 3. Get agent response
                        if mode == "mock":
                            mock_agent = ConversationMockAgent()
                            # Mock agent should use resolved_args, not turn.expected_args (which has templates)
                            # Create a temporary turn with resolved args for the mock agent
                            resolved_turn = ConversationTurn(
                                turn_id=turn.turn_id,
                                user_utterance=turn.user_utterance,
                                expected_tool=turn.expected_tool,
                                expected_args=resolved_args,  # Use resolved args
                                references_previous_turns=turn.references_previous_turns,
                                expect_success=turn.expect_success,
                                expected_error_substring=turn.expected_error_substring,
                                failure_category=turn.failure_category,
                            )
                            tool_call = mock_agent.tool_call(resolved_turn, prompt)
                            tool_calls = [tool_call]
                            raw_response = tool_call.raw_response
                        else:
                            # Adapt agent interface - create a simple wrapper object
                            # Agents expect an object with expected_tool and expected_args attributes
                            class TurnWrapper:
                                def __init__(self, turn: ConversationTurn):
                                    self.expected_tool = turn.expected_tool
                                    self.expected_args = turn.expected_args
                            
                            turn_wrapper = TurnWrapper(turn)
                            
                            # These agents parse prompt directly
                            agent_tool_call = self.agent.tool_call(turn_wrapper, prompt)
                            raw_response = agent_tool_call.raw_response
                            tool_calls = _parse_tool_calls(raw_response)
                            tool_call = tool_calls[0]  # Use first for validation

                        # 4. Execute and validate turn
                        pre = CrmStateSnapshot.from_backend(backend)
                        execution_result, tool_result = self._execute_tools(backend, tool_calls)
                        post = CrmStateSnapshot.from_backend(backend)

                        validation_result = self._validate_turn_result(
                            turn, tool_call, execution_result, resolved_args, pre, post
                        )

                        # 5. Extract result fields for template resolution
                        # Use tool_result directly if available, otherwise fall back to execution_result.details
                        turn_result_fields = self._extract_result_fields(
                            tool_call, execution_result, backend, tool_result=tool_result
                        )
                        turn_results[turn.turn_id] = turn_result_fields

                        # 6. Teacher/Student verification (if enabled)
                        verifier_result: Optional[VerifierResult] = None
                        if self._verifier_enabled and self._verifier_name:
                            tool_traces = tuple(
                                VerifierToolTrace(
                                    step=idx + 1,
                                    tool_name=tc.tool_name,
                                    arguments=dict(tc.arguments),
                                    execution_success=execution_result.success if idx == 0 else True,
                                    validator_success=validation_result.success if idx == 0 else True,
                                    message=validation_result.message if idx == 0 else "Tool executed successfully",
                                )
                                for idx, tc in enumerate(tool_calls)
                            )

                            verifier = get_registered_verifier(self._verifier_name)
                            request = VerifierRequest(
                                case_id=f"{conversation.conversation_id}-turn-{turn.turn_id}",
                                task=conversation.workflow_category,
                                utterance=turn.user_utterance,
                                expect_success=turn.expect_success,
                                expected_error_substring=turn.expected_error_substring,
                                expected_tool=turn.expected_tool,
                                expected_arguments=turn.expected_args,
                                tool_traces=tool_traces,
                                final_response=raw_response,
                                validator_result=validation_result,
                                pre_state=pre,
                                post_state=post,
                            )

                            try:
                                verifier_result = verifier.evaluate(request)
                            except Exception as exc:
                                verifier_result = VerifierResult(
                                    score=0.0,
                                    rationale=f"Verifier exception: {exc}",
                                    metadata={"exception": repr(exc)},
                                )

                        # 7. Store turn history
                        turn_execution_result = TurnExecutionResult(
                            turn_id=turn.turn_id,
                            tool_call=tool_call,
                            execution_result=execution_result,
                            validation_result=validation_result,
                            pre_state=pre,
                            post_state=post,
                            verifier_result=verifier_result,
                        )
                        turn_history.append((turn, turn_execution_result))

                        # 8. Fail-fast check
                        if not validation_result.success:
                            if conversation.success_criteria == "all_turns":
                                break

                    # 9. Validate conversation-level success (only if we didn't break early)
                    if 'conversation_result' not in locals():
                        conversation_result = self._validate_conversation(conversation, turn_history)

                        # 10. Generate and write telemetry
                        self._write_conversation_telemetry(
                            conversation, turn_history, conversation_result, log_file, mode
                        )

                        if conversation_result.overall_success:
                            successes += 1
                        else:
                            failures += 1

                except Exception as e:
                    logger.error(f"Error executing conversation {conversation.conversation_id}: {e}", exc_info=True)
                    failures += 1

        return {
            "total": len(self.conversations),
            "success_count": successes,
            "failure_count": failures,
            "episodes": episodes,
        }


def load_conversations_from_jsonl(path: Path) -> List[Conversation]:
    """Load conversations from JSONL file."""
    conversations = []
    with path.open() as f:
        for line in f:
            data = json.loads(line)
            # Reconstruct ConversationTurn objects
            turns = [
                ConversationTurn(
                    turn_id=turn_data["turn_id"],
                    user_utterance=turn_data["user_utterance"],
                    expected_tool=turn_data["expected_tool"],
                    expected_args=turn_data["expected_args"],
                    references_previous_turns=turn_data.get("references_previous_turns", []),
                    expect_success=turn_data.get("expect_success", True),
                    expected_error_substring=turn_data.get("expected_error_substring"),
                    failure_category=turn_data.get("failure_category"),
                )
                for turn_data in data["turns"]
            ]
            # Convert verification_mode string to enum if needed
            verification_mode = data.get("verification_mode", "database")
            if isinstance(verification_mode, str):
                verification_mode = VerificationMode[verification_mode.upper()]
            
            conversation = Conversation(
                conversation_id=data["conversation_id"],
                workflow_category=data["workflow_category"],
                complexity_level=data["complexity_level"],
                turns=turns,
                initial_entities=data.get("initial_entities", {}),
                final_expected_state=data.get("final_expected_state", {}),
                success_criteria=data.get("success_criteria", "all_turns"),
                contains_failure=data.get("contains_failure", False),
                failure_turn=data.get("failure_turn"),
                verification_mode=verification_mode,
            )
            conversations.append(conversation)
    return conversations

