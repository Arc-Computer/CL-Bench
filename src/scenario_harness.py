"""Baseline harness for running synthetic CRM scenarios through LLM agents.

This module extends the golden-case harness to support Scenario objects with
pre-resolved entity IDs from the Curator dataset generator.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from .crm_backend import DatabaseConfig, PostgresCrmBackend
from .crm_sandbox import MockCrmApi, ClientStatus, OpportunityStage, QuoteStatus, ContractStatus
from .harness import Agent, ClaudeAgent, OpenAIAgent, MockAgent, ToolCall, EpisodeLog
from .scenario_generator import Scenario
from .validators import CrmStateSnapshot, ValidationResult, VerificationMode
from .verifier import VerifierRequest, VerifierResult, get_registered_verifier, ToolTrace as VerifierToolTrace


def load_scenarios_from_jsonl(path: Path) -> List[Scenario]:
    """Load scenarios from JSONL file."""
    scenarios = []
    with path.open() as f:
        for line in f:
            data = json.loads(line)
            scenarios.append(Scenario(**data))
    return scenarios


def build_scenario_prompt(scenario: Scenario, backend: Union[MockCrmApi, PostgresCrmBackend]) -> str:
    """Build prompt for scenario execution.

    Includes:
    - Task description
    - Available tools
    - Entity context (created clients, opportunities, etc.)
    - Schema constraints (enums, required fields)
    """
    # Build entity context
    context_parts = []
    if scenario.setup_entities:
        context_parts.append("Context:")

        if "client_id" in scenario.setup_entities:
            client_id = scenario.setup_entities["client_id"]
            # Try to get client details from backend
            if hasattr(backend, "clients") and client_id in backend.clients:
                client = backend.clients[client_id]
                context_parts.append(f"- Client: {client.name} (ID: {client_id}, Status: {client.status})")
            else:
                context_parts.append(f"- Client ID: {client_id}")

        if "opportunity_id" in scenario.setup_entities:
            opp_id = scenario.setup_entities["opportunity_id"]
            if hasattr(backend, "opportunities") and opp_id in backend.opportunities:
                opp = backend.opportunities[opp_id]
                context_parts.append(f"- Opportunity: {opp.name} (ID: {opp_id}, Stage: {opp.stage}, Amount: ${opp.amount:,.0f})")
            else:
                context_parts.append(f"- Opportunity ID: {opp_id}")

        if "quote_id" in scenario.setup_entities:
            quote_id = scenario.setup_entities["quote_id"]
            if hasattr(backend, "quotes") and quote_id in backend.quotes:
                quote = backend.quotes[quote_id]
                context_parts.append(f"- Quote ID: {quote_id} (Status: {quote.status}, Amount: ${quote.amount:,.0f})")
            else:
                context_parts.append(f"- Quote ID: {quote_id}")

        if "contact_id" in scenario.setup_entities:
            contact_id = scenario.setup_entities["contact_id"]
            if hasattr(backend, "contacts") and contact_id in backend.contacts:
                contact = backend.contacts[contact_id]
                context_parts.append(f"- Contact: {contact.first_name} {contact.last_name} (ID: {contact_id})")
            else:
                context_parts.append(f"- Contact ID: {contact_id}")

        if "contract_id" in scenario.setup_entities:
            contract_id = scenario.setup_entities["contract_id"]
            context_parts.append(f"- Contract ID: {contract_id}")

    context_str = "\n".join(context_parts) if context_parts else "No context entities."

    # Tool directive
    tool_directive = f"""You are a CRM assistant agent. The user will provide a request in natural language.
Your task is to call the appropriate CRM tool with correct arguments.

{context_str}

User request: {scenario.utterance}

Available tools: create_new_client, modify_client, client_search, create_new_opportunity, modify_opportunity,
delete_opportunity, opportunity_search, clone_opportunity, view_opportunity_details, summarize_opportunities,
create_new_contact, modify_contact, contact_search, create_quote, modify_quote, delete_quote, cancel_quote,
quote_search, quote_details, compare_quotes, compare_quote_details, create_contract, contract_search, company_search, add_note, upload_document

Schema constraints:
- OpportunityStage: Prospecting, Qualification, Proposal, Negotiation, Closed-Won, Closed-Lost
- ClientStatus: Active, Prospect, Inactive
- QuoteStatus: Draft, Sent, Approved, Rejected, Canceled
- ContractStatus: Active, Pending, Expired
- CompanyType: Partner, Vendor, Competitor

Respond with a JSON object containing:
{{"tool_name": "...", "arguments": {{...}}}}"""

    return tool_directive


class ScenarioMockAgent:
    """Mock agent that returns ground-truth for Scenario objects."""

    provider_name = "mock"
    model_name = "ground_truth"

    def tool_call(self, scenario: Scenario, prompt: str) -> ToolCall:
        """Return ground-truth tool call from scenario."""
        return ToolCall(
            tool_name=scenario.expected_tool,
            arguments=scenario.expected_args,
            raw_response=json.dumps({"tool_name": scenario.expected_tool, "arguments": scenario.expected_args})
        )


class ScenarioBaselineHarness:
    """Execute synthetic scenarios against specified agent.

    Adapted from BaselineHarness to work with Scenario objects instead of GoldenCase.
    """

    def __init__(
        self,
        scenarios: Sequence[Scenario],
        agent: Union[Agent, ScenarioMockAgent],
        log_path: Union[str, Path],
        backend: Literal["mock", "postgres"] = "mock",
        db_config: Optional[DatabaseConfig] = None,
        reset_database_each_case: bool = True,
        enable_verifier: bool = False,
        verifier_name: str = "structured",
        verifier_reward_weight: float = 0.3,
    ):
        self.scenarios = scenarios
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

    def _create_entities_from_scenario(
        self,
        scenario: Scenario,
        backend: Union[MockCrmApi, PostgresCrmBackend]
    ) -> None:
        """Create entities from scenario.setup_entities in the backend."""
        setup = scenario.setup_entities

        # Create client if needed
        if "client_id" in setup:
            # Check if client already exists
            if hasattr(backend, "clients") and setup["client_id"] not in backend.clients:
                backend.create_new_client(
                    client_id=setup["client_id"],
                    name=setup.get("client_name", f"Client {setup['client_id'][:8]}"),
                    industry=setup.get("industry", "Technology"),
                    status=setup.get("client_status", "Active"),
                    email=setup.get("client_email"),
                    phone=setup.get("client_phone"),
                    address=setup.get("client_address"),
                    owner=setup.get("client_owner", "sales@example.com"),
                )

        # Create opportunity if needed
        if "opportunity_id" in setup:
            if hasattr(backend, "opportunities") and setup["opportunity_id"] not in backend.opportunities:
                # Build kwargs, filtering out None values
                opp_kwargs = {
                    "opportunity_id": setup["opportunity_id"],
                    "name": setup.get("opportunity_name", f"Opportunity {setup['opportunity_id'][:8]}"),
                    "client_id": setup.get("client_id"),
                    "stage": setup.get("opportunity_stage", "Prospecting"),
                    "amount": setup.get("opportunity_amount", 100000.0),
                    "owner": setup.get("opportunity_owner", "sales@example.com"),
                    "probability": setup.get("probability", 50),
                }
                # Only add optional fields if they're not None
                if setup.get("close_date") is not None:
                    opp_kwargs["close_date"] = setup.get("close_date")
                if setup.get("opportunity_notes") is not None:
                    opp_kwargs["notes"] = setup.get("opportunity_notes")
                backend.create_new_opportunity(**opp_kwargs)

        # Create quotes if referenced in expected_args (for scenarios like compare_quote_details)
        quote_ids_from_args = []
        if "quote_ids" in scenario.expected_args:
            quote_ids_from_args = scenario.expected_args["quote_ids"]
        elif "quote_id" in scenario.expected_args:
            quote_ids_from_args = [scenario.expected_args["quote_id"]]

        for quote_id in quote_ids_from_args:
            if hasattr(backend, "quotes") and quote_id not in backend.quotes:
                backend.create_quote(
                    quote_id=quote_id,
                    opportunity_id=setup.get("opportunity_id"),
                    amount=setup.get("opportunity_amount", 100000.0) * 0.95,  # Quote typically ~95% of opportunity
                    status="Sent",
                )

        # Create quote if needed (from setup_entities)
        if "quote_id" in setup:
            if hasattr(backend, "quotes") and setup["quote_id"] not in backend.quotes:
                backend.create_quote(
                    quote_id=setup["quote_id"],
                    opportunity_id=setup.get("opportunity_id"),
                    amount=setup.get("quote_amount", 100000.0),
                    status=setup.get("quote_status", "Draft"),
                    valid_until=setup.get("valid_until"),
                    version=setup.get("version", "1.0"),
                    quote_prefix=setup.get("quote_prefix"),
                )

        # Create contact if needed
        if "contact_id" in setup:
            if hasattr(backend, "contacts") and setup["contact_id"] not in backend.contacts:
                backend.create_new_contact(
                    contact_id=setup["contact_id"],
                    first_name=setup.get("first_name", "John"),
                    last_name=setup.get("last_name", "Doe"),
                    client_id=setup.get("client_id"),
                    title=setup.get("title"),
                    email=setup.get("contact_email"),
                    phone=setup.get("contact_phone"),
                    notes=setup.get("contact_notes"),
                )

        # Create contracts if referenced in expected_args
        contract_ids_from_args = []
        if "contract_ids" in scenario.expected_args:
            contract_ids_from_args = scenario.expected_args["contract_ids"]
        elif "contract_id" in scenario.expected_args:
            contract_ids_from_args = [scenario.expected_args["contract_id"]]

        for contract_id in contract_ids_from_args:
            if hasattr(backend, "contracts") and contract_id not in backend.contracts:
                backend.create_contract(
                    contract_id=contract_id,
                    client_id=setup.get("client_id"),
                    opportunity_id=setup.get("opportunity_id"),
                    status="Draft",
                )

        # Create contract if needed (from setup_entities)
        if "contract_id" in setup:
            if hasattr(backend, "contracts") and setup["contract_id"] not in backend.contracts:
                backend.create_contract(
                    contract_id=setup["contract_id"],
                    client_id=setup.get("client_id"),
                    opportunity_id=setup.get("opportunity_id"),
                    start_date=setup.get("start_date"),
                    end_date=setup.get("end_date"),
                    value=setup.get("contract_value", 100000.0),
                    status=setup.get("contract_status", "Active"),
                    document_url=setup.get("document_url"),
                )

    def _execute_tool(
        self,
        backend: Union[MockCrmApi, PostgresCrmBackend],
        tool_call: ToolCall
    ) -> ValidationResult:
        """Execute tool call against backend and return result."""
        try:
            method = getattr(backend, tool_call.tool_name, None)
            if not method:
                return ValidationResult.fail(f"Tool '{tool_call.tool_name}' not found in backend")

            result = method(**tool_call.arguments)
            return ValidationResult.ok("Tool executed successfully", {"result": result})
        except Exception as e:
            return ValidationResult.fail(str(e), {"exception": repr(e)})

    def _validate_scenario_result(
        self,
        scenario: Scenario,
        tool_call: ToolCall,
        execution_result: ValidationResult,
        pre: CrmStateSnapshot,
        post: CrmStateSnapshot,
    ) -> ValidationResult:
        """Validate scenario result based on expected success/failure."""
        tool_correct = tool_call.tool_name == scenario.expected_tool

        if scenario.expect_success:
            if not tool_correct:
                return ValidationResult.fail(
                    f"Expected tool '{scenario.expected_tool}' but agent called '{tool_call.tool_name}'"
                )
            if not execution_result.success:
                return ValidationResult.fail(f"Tool execution failed: {execution_result.message}")

            # Check arguments match (basic validation)
            # For now, just check execution succeeded - detailed arg checking can be added later
            return ValidationResult.ok("Scenario passed - tool executed successfully")
        else:
            # Expect failure
            if not tool_correct:
                return ValidationResult.fail(
                    f"Expected tool '{scenario.expected_tool}' but agent called '{tool_call.tool_name}'"
                )
            if execution_result.success:
                return ValidationResult.fail("Expected failure but tool executed successfully")

            # Check expected error substring
            if scenario.expected_error_substring:
                if scenario.expected_error_substring not in execution_result.message:
                    return ValidationResult.fail(
                        f"Expected error substring '{scenario.expected_error_substring}' not found in: {execution_result.message}"
                    )

            # Check state unchanged
            if pre != post:
                return ValidationResult.fail("Expected state to be unchanged after failed operation")

            return ValidationResult.ok(f"Scenario passed - expected failure occurred: {execution_result.message}")

    def run(self, mode: str = "agent") -> Dict[str, Any]:
        """Run scenarios through agent and log results.

        Args:
            mode: "agent" to use configured agent, "mock" to use ground-truth

        Returns:
            Dict with success_count, failure_count, episodes
        """
        episodes: List[EpisodeLog] = []
        successes = 0
        failures = 0

        with self.log_path.open("w", encoding="utf-8") as log_file:
            for scenario in self.scenarios:
                # Initialize backend
                if self._backend_mode == "postgres":
                    if not self._db_backend:
                        raise RuntimeError("Postgres backend not initialized")
                    backend = self._db_backend
                    backend.begin_session(reset=self._reset_database_each_case)
                else:
                    backend = MockCrmApi()

                try:
                    # Create entities from scenario setup
                    self._create_entities_from_scenario(scenario, backend)

                    # Build prompt
                    prompt = build_scenario_prompt(scenario, backend)

                    # Get tool call from agent
                    if mode == "mock":
                        mock_agent = ScenarioMockAgent()
                        tool_call = mock_agent.tool_call(scenario, prompt)
                    else:
                        # Adapt agent interface
                        if hasattr(self.agent, "tool_call"):
                            # Check if agent expects GoldenCase or can handle generic call
                            if isinstance(self.agent, (ClaudeAgent, OpenAIAgent)):
                                # These agents parse prompt directly
                                tool_call = self.agent.tool_call(scenario, prompt)
                            else:
                                # Try generic call
                                tool_call = self.agent.tool_call(scenario, prompt)
                        else:
                            raise RuntimeError(f"Agent {type(self.agent)} does not support tool_call interface")

                    # Execute tool
                    pre = CrmStateSnapshot.from_backend(backend)
                    execution_result = self._execute_tool(backend, tool_call)
                    post = CrmStateSnapshot.from_backend(backend)

                    # Validate result
                    validation_result = self._validate_scenario_result(
                        scenario, tool_call, execution_result, pre, post
                    )

                    case_passed = validation_result.success
                    outcome_message = validation_result.message

                    # Teacher/Student verification (if enabled)
                    verifier_result: Optional[VerifierResult] = None
                    if self._verifier_enabled and self._verifier_name:
                        tool_trace = VerifierToolTrace(
                            step=1,
                            tool_name=tool_call.tool_name,
                            arguments=dict(tool_call.arguments),
                            execution_success=execution_result.success,
                            validator_success=validation_result.success,
                            message=validation_result.message,
                        )

                        verifier = get_registered_verifier(self._verifier_name)
                        request = VerifierRequest(
                            case_id=scenario.scenario_id,
                            task=scenario.task,
                            utterance=scenario.utterance,
                            expect_success=scenario.expect_success,
                            expected_error_substring=scenario.expected_error_substring,
                            expected_tool=scenario.expected_tool,
                            expected_arguments=scenario.expected_args,
                            tool_traces=(tool_trace,),
                            final_response=tool_call.raw_response,
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

                    if case_passed:
                        successes += 1
                    else:
                        failures += 1

                    # Build reward breakdown
                    base_reward = 1.0 if case_passed else 0.0
                    verifier_contrib = 0.0
                    if verifier_result:
                        verifier_contrib = self._verifier_reward_weight * verifier_result.score
                    final_reward = (1 - self._verifier_reward_weight) * base_reward + verifier_contrib

                    reward_breakdown = {
                        "base": base_reward,
                        "verifier": {
                            "enabled": self._verifier_enabled,
                            "weight": self._verifier_reward_weight,
                            "score": verifier_result.score if verifier_result else 0.0,
                            "contribution": verifier_contrib,
                        },
                        "final": final_reward,
                    }

                    # Build learning signals
                    learning_signals = {
                        "student": {
                            "summary": validation_result.message,
                            "score": base_reward,
                        },
                        "teacher": {
                            "summary": verifier_result.rationale if verifier_result else "",
                            "score": verifier_result.score if verifier_result else 0.0,
                        },
                        "adapter_events": [],
                        "reason": outcome_message,
                        "drift_notes": "",
                        "reward_observation": final_reward,
                    }

                    # Build episode log
                    episode = EpisodeLog(
                        case_id=scenario.scenario_id,
                        task=scenario.task,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        provider=self.agent.provider_name,
                        model=self.agent.model_name,
                        success=case_passed,
                        expected_success=scenario.expect_success,
                        message=outcome_message,
                        tool_call={"tool_name": tool_call.tool_name, "arguments": tool_call.arguments},
                        agent_response=tool_call.raw_response,
                        validator_details=validation_result.details,
                        expected_tool=scenario.expected_tool,
                        expected_arguments=scenario.expected_args,
                        verification_mode=scenario.verification_mode.value if hasattr(scenario.verification_mode, 'value') else scenario.verification_mode,
                        verifier_name=self._verifier_name if self._verifier_enabled else None,
                        verifier_score=verifier_result.score if verifier_result else None,
                        verifier_rationale=verifier_result.rationale if verifier_result else None,
                        verifier_metadata=verifier_result.metadata if verifier_result else None,
                        reward_breakdown=reward_breakdown,
                        learning_signals=learning_signals,
                        validator_metadata={
                            "tool_correct": tool_call.tool_name == scenario.expected_tool,
                            "execution_success": execution_result.success,
                        },
                    )

                    episodes.append(episode)
                    log_file.write(episode.to_json() + "\n")
                    log_file.flush()

                except Exception as exc:
                    print(f"Error processing scenario {scenario.scenario_id}: {exc}")
                    failures += 1
                    continue

        return {
            "success_count": successes,
            "failure_count": failures,
            "total": len(self.scenarios),
            "episodes": episodes,
        }
