"""Curator-based multi-turn conversation generator for CRM benchmark.

This module generates multi-turn conversations using Bespoke Curator with GPT-5-mini.
Conversations are generated iteratively turn-by-turn, with each turn building on
previous conversation history to create realistic multi-turn interactions.

Key features:
- Iterative generation: Turn 1 generates initial request, Turn 2+ includes conversation history
- Template resolution: Uses {{turn_N.field}} syntax for cross-turn entity references
- State simulation: Simulates CRM state changes between turns
- Validation: Ensures all templates resolve correctly before completion

GPT-5-mini is chosen for:
- Better structured output reliability (eliminates null pollution)
- More consistent schema compliance
- Lower cost ($0.25/$2.00 per M tokens)
- Mature structured output support via LiteLLM
"""

import json
import logging
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset
from pydantic import BaseModel, Field
from bespokelabs import curator
from dotenv import load_dotenv

load_dotenv()

from .crm_sandbox import MockCrmApi
from .entity_sampler import EntitySampler, SamplerConfig
from .conversation_schema import Conversation, ConversationTurn
from .conversation_templates import (
    WorkflowTemplate,
    TurnTemplate,
    get_templates_by_complexity,
    get_all_templates,
    WORKFLOW_TEMPLATES,
)
from .reference_resolver import resolve_template, extract_template_references, validate_template_references
from .validators import VerificationMode
from .failure_blueprints import FailureCategory

logger = logging.getLogger(__name__)


class ConversationTurnResponse(BaseModel):
    """Structured output format for a single turn generation from LLM.

    Note: The LLM only generates user_utterance. The expected_args and
    expected_tool come from the turn template for deterministic ground truth.
    """

    user_utterance: str = Field(
        description="Natural language user utterance for this turn. Use pronouns like 'them', 'it', 'that' to reference previous entities when appropriate."
    )


class CuratorConversationGenerator(curator.LLM):
    """Curator LLM class for generating individual conversation turns using GPT-5-mini.
    
    This generator is called iteratively for each turn in a conversation, with
    conversation history passed in the input to enable natural multi-turn flows.
    """

    response_format = ConversationTurnResponse  # Structured outputs via GPT-5-mini
    batch = True  # Enable batch mode for 50% cost savings

    def __init__(
        self,
        entity_pool: Dict[str, List[Dict[str, Any]]],
        model_name: str = "gpt-5-mini",
    ):
        super().__init__(
            model_name=model_name,
            backend="litellm",
            backend_params={
                "max_requests_per_minute": 2000,
                "max_tokens_per_minute": 4000000,
            },
            generation_params={},  # GPT-5-mini only supports default temperature (1)
        )
        self.entity_pool = entity_pool

    def prompt(self, input: Dict) -> str:
        """Convert input row to LLM prompt for turn generation."""
        turn_number = input["turn_number"]
        workflow_category = input["workflow_category"]
        turn_template = input["turn_template"]
        conversation_history = input.get("conversation_history", [])
        current_crm_state = input.get("current_crm_state", {})
        conversation_id = input.get("conversation_id", "")  # For logging/debugging

        if turn_number == 1:
            return self._build_first_turn_prompt(
                workflow_category, turn_template, current_crm_state
            )
        else:
            return self._build_subsequent_turn_prompt(
                turn_number,
                workflow_category,
                turn_template,
                conversation_history,
                current_crm_state,
            )

    def _build_first_turn_prompt(
        self,
        workflow_category: str,
        turn_template: Dict[str, Any],
        current_crm_state: Dict[str, Any],
    ) -> str:
        """Build prompt for Turn 1 (initial request)."""
        tool_name = turn_template["tool_name"]
        argument_template = turn_template["argument_template"]
        user_utterance_pattern = turn_template["user_utterance_pattern"]

        # Format entity pool for prompt
        entity_pool_text = self._format_entity_pool()

        prompt = f"""Generate the first turn of a multi-turn CRM conversation.

Workflow Category: {workflow_category}
Task: {tool_name}
Pattern: {user_utterance_pattern}

Available Entities:
{entity_pool_text}

Generate ONLY a natural language user utterance that:
- Matches the pattern "{user_utterance_pattern}"
- Uses realistic company names from the entity pool
- Sounds like real user input (casual, abbreviated like "$330k", "opp" instead of "opportunity")
- Is specific and realistic (e.g., "Show me Acme Corp" not "Show me a client")

Examples:
- "Show me Acme Corp"
- "Create a $250k opp for cloud migration"
- "Generate a quote for Enterprise Integration Services"

Output JSON:
{{
  "user_utterance": "your generated utterance here"
}}
"""
        return prompt

    def _build_subsequent_turn_prompt(
        self,
        turn_number: int,
        workflow_category: str,
        turn_template: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
        current_crm_state: Dict[str, Any],
    ) -> str:
        """Build prompt for Turn 2+ (with conversation history)."""
        tool_name = turn_template["tool_name"]
        argument_template = turn_template["argument_template"]
        user_utterance_pattern = turn_template["user_utterance_pattern"]
        references_previous_turns = turn_template.get("references_previous_turns", [])

        # Format conversation history
        history_text = self._format_conversation_history(conversation_history)

        # Format entity pool
        entity_pool_text = self._format_entity_pool()

        prompt = f"""Generate turn {turn_number} of a multi-turn CRM conversation.

Workflow Category: {workflow_category}
Task: {tool_name}
Pattern: {user_utterance_pattern}

Conversation History:
{history_text}

Current CRM State:
{self._format_crm_state(current_crm_state)}

Generate ONLY a natural language user utterance that:
- Naturally continues the conversation using pronouns ("them", "that", "it", "the opp")
- References entities from previous turns implicitly (NOT explicit IDs)
- Matches the pattern "{user_utterance_pattern}"
- Sounds like real user input (casual, abbreviated like "$250k", "opp" instead of "opportunity")
- Follows the conversation flow naturally

Examples:
- "Create a $250k opp for them"
- "Update the status to Active"
- "Move that opportunity to Negotiation"
- "Send the quote"
- "Add a contact for them"

DO NOT:
- Use explicit IDs (e.g., "Create opp for client abc-123" - use "them" instead)
- Be generic (e.g., "Do the thing" - be specific)
- Use entity names that don't exist in the conversation history

Output JSON:
{{
  "user_utterance": "your generated utterance here"
}}
"""
        return prompt

    def _format_entity_pool(self) -> str:
        """Format entity pool for inclusion in prompts."""
        lines = []
        lines.append("Clients:")
        for client in self.entity_pool.get("clients", [])[:50]:
            lines.append(
                f"  - {client.get('name', 'Unknown')} (ID: {client.get('client_id')}, Industry: {client.get('industry', 'N/A')}, Status: {client.get('status', 'N/A')})"
            )

        lines.append("\nOpportunities:")
        for opp in self.entity_pool.get("opportunities", [])[:50]:
            lines.append(
                f"  - {opp.get('name', 'Unknown')} (ID: {opp.get('opportunity_id')}, Client: {opp.get('client_id')}, Amount: ${opp.get('amount', 0):,.0f}, Stage: {opp.get('stage', 'N/A')})"
            )

        lines.append("\nQuotes:")
        for quote in self.entity_pool.get("quotes", [])[:30]:
            lines.append(
                f"  - Quote {quote.get('quote_id')} (Opportunity: {quote.get('opportunity_id')}, Amount: ${quote.get('amount', 0):,.0f}, Status: {quote.get('status', 'N/A')})"
            )

        lines.append("\nContacts:")
        for contact in self.entity_pool.get("contacts", [])[:30]:
            lines.append(
                f"  - {contact.get('first_name', '')} {contact.get('last_name', '')} (ID: {contact.get('contact_id')}, Client: {contact.get('client_id')})"
            )

        return "\n".join(lines)

    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for inclusion in prompts."""
        if not history:
            return "No previous turns."

        lines = []
        for turn_data in history:
            turn_id = turn_data.get("turn_id", "?")
            user_utterance = turn_data.get("user_utterance", "")
            tool_name = turn_data.get("tool_name", "")
            result = turn_data.get("result", {})
            
            lines.append(f"Turn {turn_id}:")
            lines.append(f"  User: {user_utterance}")
            lines.append(f"  Tool: {tool_name}")
            if result:
                # Show key fields from result (entity IDs created)
                result_summary = []
                for key in ["client_id", "contact_id", "opportunity_id", "quote_id", "contract_id"]:
                    if key in result:
                        result_summary.append(f"{key}={result[key]}")
                if result_summary:
                    lines.append(f"  Result: {', '.join(result_summary)}")
            lines.append("")

        return "\n".join(lines)

    def _format_crm_state(self, state: Dict[str, Any]) -> str:
        """Format current CRM state for inclusion in prompts."""
        if not state:
            return "No entities in CRM state yet."

        lines = []
        for entity_type, entities in state.items():
            lines.append(f"{entity_type.title()}:")
            if isinstance(entities, list):
                for entity in entities[:5]:  # Limit to 5 per type
                    lines.append(f"  - {json.dumps(entity, default=str)}")
            else:
                lines.append(f"  - {json.dumps(entities, default=str)}")

        return "\n".join(lines) if lines else "No entities in CRM state."

    def parse(self, input: Dict, response: Any) -> Dict:
        """Convert LLM response to turn dict.

        Note: expected_args comes from the turn template, not the LLM response.
        The LLM only generates the natural language user_utterance.
        """
        if isinstance(response, ConversationTurnResponse):
            return {
                "conversation_id": input.get("conversation_id", ""),
                "turn_number": input["turn_number"],
                "user_utterance": response.user_utterance,
                "expected_args": input["turn_template"]["argument_template"],
                "tool_name": input["turn_template"]["tool_name"],
            }
        elif isinstance(response, str):
            # Fallback parsing
            try:
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                if json_match:
                    response_json = json.loads(json_match.group(1))
                else:
                    response_json = json.loads(response.strip())
                
                return {
                    "conversation_id": input.get("conversation_id", ""),
                    "turn_number": input["turn_number"],
                    "user_utterance": response_json.get("user_utterance", ""),
                    "expected_args": response_json.get("expected_args", {}),
                    "tool_name": input["turn_template"]["tool_name"],
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from response: {response[:200]}... Error: {e}")
                return {
                    "conversation_id": input.get("conversation_id", ""),
                    "turn_number": input["turn_number"],
                    "user_utterance": "",
                    "expected_args": {},
                    "tool_name": input["turn_template"]["tool_name"],
                }
        else:
            # Try to convert dict-like response
            response_dict = dict(response) if hasattr(response, '__dict__') else response
            return {
                "conversation_id": input.get("conversation_id", ""),
                "turn_number": input["turn_number"],
                "user_utterance": response_dict.get("user_utterance", ""),
                "expected_args": response_dict.get("expected_args", {}),
                "tool_name": input["turn_template"]["tool_name"],
            }


def populate_argument_placeholders(
    argument_template: Dict[str, Any],
    current_crm_state: Dict[str, Any],
    user_utterance: Optional[str] = None,
) -> Dict[str, Any]:
    """Fill placeholder values in argument_template with actual entity data.

    Placeholders like empty strings ("") or None in the template are filled
    with corresponding values from current_crm_state.

    Args:
        argument_template: Template with placeholders (e.g., {"name": "", "status": null})
        current_crm_state: Current CRM entities available for this turn
        user_utterance: Optional user utterance to extract amounts from (for amount=0.0 placeholders)

    Returns:
        Populated arguments with actual values
    """
    import re
    populated_args = {}

    for key, value in argument_template.items():
        # Handle amount placeholder specially - extract from user utterance if 0.0
        if key == "amount" and value == 0.0:
            if user_utterance:
                # Try to extract amount from user utterance (e.g., "$250k", "$330,000")
                amount_match = re.search(r'\$(\d+(?:\.\d+)?)[kK]', user_utterance)
                if amount_match:
                    populated_args[key] = float(amount_match.group(1)) * 1000
                    continue
                else:
                    # Try comma-separated format
                    amount_match = re.search(r'\$(\d{1,3}(?:,\d{3})+)', user_utterance.replace(',', ''))
                    if amount_match:
                        populated_args[key] = float(amount_match.group(1).replace(',', ''))
                        continue
            
            # If we can't extract from utterance, use a default based on context
            # Default to 100k for opportunities/quotes, but this should ideally be fixed at generation
            logger.warning(f"Amount placeholder 0.0 not resolved from utterance, using default 100000.0")
            populated_args[key] = 100000.0
            continue
        
        # Keep non-placeholder values as-is
        if value not in ("", None) and value != 0.0:
            populated_args[key] = value
            continue

        # Fill placeholders from current_crm_state
        if key == "client_id" and "clients" in current_crm_state and current_crm_state["clients"]:
            populated_args[key] = current_crm_state["clients"][0]["client_id"]
        elif key == "name" and "clients" in current_crm_state and current_crm_state["clients"]:
            populated_args[key] = current_crm_state["clients"][0]["name"]
        elif key == "email" and "clients" in current_crm_state and current_crm_state["clients"]:
            populated_args[key] = current_crm_state["clients"][0].get("email")
        elif key == "opportunity_id" and "opportunities" in current_crm_state and current_crm_state["opportunities"]:
            populated_args[key] = current_crm_state["opportunities"][0]["opportunity_id"]
        elif key == "quote_id" and "quotes" in current_crm_state and current_crm_state["quotes"]:
            populated_args[key] = current_crm_state["quotes"][0]["quote_id"]
        elif key == "contact_id" and "contacts" in current_crm_state and current_crm_state["contacts"]:
            populated_args[key] = current_crm_state["contacts"][0]["contact_id"]
        elif key == "contract_id" and "contracts" in current_crm_state and current_crm_state["contracts"]:
            populated_args[key] = current_crm_state["contracts"][0]["contract_id"]
        else:
            # Keep placeholder if we can't fill it
            populated_args[key] = value

    return populated_args


def simulate_turn_execution(
    tool_name: str,
    expected_args: Dict[str, Any],
    api: MockCrmApi,
    entity_pool: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Simulate tool execution to update CRM state for next turn.

    This doesn't actually execute the tool, but simulates what would happen
    to extract entity IDs from search results or created entities.

    Args:
        tool_name: Name of the tool that was called
        expected_args: Arguments passed to the tool
        api: MockCrmApi instance (for validation, not actual execution)
        entity_pool: Entity pool to select from for searches

    Returns:
        Dictionary with entity IDs and other fields extracted from simulated execution
    """
    result = {}
    
    # Handle create operations - extract ID from entity pool or generate
    if tool_name == "create_new_client":
        # For generation, we'll use the client_id from the entity pool if available
        # Otherwise, generate a new UUID
        result["client_id"] = expected_args.get("client_id") or str(uuid.uuid4())
        
    elif tool_name == "create_new_contact":
        result["contact_id"] = expected_args.get("contact_id") or str(uuid.uuid4())
        result["client_id"] = expected_args.get("client_id")
        
    elif tool_name == "create_new_opportunity":
        result["opportunity_id"] = expected_args.get("opportunity_id") or str(uuid.uuid4())
        result["client_id"] = expected_args.get("client_id")
        
    elif tool_name == "create_quote":
        result["quote_id"] = expected_args.get("quote_id") or str(uuid.uuid4())
        result["opportunity_id"] = expected_args.get("opportunity_id")
        
    elif tool_name == "create_contract":
        result["contract_id"] = expected_args.get("contract_id") or str(uuid.uuid4())
        result["client_id"] = expected_args.get("client_id")
        result["opportunity_id"] = expected_args.get("opportunity_id")
        
    # Handle search operations - extract first result's ID
    elif tool_name == "client_search":
        # Simulate search by selecting matching entity from pool
        # Prioritize exact matches, then partial matches, then fallback
        name_filter = expected_args.get("name", "")
        if name_filter and entity_pool.get("clients"):
            # First try exact match (case-insensitive)
            for client in entity_pool["clients"]:
                if client.get("name", "").lower() == name_filter.lower():
                    result["client_id"] = client.get("client_id")
                    result["name"] = client.get("name")  # Include for context
                    break
            
            # If no exact match, try partial match
            if "client_id" not in result:
                for client in entity_pool["clients"]:
                    if name_filter.lower() in client.get("name", "").lower():
                        result["client_id"] = client.get("client_id")
                        result["name"] = client.get("name")
                        break
        
        # Fallback to first client if no match
        if "client_id" not in result and entity_pool.get("clients"):
            result["client_id"] = entity_pool["clients"][0].get("client_id")
            result["name"] = entity_pool["clients"][0].get("name")
            
    elif tool_name == "opportunity_search":
        client_id_filter = expected_args.get("client_id")
        name_filter = expected_args.get("name", "")
        if entity_pool.get("opportunities"):
            for opp in entity_pool["opportunities"]:
                if client_id_filter and opp.get("client_id") == client_id_filter:
                    result["opportunity_id"] = opp.get("opportunity_id")
                    break
                elif name_filter and name_filter.lower() in opp.get("name", "").lower():
                    result["opportunity_id"] = opp.get("opportunity_id")
                    break
        if "opportunity_id" not in result and entity_pool.get("opportunities"):
            result["opportunity_id"] = entity_pool["opportunities"][0].get("opportunity_id")
            
    elif tool_name == "quote_search":
        opportunity_id_filter = expected_args.get("opportunity_id")
        if opportunity_id_filter and entity_pool.get("quotes"):
            for quote in entity_pool["quotes"]:
                if quote.get("opportunity_id") == opportunity_id_filter:
                    result["quote_id"] = quote.get("quote_id")
                    break
        if "quote_id" not in result and entity_pool.get("quotes"):
            result["quote_id"] = entity_pool["quotes"][0].get("quote_id")
            
    elif tool_name == "contact_search":
        client_id_filter = expected_args.get("client_id")
        if client_id_filter and entity_pool.get("contacts"):
            for contact in entity_pool["contacts"]:
                if contact.get("client_id") == client_id_filter:
                    result["contact_id"] = contact.get("contact_id")
                    break
        if "contact_id" not in result and entity_pool.get("contacts"):
            result["contact_id"] = entity_pool["contacts"][0].get("contact_id")
            
    elif tool_name in ["view_opportunity_details", "opportunity_details"]:
        result["opportunity_id"] = expected_args.get("opportunity_id")
        
    elif tool_name == "quote_details":
        result["quote_id"] = expected_args.get("quote_id")
        
    # Handle modify operations - pass through IDs
    elif tool_name == "modify_client":
        result["client_id"] = expected_args.get("client_id")
        
    elif tool_name == "modify_contact":
        result["contact_id"] = expected_args.get("contact_id")
        
    elif tool_name == "modify_opportunity":
        result["opportunity_id"] = expected_args.get("opportunity_id")
        # Also pass through any updated fields
        if "updates" in expected_args:
            updates = expected_args["updates"]
            if "stage" in updates:
                result["stage"] = updates["stage"]
                
    elif tool_name == "modify_quote":
        result["quote_id"] = expected_args.get("quote_id")
        
    # Handle other operations
    elif tool_name == "clone_opportunity":
        result["opportunity_id"] = expected_args.get("opportunity_id") or str(uuid.uuid4())
        
    elif tool_name == "add_note":
        result["entity_id"] = expected_args.get("entity_id")
        result["entity_type"] = expected_args.get("entity_type")
        
    elif tool_name == "upload_document":
        result["entity_id"] = expected_args.get("entity_id")
        result["entity_type"] = expected_args.get("entity_type")
        
    # Pass through any IDs from expected_args
    for key in ["client_id", "contact_id", "opportunity_id", "quote_id", "contract_id"]:
        if key in expected_args and key not in result:
            result[key] = expected_args[key]
    
    return result


class CuratorConversationDatasetGenerator:
    """Main generator class that orchestrates multi-turn conversation generation."""
    
    def __init__(
        self,
        api: Optional[MockCrmApi] = None,
        sampler: Optional[EntitySampler] = None,
        seed: Optional[int] = 42,
    ):
        self.api = api or MockCrmApi()
        self.sampler = sampler or EntitySampler(self.api, SamplerConfig(seed=seed))
        self.entity_pool: Dict[str, List[Dict[str, Any]]] = {}
        
    def _build_entity_pool(self, client_count: int = 400, opportunity_count: int = 400) -> None:
        """Pre-generate realistic entity pool using EntitySampler."""
        logger.info(f"Building entity pool: {client_count} clients, {opportunity_count} opportunities...")
        
        clients = []
        for _ in range(client_count):
            client = self.sampler.sample_client()
            clients.append({
                "client_id": client.client_id,
                "name": client.name,
                "industry": client.industry,
                "status": client.status.value if hasattr(client.status, 'value') else str(client.status),
                "email": client.email,
            })
        
        opportunities = []
        quotes = []
        contacts = []
        
        for client in clients[:opportunity_count]:
            # Create opportunity for this client
            opp = self.sampler.sample_opportunity(client["client_id"])
            opportunities.append({
                "opportunity_id": opp.opportunity_id,
                "name": opp.name,
                "client_id": opp.client_id,
                "amount": round(opp.amount or 0.0),
                "stage": opp.stage.value if hasattr(opp.stage, 'value') else str(opp.stage),
                "probability": opp.probability,
            })
            
            # Create quote for some opportunities
            if len(quotes) < 150:
                quote = self.sampler.sample_quote(opp.opportunity_id)
                quotes.append({
                    "quote_id": quote.quote_id,
                    "opportunity_id": quote.opportunity_id,
                    "amount": round(quote.amount or 0.0),
                    "status": quote.status.value if hasattr(quote.status, 'value') else str(quote.status),
                })
            
            # Create contact for some clients
            if len(contacts) < 100:
                contact = self.sampler.sample_contact(client["client_id"])
                contacts.append({
                    "contact_id": contact.contact_id,
                    "first_name": contact.first_name,
                    "last_name": contact.last_name,
                    "client_id": contact.client_id,
                    "email": contact.email,
                })
        
        self.entity_pool = {
            "clients": clients,
            "opportunities": opportunities,
            "quotes": quotes,
            "contacts": contacts,
        }
        
        logger.info(
            f"Entity pool built: {len(clients)} clients, {len(opportunities)} opportunities, "
            f"{len(quotes)} quotes, {len(contacts)} contacts"
        )
    
    def generate_conversation(
        self,
        workflow_template: WorkflowTemplate,
        conversation_id: str,
        generator: CuratorConversationGenerator,
    ) -> Optional[Conversation]:
        """Generate a single conversation using iterative turn-by-turn generation.
        
        Args:
            workflow_template: Template defining the workflow pattern
            conversation_id: Unique identifier for this conversation
            generator: CuratorConversationGenerator instance
            
        Returns:
            Conversation object if generation succeeds, None otherwise
        """
        # Validate template dependencies before generation
        for turn_template in workflow_template.turn_templates:
            for ref_turn in turn_template.references_previous_turns:
                if ref_turn >= turn_template.turn_number:
                    logger.error(
                        f"{conversation_id}: Turn {turn_template.turn_number} references "
                        f"turn {ref_turn} (forward reference detected)"
                    )
                    return None
        
        turns = []
        conversation_history = []
        current_crm_state = {}
        
        # Build initial CRM state from required entities
        # For first turn searches, we need to generate the turn first to know what to search for
        # Then create matching initial entities. For now, we'll create entities and ensure
        # the first turn search criteria matches them.
        initial_entities = {}
        for entity_type in workflow_template.required_initial_entities:
            if entity_type == "client" and self.entity_pool.get("clients"):
                client = random.choice(self.entity_pool["clients"])
                initial_entities["client_id"] = client["client_id"]
                initial_entities["client_name"] = client["name"]
                initial_entities["client_email"] = client.get("email")
                initial_entities["client_status"] = client.get("status")
                current_crm_state["clients"] = [client]
            elif entity_type == "opportunity" and self.entity_pool.get("opportunities"):
                opp = random.choice(self.entity_pool["opportunities"])
                initial_entities["opportunity_id"] = opp["opportunity_id"]
                initial_entities["opportunity_name"] = opp.get("name")
                current_crm_state["opportunities"] = [opp]
        
        # Generate each turn iteratively
        for turn_template in workflow_template.turn_templates:
            turn_number = turn_template.turn_number
            
            # Prepare input for Curator
            input_data = {
                "turn_number": turn_number,
                "workflow_category": workflow_template.workflow_category,
                "turn_template": {
                    "tool_name": turn_template.tool_name,
                    "argument_template": turn_template.argument_template,
                    "user_utterance_pattern": turn_template.user_utterance_pattern,
                    "references_previous_turns": turn_template.references_previous_turns,
                },
                "conversation_history": conversation_history,
                "current_crm_state": current_crm_state,
            }
            
            # Generate turn via Curator (single-item batch)
            try:
                dataset = Dataset.from_list([input_data])
                result = generator(dataset)
                
                # Extract turn result
                turn_result = None
                for row in result.dataset:
                    turn_result = dict(row)
                    break
                
                if not turn_result or not turn_result.get("user_utterance"):
                    logger.warning(f"Failed to generate turn {turn_number} for {conversation_id}")
                    return None
                
                # Populate argument placeholders with actual entity data
                populated_args = populate_argument_placeholders(
                    turn_result["expected_args"],
                    current_crm_state,
                    user_utterance=turn_result.get("user_utterance")
                )
                
                # For first turn searches, ensure search criteria matches initial entities
                if turn_number == 1 and turn_result["tool_name"].endswith("_search"):
                    if turn_result["tool_name"] == "client_search" and initial_entities.get("client_name"):
                        # Ensure search criteria matches the initial client
                        search_name = populated_args.get("name")
                        if not search_name or search_name.lower() not in initial_entities["client_name"].lower():
                            # Update search to match initial entity
                            populated_args["name"] = initial_entities["client_name"]
                            logger.debug(f"{conversation_id}: Updated turn 1 client_search to match initial entity: {initial_entities['client_name']}")
                
                # Create ConversationTurn
                turn = ConversationTurn(
                    turn_id=turn_number,
                    user_utterance=turn_result["user_utterance"],
                    expected_tool=turn_result["tool_name"],
                    expected_args=populated_args,
                    references_previous_turns=turn_template.references_previous_turns,
                )
                turns.append(turn)
                
                # Simulate execution to update state
                execution_result = simulate_turn_execution(
                    turn_result["tool_name"],
                    populated_args,
                    self.api,
                    self.entity_pool,
                )
                
                # Update conversation history
                conversation_history.append({
                    "turn_id": turn_number,
                    "user_utterance": turn_result["user_utterance"],
                    "tool_name": turn_result["tool_name"],
                    "result": execution_result,
                })
                
                # Update CRM state for next turn
                # Add created entities to state
                for key, value in execution_result.items():
                    if key.endswith("_id") and value:
                        if key == "client_id":
                            # Find client in pool
                            for client in self.entity_pool.get("clients", []):
                                if client["client_id"] == value:
                                    if "clients" not in current_crm_state:
                                        current_crm_state["clients"] = []
                                    if client not in current_crm_state["clients"]:
                                        current_crm_state["clients"].append(client)
                                    break
                        elif key == "opportunity_id":
                            for opp in self.entity_pool.get("opportunities", []):
                                if opp["opportunity_id"] == value:
                                    if "opportunities" not in current_crm_state:
                                        current_crm_state["opportunities"] = []
                                    if opp not in current_crm_state["opportunities"]:
                                        current_crm_state["opportunities"].append(opp)
                                    break
                        elif key == "quote_id":
                            for quote in self.entity_pool.get("quotes", []):
                                if quote["quote_id"] == value:
                                    if "quotes" not in current_crm_state:
                                        current_crm_state["quotes"] = []
                                    if quote not in current_crm_state["quotes"]:
                                        current_crm_state["quotes"].append(quote)
                                    break
                
            except Exception as e:
                logger.error(f"Error generating turn {turn_number} for {conversation_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
        
        # Validate template resolution
        previous_turns_dict = {}
        for hist_entry in conversation_history:
            turn_id = hist_entry["turn_id"]
            previous_turns_dict[turn_id] = hist_entry.get("result", {})
        
        # Validate all templates resolve
        for turn in turns:
            if turn.expected_args:
                errors = validate_template_references(
                    turn.expected_args,
                    previous_turns_dict,
                    turn.turn_id,
                )
                if errors:
                    logger.warning(
                        f"Template validation errors for {conversation_id} turn {turn.turn_id}: {errors}"
                    )
                    # Try to resolve anyway - might be recoverable
                    try:
                        resolved = resolve_template(
                            turn.expected_args,
                            previous_turns_dict,
                            turn.turn_id,
                            strict=False,
                        )
                        # If resolution succeeded, update expected_args
                        turn.expected_args = resolved
                    except Exception as e:
                        logger.error(f"Failed to resolve templates for turn {turn.turn_id}: {e}")
                        return None
        
        # Create Conversation object
        conversation = Conversation(
            conversation_id=conversation_id,
            workflow_category=workflow_template.workflow_category,
            complexity_level=workflow_template.complexity_level,
            turns=turns,
            initial_entities=initial_entities,
            verification_mode=VerificationMode.DATABASE,
        )
        
        return conversation
    
    def generate_conversations(
        self,
        target_count: int = 1500,
        simple_count: int = 900,
        medium_count: int = 450,
        complex_count: int = 150,
        batch_size: int = 10,
    ) -> List[Conversation]:
        """Generate conversations using Curator.
        
        Args:
            target_count: Total number of conversations to generate
            simple_count: Number of simple (1-3 turn) conversations
            medium_count: Number of medium (4-6 turn) conversations
            complex_count: Number of complex (7-10 turn) conversations
            batch_size: Number of conversations to generate per batch
            
        Returns:
            List of Conversation objects
        """
        # Build entity pool if not already built
        if not self.entity_pool:
            self._build_entity_pool()
        
        # Get templates by complexity
        simple_templates = get_templates_by_complexity("simple")
        medium_templates = get_templates_by_complexity("medium")
        complex_templates = get_templates_by_complexity("complex")
        
        if not simple_templates or not medium_templates or not complex_templates:
            raise ValueError("Missing workflow templates for one or more complexity levels")
        
        # Initialize generator
        generator = CuratorConversationGenerator(entity_pool=self.entity_pool)
        
        # Prepare conversation generation tasks
        tasks = []
        conversation_counter = 0
        
        # Simple conversations
        for _ in range(simple_count):
            template = random.choice(simple_templates)
            conversation_counter += 1
            tasks.append({
                "conversation_id": f"CONV-{conversation_counter:05d}",
                "template": template,
            })
        
        # Medium conversations
        for _ in range(medium_count):
            template = random.choice(medium_templates)
            conversation_counter += 1
            tasks.append({
                "conversation_id": f"CONV-{conversation_counter:05d}",
                "template": template,
            })
        
        # Complex conversations
        for _ in range(complex_count):
            template = random.choice(complex_templates)
            conversation_counter += 1
            tasks.append({
                "conversation_id": f"CONV-{conversation_counter:05d}",
                "template": template,
            })
        
        # Shuffle tasks
        random.shuffle(tasks)
        
        logger.info(f"Generating {len(tasks)} conversations using Curator...")
        
        # Initialize conversation states
        conversation_states = {}
        for task in tasks:
            conversation_states[task["conversation_id"]] = {
                "template": task["template"],
                "turns": [],
                "history": [],
                "crm_state": {},
                "initial_entities": {},
            }
        
        # Build initial entities for all conversations
        for conv_id, state in conversation_states.items():
            template = state["template"]
            for entity_type in template.required_initial_entities:
                if entity_type == "client" and self.entity_pool.get("clients"):
                    client = random.choice(self.entity_pool["clients"])
                    state["initial_entities"]["client_id"] = client["client_id"]
                    state["crm_state"]["clients"] = [client]
                elif entity_type == "opportunity" and self.entity_pool.get("opportunities"):
                    opp = random.choice(self.entity_pool["opportunities"])
                    state["initial_entities"]["opportunity_id"] = opp["opportunity_id"]
                    state["crm_state"]["opportunities"] = [opp]
        
        # Find maximum number of turns across all templates
        max_turns = max(len(t["template"].turn_templates) for t in tasks)
        
        # Generate turn-by-turn, batching across conversations
        for turn_number in range(1, max_turns + 1):
            # Prepare batch for this turn across all conversations that need it
            batch_inputs = []
            batch_conversation_ids = []
            
            for conv_id, state in conversation_states.items():
                template = state["template"]
                if turn_number <= len(template.turn_templates):
                    turn_template = template.turn_templates[turn_number - 1]
                    
                    batch_inputs.append({
                        "conversation_id": conv_id,
                        "turn_number": turn_number,
                        "workflow_category": template.workflow_category,
                        "turn_template": {
                            "tool_name": turn_template.tool_name,
                            "argument_template": turn_template.argument_template,
                            "user_utterance_pattern": turn_template.user_utterance_pattern,
                            "references_previous_turns": turn_template.references_previous_turns,
                        },
                        "conversation_history": state["history"],
                        "current_crm_state": state["crm_state"],
                    })
                    batch_conversation_ids.append(conv_id)
            
            if not batch_inputs:
                continue  # No conversations need this turn
            
            logger.info(f"Generating Turn {turn_number} for {len(batch_inputs)} conversations...")
            
            # Generate via Curator in batch
            try:
                dataset = Dataset.from_list(batch_inputs)
                result = generator(dataset)
                
                # Process results
                for row in result.dataset:
                    conv_id = row["conversation_id"]
                    state = conversation_states[conv_id]
                    template = state["template"]
                    turn_template = template.turn_templates[turn_number - 1]

                    if not row.get("user_utterance"):
                        logger.warning(f"Failed to generate turn {turn_number} for {conv_id}")
                        continue

                    # Populate argument placeholders with actual entity data
                    populated_args = populate_argument_placeholders(
                        row["expected_args"],
                        state["crm_state"],
                        user_utterance=row.get("user_utterance")
                    )

                    # Create ConversationTurn
                    turn = ConversationTurn(
                        turn_id=turn_number,
                        user_utterance=row["user_utterance"],
                        expected_tool=row["tool_name"],
                        expected_args=populated_args,
                        references_previous_turns=turn_template.references_previous_turns,
                    )
                    state["turns"].append(turn)

                    # Simulate execution
                    execution_result = simulate_turn_execution(
                        row["tool_name"],
                        populated_args,
                        self.api,
                        self.entity_pool,
                    )
                    
                    # Update history
                    state["history"].append({
                        "turn_id": turn_number,
                        "user_utterance": row["user_utterance"],
                        "tool_name": row["tool_name"],
                        "result": execution_result,
                    })
                    
                    # Update CRM state
                    for key, value in execution_result.items():
                        if key.endswith("_id") and value:
                            if key == "client_id":
                                for client in self.entity_pool.get("clients", []):
                                    if client["client_id"] == value:
                                        if "clients" not in state["crm_state"]:
                                            state["crm_state"]["clients"] = []
                                        if client not in state["crm_state"]["clients"]:
                                            state["crm_state"]["clients"].append(client)
                                        break
                            elif key == "opportunity_id":
                                for opp in self.entity_pool.get("opportunities", []):
                                    if opp["opportunity_id"] == value:
                                        if "opportunities" not in state["crm_state"]:
                                            state["crm_state"]["opportunities"] = []
                                        if opp not in state["crm_state"]["opportunities"]:
                                            state["crm_state"]["opportunities"].append(opp)
                                        break
                            elif key == "quote_id":
                                for quote in self.entity_pool.get("quotes", []):
                                    if quote["quote_id"] == value:
                                        if "quotes" not in state["crm_state"]:
                                            state["crm_state"]["quotes"] = []
                                        if quote not in state["crm_state"]["quotes"]:
                                            state["crm_state"]["quotes"].append(quote)
                                        break
                
            except Exception as e:
                logger.error(f"Error generating Turn {turn_number}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        # Build Conversation objects and validate templates
        all_conversations = []
        for conv_id, state in conversation_states.items():
            if not state["turns"]:
                logger.warning(f"No turns generated for {conv_id}")
                continue
            
            # Validate template resolution
            previous_turns_dict = {}
            for hist_entry in state["history"]:
                turn_id = hist_entry["turn_id"]
                previous_turns_dict[turn_id] = hist_entry.get("result", {})
            
            # Validate all templates resolve
            all_valid = True
            for turn in state["turns"]:
                if turn.expected_args:
                    errors = validate_template_references(
                        turn.expected_args,
                        previous_turns_dict,
                        turn.turn_id,
                    )
                    if errors:
                        logger.warning(
                            f"Template validation errors for {conv_id} turn {turn.turn_id}: {errors}"
                        )
                        try:
                            resolved = resolve_template(
                                turn.expected_args,
                                previous_turns_dict,
                                turn.turn_id,
                                strict=False,
                            )
                            turn.expected_args = resolved
                        except Exception as e:
                            logger.error(f"Failed to resolve templates for {conv_id} turn {turn.turn_id}: {e}")
                            all_valid = False
                            break
            
            if not all_valid:
                continue
            
            # Create Conversation object
            template = state["template"]
            conversation = Conversation(
                conversation_id=conv_id,
                workflow_category=template.workflow_category,
                complexity_level=template.complexity_level,
                turns=state["turns"],
                initial_entities=state["initial_entities"],
                verification_mode=VerificationMode.DATABASE,
            )
            all_conversations.append(conversation)
        
        logger.info(f"Generated {len(all_conversations)} conversations successfully")
        return all_conversations

