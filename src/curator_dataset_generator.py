"""Curator-based dataset generator for production-faithful CRM scenarios.

This module uses Bespoke Curator with GPT-5-mini in batch mode to generate
realistic CRM scenarios with natural language utterances, realistic company names,
rounded amounts, and contextual failures.

GPT-5-mini is chosen over Gemini 2.5 Flash for:
- Better structured output reliability (eliminates null pollution)
- More consistent schema compliance
- Slightly lower cost ($0.25/$2.00 vs $0.30/$2.50 per M tokens)
- Mature structured output support via LiteLLM

Trade-off: Slower than Gemini (~7-10x slower), but worth it for data quality.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict

from datasets import Dataset
from pydantic import BaseModel, Field
from bespokelabs import curator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .crm_sandbox import MockCrmApi
from .entity_sampler import EntitySampler, SamplerConfig
from .scenario_generator import Scenario
from .failure_blueprints import (
    get_all_blueprints,
    FailureBlueprint,
    parse_taxonomy_csv,
    TaxonomyEntry,
)
from .intent_blueprints import get_all_intent_blueprints, IntentBlueprint
from .validators import VerificationMode
from .data_pools import COMPANY_NAMES

logger = logging.getLogger(__name__)


class ScenarioResponse(BaseModel):
    """Structured output format for scenario generation from LLM."""

    utterance: str = Field(
        description="Natural language user input matching CSV phrasing patterns. Use casual, abbreviated language like '$330k', 'opp', etc."
    )
    setup_entities: Dict[str, Any] = Field(
        description="Entity references from pool (client_id, opportunity_id, etc.). Must reference pre-existing entities."
    )
    expected_args: Dict[str, Any] = Field(
        description="Tool arguments matching the utterance. Amounts should be rounded (e.g., 330000 not 330801.38)."
    )
    expected_tool: str = Field(description="CRM tool name matching the task")


class CuratorScenarioGenerator(curator.LLM):
    """Curator LLM class for generating CRM scenarios using GPT-5-mini.
    
    GPT-5-mini has better structured output support than Gemini 2.5 Flash,
    which should eliminate null pollution and improve schema compliance.
    """

    response_format = ScenarioResponse  # Use structured outputs - GPT-5-mini supports this reliably
    batch = True  # Enable batch mode for 50% cost savings

    def __init__(
        self,
        entity_pool: Dict[str, List[Dict[str, Any]]],
        task_taxonomy: Dict[str, TaxonomyEntry],
        schema_constraints: Dict[str, Any],
        failure_blueprints: List[FailureBlueprint],
        model_name: str = "gpt-5-mini",  # Allow model switching for testing
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
        self.task_taxonomy = task_taxonomy
        self.schema_constraints = schema_constraints
        self.failure_blueprints = failure_blueprints

    def prompt(self, input: Dict) -> str:
        """Convert input row to LLM prompt."""
        task = input["task"]
        intent = input["intent"]
        is_failure = input.get("is_failure", False)
        failure_blueprint_dict = input.get("failure_blueprint")
        intent_blueprint_dict = input.get("intent_blueprint")

        # Get taxonomy entry for this task
        taxonomy_entry = self.task_taxonomy.get(task)
        typical_phrasing = taxonomy_entry.typical_phrasing if taxonomy_entry else ""

        # Build prompt based on success vs failure
        if is_failure and failure_blueprint_dict:
            # Reconstruct FailureBlueprint object from dict for prompt building
            failure_blueprint = self._dict_to_failure_blueprint(failure_blueprint_dict)
            prompt = self._build_failure_prompt(
                task, intent, failure_blueprint, typical_phrasing, intent_blueprint_dict
            )
        else:
            prompt = self._build_success_prompt(
                task, intent, typical_phrasing, intent_blueprint_dict
            )

        return prompt

    def _dict_to_failure_blueprint(self, bp_dict: Dict[str, Any]) -> Optional[FailureBlueprint]:
        """Convert failure blueprint dict back to FailureBlueprint object."""
        if not bp_dict:
            return None
        
        # Find matching blueprint in our list
        for bp in self.failure_blueprints:
            if bp.blueprint_id == bp_dict.get("blueprint_id"):
                return bp
        return None

    def _build_success_prompt(
        self,
        task: str,
        intent: str,
        typical_phrasing: str,
        intent_blueprint: Optional[Dict[str, Any]],
    ) -> str:
        """Build prompt for success scenario generation."""
        # Format entity pool for prompt
        entity_pool_text = self._format_entity_pool()

        # Get schema constraints for this task
        schema_info = self._get_schema_info_for_task(task, intent_blueprint)

        # Get argument template to show exactly which fields are expected
        argument_template_text = ""
        if intent_blueprint and intent_blueprint.get('argument_template'):
            arg_template = intent_blueprint['argument_template']
            argument_template_text = "Expected Arguments (use ONLY these fields):\n"
            for key, value in arg_template.items():
                argument_template_text += f"  - {key}: {type(value).__name__}\n"

        prompt = f"""You are generating CRM scenarios for a sales automation system. Generate a realistic scenario based on:

Task: {task}
Intent Category: {intent}
Typical User Phrasing Examples: {typical_phrasing or "Create a new opportunity, Show me open deals"}

Available Entity Pool:
{entity_pool_text}

Schema Constraints:
{schema_info}

{argument_template_text}

Requirements:
1. Utterance: Generate natural language matching the "Typical User Phrasing" patterns. Use casual, abbreviated language like "$330k" instead of "$330,000", "opp" instead of "opportunity", etc. Make it sound like real user input.

2. Setup Entities: Select entities from the available pool above. Use realistic company names (e.g., "Acme Corp", "TechVision Inc"), not generic names like "Client 214". Reference entities by their IDs.

3. Expected Args: Generate ONLY the fields listed in "Expected Arguments" above. Match the utterance contextually:
   - Amounts should be rounded to thousands (e.g., 330000 not 330801.38)
   - For opportunity tasks, always include probability (0-100)
   - Match enum values exactly (case-sensitive)
   - Do NOT include extra fields like "client", "opportunity", "client_name" etc. that are not in the template
   - Replace empty string values with realistic data

4. Expected Tool: Use the exact tool name: {intent_blueprint.get('expected_tool', task) if intent_blueprint else task}

Output JSON matching this schema:
{{
  "utterance": "string - natural language user input",
  "setup_entities": {{"client_id": "uuid", "opportunity_id": "uuid", ...}},
  "expected_args": {{ONLY fields from Expected Arguments above}},
  "expected_tool": "string - exact tool name"
}}
"""
        return prompt

    def _build_failure_prompt(
        self,
        task: str,
        intent: str,
        failure_blueprint: FailureBlueprint,
        typical_phrasing: str,
        intent_blueprint: Optional[Dict[str, Any]],
    ) -> str:
        """Build prompt for failure scenario generation."""
        entity_pool_text = self._format_entity_pool()

        # Get mutation description
        mutation_desc = ""
        if failure_blueprint.argument_mutations:
            for mutation in failure_blueprint.argument_mutations:
                mutation_desc += f"- {mutation.description}: {mutation.field_name} -> {mutation.mutation_type}\n"

        prompt = f"""You are generating CRM failure scenarios for a sales automation system. Generate a realistic scenario where a user would encounter this error:

Task: {task}
Intent Category: {intent}
Failure Category: {failure_blueprint.category.value}
Expected Error: {failure_blueprint.validator_expectation.expected_error_substring or "Validation error"}
Description: {failure_blueprint.description}
Mutation Pattern:
{mutation_desc}

Available Entity Pool:
{entity_pool_text}

Typical User Phrasing: {typical_phrasing or "Create a new opportunity"}

Requirements:
1. Utterance: Generate natural language that would lead to this error. Make it sound like a real user mistake or misunderstanding.

2. Setup Entities: Select entities from the available pool. Use realistic company names.

3. Expected Args: Apply the mutation pattern to create invalid input that triggers the expected error. The error should be contextual (not random) - explain why this error occurs.

4. Expected Tool: {intent_blueprint.get('expected_tool', task) if intent_blueprint else task}

Output JSON matching this schema:
{{
  "utterance": "string - natural language that would cause this error",
  "setup_entities": {{"client_id": "uuid", ...}},
  "expected_args": {{"field": "invalid_value", ...}},
  "expected_tool": "string - exact tool name"
}}
"""
        return prompt

    def _format_entity_pool(self) -> str:
        """Format entity pool for inclusion in prompts."""
        lines = []
        lines.append("Clients:")
        for client in self.entity_pool.get("clients", [])[:50]:  # Limit to 50 for prompt size
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

    def _get_schema_info_for_task(
        self, task: str, intent_blueprint: Optional[Dict[str, Any]]
    ) -> str:
        """Get schema constraints for a specific task."""
        if not intent_blueprint:
            return "Refer to schema constraints in fake_crm_tables_schema.json"

        info = []
        info.append(f"Expected Tool: {intent_blueprint.get('expected_tool', task)}")
        required_entities = intent_blueprint.get('required_entities', '')
        if required_entities:
            # Handle both string (from Dataset) and list formats
            if isinstance(required_entities, str):
                entities_list = [e.strip() for e in required_entities.split(',') if e.strip()]
            else:
                entities_list = required_entities
            if entities_list:
                info.append(f"Required Entities: {', '.join(entities_list)}")

        # Add common enum constraints based on task
        if "opportunity" in task.lower() or "opp" in task.lower():
            info.append(
                "Stage enum: Prospecting, Qualification, Proposal, Negotiation, Closed-Won, Closed-Lost"
            )
        if "client" in task.lower():
            info.append("Client Status enum: Active, Prospect, Inactive")
        if "quote" in task.lower():
            info.append("Quote Status enum: Draft, Sent, Approved, Rejected, Canceled")

        return "\n".join(info)

    def parse(self, input: Dict, response: ScenarioResponse) -> Dict:
        """Convert LLM response (Pydantic ScenarioResponse) to scenario dict.

        CRITICAL: Serializes nested dicts to JSON strings to avoid Arrow schema conflicts.
        Arrow cannot mix scalar columns with nested dict columns in the same Dataset row.
        """
        import json

        # Handle failure_blueprint extraction (now a dict)
        failure_blueprint_dict = input.get("failure_blueprint")
        if failure_blueprint_dict and isinstance(failure_blueprint_dict, dict):
            expected_error_substring = failure_blueprint_dict.get("expected_error_substring") or "Validation error"
            failure_category = failure_blueprint_dict.get("category")
        else:
            expected_error_substring = None
            failure_category = input.get("failure_category")

        # Ensure failure scenarios always have expected_error_substring
        if input.get("is_failure") and not expected_error_substring:
            expected_error_substring = "Validation error"

        # Extract from Pydantic model
        setup_entities = response.setup_entities if isinstance(response.setup_entities, dict) else dict(response.setup_entities)
        expected_args = response.expected_args if isinstance(response.expected_args, dict) else dict(response.expected_args)

        # Filter expected_args to only include fields from argument_template
        argument_template = input.get("intent_blueprint", {}).get("argument_template", {})
        filtered_args = self._filter_expected_args(expected_args, argument_template)

        # Serialize nested dicts to JSON strings for Arrow compatibility
        scenario = {
            "scenario_id": input["scenario_id"],
            "task": input["task"],
            "intent": input["intent"],
            "utterance": response.utterance,
            "expected_tool": response.expected_tool,
            "setup_entities_json": json.dumps(setup_entities),  # Serialize to JSON string
            "expected_args_json": json.dumps(filtered_args),    # Serialize to JSON string
            "expect_success": not input.get("is_failure", False),
            "expected_error_substring": expected_error_substring,
            "failure_category": failure_category,
            "verification_mode": input.get("verification_mode", "database"),
        }
        return scenario

    def _filter_expected_args(self, args: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
        """Filter expected_args to only include fields from argument_template, removing nulls."""
        if not template:
            # If no template, remove nulls but keep all fields
            return {k: v for k, v in args.items() if v is not None}
        
        # Only include fields that are in the template AND have non-null values
        filtered = {}
        for key in template.keys():
            if key in args:
                value = args[key]
                # Skip None values and empty strings (unless they're explicitly empty string in template)
                if value is not None and value != "":
                    filtered[key] = value
        
        return filtered

    def _normalize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize dictionary to ensure consistent types for Dataset serialization.
        
        Removes None values and ensures all values are serializable.
        Lists are converted to strings to avoid Arrow type conflicts.
        """
        normalized = {}
        for key, value in d.items():
            # Skip None values to avoid type conflicts
            if value is None:
                continue
            
            # Convert lists to comma-separated strings for Dataset compatibility
            if isinstance(value, list):
                normalized[key] = ",".join(str(v) for v in value)
            # Keep dicts as-is but normalize recursively
            elif isinstance(value, dict):
                normalized[key] = self._normalize_dict(value)
            # Keep primitives as-is
            else:
                normalized[key] = value
        
        return normalized


class CuratorDatasetGenerator:
    """Main generator class that orchestrates scenario generation using Curator."""

    def __init__(
        self,
        api: Optional[MockCrmApi] = None,
        sampler: Optional[EntitySampler] = None,
        seed: Optional[int] = 42,
    ):
        self.api = api or MockCrmApi()
        self.sampler = sampler or EntitySampler(
            self.api, SamplerConfig(seed=seed)
        )
        self.entity_pool: Dict[str, List[Dict[str, Any]]] = {}
        self.task_taxonomy = parse_taxonomy_csv()
        self.failure_blueprints = get_all_blueprints()
        self.intent_blueprints = {
            bp.task: bp for bp in get_all_intent_blueprints()
        }

        # Load schema constraints
        schema_path = Path(__file__).parent.parent / "data" / "fake_crm_tables_schema.json"
        if schema_path.exists():
            with schema_path.open() as f:
                self.schema_constraints = json.load(f)
        else:
            self.schema_constraints = {}

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
        contracts = []

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

            # Create contract for some clients
            if len(contracts) < 50:
                contract = self.sampler.sample_contract(client["client_id"], opp.opportunity_id)
                contracts.append({
                    "contract_id": contract.contract_id,
                    "client_id": contract.client_id,
                    "opportunity_id": contract.opportunity_id,
                    "value": round(contract.value or 0.0),
                    "status": contract.status.value if hasattr(contract.status, 'value') else str(contract.status),
                })

        self.entity_pool = {
            "clients": clients,
            "opportunities": opportunities,
            "quotes": quotes,
            "contacts": contacts,
            "contracts": contracts,
        }

        logger.info(
            f"Entity pool built: {len(clients)} clients, {len(opportunities)} opportunities, "
            f"{len(quotes)} quotes, {len(contacts)} contacts, {len(contracts)} contracts"
        )

    def generate_scenarios(
        self,
        target_count: int = 1500,
        success_ratio: float = 0.6,
        batch_size: int = 100,
    ) -> List[Scenario]:
        """Generate scenarios using Curator."""
        import random

        # Build entity pool if not already built
        if not self.entity_pool:
            self._build_entity_pool()

        # Calculate counts
        success_count = int(target_count * success_ratio)
        failure_count = target_count - success_count

        # Sample intents for success scenarios
        success_intents = random.choices(
            get_all_intent_blueprints(),
            weights=[bp.frequency for bp in get_all_intent_blueprints()],
            k=success_count,
        )

        # Sample intents for failure scenarios
        failure_intents = random.choices(
            get_all_intent_blueprints(),
            weights=[bp.frequency for bp in get_all_intent_blueprints()],
            k=failure_count,
        )

        # Prepare input dataset
        input_dataset = []
        scenario_counter = 0

        # Add success scenarios
        for intent in success_intents:
            scenario_counter += 1
            input_dataset.append({
                "scenario_id": f"SC-{scenario_counter:05d}",
                "task": intent.task,
                "intent": intent.intent_category,
                "is_failure": False,
                "intent_blueprint": {
                    "intent_id": intent.intent_id,
                    "task": intent.task,
                    "expected_tool": intent.expected_tool,
                    "required_entities": ",".join(intent.required_entities),  # Convert list to string for Dataset compatibility
                    "argument_template": dict(intent.argument_template) if hasattr(intent, 'argument_template') else {},
                },
                "verification_mode": "database",
            })

        # Add failure scenarios
        for intent in failure_intents:
            scenario_counter += 1
            # Find applicable failure blueprint
            applicable_failures = [
                fb
                for fb in self.failure_blueprints
                if fb.task == intent.task or fb.blueprint_id in intent.failure_blueprint_ids
            ]

            failure_blueprint = (
                random.choice(applicable_failures) if applicable_failures else None
            )

            # Convert FailureBlueprint to serializable dict for Dataset
            failure_blueprint_dict = None
            if failure_blueprint:
                failure_blueprint_dict = {
                    "blueprint_id": failure_blueprint.blueprint_id,
                    "category": failure_blueprint.category.value,
                    "task": failure_blueprint.task,
                    "expected_error_substring": failure_blueprint.validator_expectation.expected_error_substring,
                    "description": failure_blueprint.description,
                }

            input_dataset.append({
                "scenario_id": f"SC-{scenario_counter:05d}",
                "task": intent.task,
                "intent": intent.intent_category,
                "is_failure": True,
                "failure_blueprint": failure_blueprint_dict,
                "failure_category": failure_blueprint.category.value if failure_blueprint else None,
                "intent_blueprint": {
                    "intent_id": intent.intent_id,
                    "task": intent.task,
                    "expected_tool": intent.expected_tool,
                    "required_entities": ",".join(intent.required_entities),  # Convert list to string for Dataset compatibility
                    "argument_template": dict(intent.argument_template) if hasattr(intent, 'argument_template') else {},
                },
                "verification_mode": "database",
            })

        # Shuffle to mix success and failure
        random.shuffle(input_dataset)

        logger.info(f"Generating {len(input_dataset)} scenarios using Curator...")

        # Initialize Curator generator
        generator = CuratorScenarioGenerator(
            entity_pool=self.entity_pool,
            task_taxonomy=self.task_taxonomy,
            schema_constraints=self.schema_constraints,
            failure_blueprints=self.failure_blueprints,
        )

        # Generate in batches
        all_scenarios = []
        for i in range(0, len(input_dataset), batch_size):
            batch = input_dataset[i : i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} ({len(batch)} scenarios)...")

            try:
                # Convert to Dataset format
                dataset = Dataset.from_list(batch)

                # Generate via Curator
                result = generator(dataset)

                # Access results from result.dataset (Curator's output Dataset)
                logger.info(f"Processing {len(result.dataset)} scenarios from batch...")
                for row in result.dataset:
                    # Convert Dataset row to dict
                    row_dict = dict(row)
                    scenario = self._dict_to_scenario(row_dict)
                    if scenario:
                        all_scenarios.append(scenario)

                logger.info(f"Total scenarios so far: {len(all_scenarios)}")

            except Exception as e:
                logger.error(f"Error generating batch {i // batch_size + 1}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Continue with next batch
                continue

        logger.info(f"Generated {len(all_scenarios)} scenarios successfully")
        return all_scenarios

    def _dict_to_scenario(self, row: Dict[str, Any]) -> Optional[Scenario]:
        """Convert dict to Scenario dataclass.

        Deserializes JSON strings back to dicts for setup_entities and expected_args.
        """
        from .failure_blueprints import FailureCategory
        import json

        try:
            # Deserialize JSON strings back to dicts
            setup_entities = {}
            if "setup_entities_json" in row:
                setup_entities = json.loads(row["setup_entities_json"])
            elif "setup_entities" in row:
                setup_entities = row["setup_entities"]

            expected_args = {}
            if "expected_args_json" in row:
                expected_args = json.loads(row["expected_args_json"])
            elif "expected_args" in row:
                expected_args = row["expected_args"]

            # Convert failure_category string to enum if present
            failure_category = None
            if row.get("failure_category"):
                try:
                    failure_category = FailureCategory(row["failure_category"])
                except ValueError:
                    logger.warning(f"Invalid failure_category: {row.get('failure_category')}")

            # Parse verification_mode
            verification_mode_str = row.get("verification_mode", "database")
            if isinstance(verification_mode_str, str):
                # Map string to enum value
                mode_map = {
                    "database": VerificationMode.DATABASE,
                    "runtime_response": VerificationMode.RUNTIME_RESPONSE,
                    "unknown": VerificationMode.UNKNOWN,
                }
                verification_mode = mode_map.get(verification_mode_str.lower(), VerificationMode.DATABASE)
            elif isinstance(verification_mode_str, VerificationMode):
                verification_mode = verification_mode_str
            else:
                verification_mode = VerificationMode.DATABASE

            return Scenario(
                scenario_id=row["scenario_id"],
                task=row["task"],
                intent=row["intent"],
                utterance=row.get("utterance", ""),
                expected_tool=row["expected_tool"],
                setup_entities=setup_entities,
                expected_args=expected_args,
                expect_success=row.get("expect_success", True),
                expected_error_substring=row.get("expected_error_substring"),
                failure_category=failure_category,
                verification_mode=verification_mode,
            )
        except Exception as e:
            logger.error(f"Error converting dict to Scenario: {e}")
            logger.error(f"Row data: {row}")
            return None

