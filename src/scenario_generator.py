from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional
import random
import uuid

from .crm_sandbox import MockCrmApi
from .entity_sampler import EntitySampler, SamplerConfig
from .intent_blueprints import get_all_intent_blueprints, IntentBlueprint
from .failure_blueprints import get_all_blueprints, FailureBlueprint, FailureCategory
from .validators import VerificationMode


@dataclass
class Scenario:
    scenario_id: str
    task: str
    intent: str
    expected_tool: str
    setup_entities: Dict[str, Any]
    expected_args: Dict[str, Any]
    expect_success: bool
    expected_error_substring: Optional[str]
    failure_category: Optional[FailureCategory]
    verification_mode: VerificationMode


class DistributionSampler:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def sample_intents(self, total_count: int) -> List[IntentBlueprint]:
        blueprints = get_all_intent_blueprints()
        total_frequency = sum(bp.frequency for bp in blueprints)

        weights = [bp.frequency / total_frequency for bp in blueprints]

        sampled = random.choices(blueprints, weights=weights, k=total_count)
        return sampled


class ScenarioGenerator:
    def __init__(self, api: MockCrmApi, sampler: EntitySampler):
        self.api = api
        self.sampler = sampler
        self._scenario_counter = 0

    def _generate_scenario_id(self) -> str:
        self._scenario_counter += 1
        return f"SC-{self._scenario_counter:05d}"

    def _setup_entities_for_intent(self, intent: IntentBlueprint) -> Dict[str, Any]:
        setup = {}

        if "client" in intent.required_entities:
            client = self.sampler.sample_client()
            setup["client"] = client
            setup["client_id"] = client.client_id

        if "contact" in intent.required_entities and "client" in setup:
            contact = self.sampler.sample_contact(setup["client_id"])
            setup["contact"] = contact
            setup["contact_id"] = contact.contact_id

        if "opportunity" in intent.required_entities:
            if "client" not in setup:
                client = self.sampler.sample_client()
                setup["client"] = client
                setup["client_id"] = client.client_id

            opportunity = self.sampler.sample_opportunity(setup["client_id"])
            setup["opportunity"] = opportunity
            setup["opportunity_id"] = opportunity.opportunity_id

        if "quote" in intent.required_entities:
            if "opportunity" not in setup:
                if "client" not in setup:
                    client = self.sampler.sample_client()
                    setup["client"] = client
                    setup["client_id"] = client.client_id
                opportunity = self.sampler.sample_opportunity(setup["client_id"])
                setup["opportunity"] = opportunity
                setup["opportunity_id"] = opportunity.opportunity_id

            quote = self.sampler.sample_quote(setup["opportunity_id"])
            setup["quote"] = quote
            setup["quote_id"] = quote.quote_id

        if "contract" in intent.required_entities:
            if "client" not in setup:
                client = self.sampler.sample_client()
                setup["client"] = client
                setup["client_id"] = client.client_id

            contract = self.sampler.sample_contract(setup["client_id"])
            setup["contract"] = contract
            setup["contract_id"] = contract.contract_id

        if "entity" in intent.required_entities:
            if "client" not in setup:
                client = self.sampler.sample_client()
                setup["client"] = client
                setup["client_id"] = client.client_id
            setup["entity_type"] = "Client"
            setup["entity_id"] = setup["client_id"]

        return setup

    def _build_arguments_from_template(self, intent: IntentBlueprint, setup: Dict[str, Any], variant: Optional[Any] = None) -> Dict[str, Any]:
        args = dict(intent.argument_template)

        for key, value in args.items():
            if value == "" and key in setup:
                args[key] = setup[key]
            elif value == 0.0 and key == "amount":
                args[key] = round(random.uniform(self.sampler.config.min_amount, self.sampler.config.max_amount), 2)

        if variant:
            args.update(variant.arguments)

        if "stage" in args and args["stage"] == "":
            args["stage"] = "Prospecting"
        if "status" in args and args["status"] == "":
            if "client" in setup:
                args["status"] = "Active"
            elif "quote" in setup:
                args["status"] = "Draft"

        if "name" in args and args["name"] == "":
            if "opportunity" in setup:
                args["name"] = setup["opportunity"].name
            elif "client" in setup:
                args["name"] = setup["client"].name

        if "first_name" in args and args["first_name"] == "":
            args["first_name"] = random.choice(["John", "Jane", "Bob", "Alice"])
        if "last_name" in args and args["last_name"] == "":
            args["last_name"] = random.choice(["Doe", "Smith", "Johnson"])

        if "file_name" in args and args["file_name"] == "":
            args["file_name"] = "document.pdf"

        if "content" in args and args["content"] == "":
            args["content"] = "Note content"

        if "email" in args and args["email"] == "":
            args["email"] = "test@example.com"

        return args

    def generate_success_scenario(self, intent: IntentBlueprint) -> Scenario:
        setup = self._setup_entities_for_intent(intent)

        variant = random.choice(intent.success_variants) if intent.success_variants else None
        args = self._build_arguments_from_template(intent, setup, variant)

        return Scenario(
            scenario_id=self._generate_scenario_id(),
            task=intent.task,
            intent=intent.intent_category,
            expected_tool=intent.expected_tool,
            setup_entities=setup,
            expected_args=args,
            expect_success=True,
            expected_error_substring=None,
            failure_category=None,
            verification_mode=VerificationMode.DATABASE,
        )

    def generate_failure_scenario(self, intent: IntentBlueprint, failure: FailureBlueprint) -> Scenario:
        setup = self._setup_entities_for_intent(intent)
        base_args = self._build_arguments_from_template(intent, setup)

        if failure.argument_mutations:
            mutated_args = failure.argument_mutations[0].apply(base_args)
        else:
            mutated_args = base_args

        return Scenario(
            scenario_id=self._generate_scenario_id(),
            task=intent.task,
            intent=intent.intent_category,
            expected_tool=intent.expected_tool,
            setup_entities=setup,
            expected_args=mutated_args,
            expect_success=False,
            expected_error_substring=failure.validator_expectation.expected_error_substring,
            failure_category=failure.category,
            verification_mode=VerificationMode.DATABASE,
        )

    def generate_batch(self, target_count: int, success_ratio: float = 0.6) -> List[Scenario]:
        success_count = int(target_count * success_ratio)
        failure_count = target_count - success_count

        dist_sampler = DistributionSampler(seed=self.sampler.config.seed)
        success_intents = dist_sampler.sample_intents(success_count)
        failure_intents = dist_sampler.sample_intents(failure_count)

        scenarios = []

        for intent in success_intents:
            scenario = self.generate_success_scenario(intent)
            scenarios.append(scenario)

        failure_blueprints = get_all_blueprints()
        if not failure_blueprints:
            failure_blueprints = []

        for intent in failure_intents:
            applicable_failures = [
                fb for fb in failure_blueprints
                if fb.task == intent.task or fb.blueprint_id in intent.failure_blueprint_ids
            ]

            if applicable_failures:
                failure = random.choice(applicable_failures)
                scenario = self.generate_failure_scenario(intent, failure)
            else:
                scenario = self.generate_success_scenario(intent)

            scenarios.append(scenario)

        random.shuffle(scenarios)
        return scenarios
