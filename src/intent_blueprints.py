from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ArgumentVariation:
    description: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class IntentBlueprint:
    intent_id: str
    task: str
    intent_category: str
    frequency: int
    expected_tool: str
    required_entities: Tuple[str, ...]
    argument_template: Dict[str, Any]
    optional_fields: Tuple[str, ...]
    success_variants: Tuple[ArgumentVariation, ...]
    failure_blueprint_ids: Tuple[str, ...]


INTENT_BLUEPRINTS_REGISTRY: Dict[str, IntentBlueprint] = {}


def register_intent_blueprint(blueprint: IntentBlueprint) -> None:
    INTENT_BLUEPRINTS_REGISTRY[blueprint.intent_id] = blueprint


def get_intent_blueprint(intent_id: str) -> Optional[IntentBlueprint]:
    return INTENT_BLUEPRINTS_REGISTRY.get(intent_id)


def get_all_intent_blueprints() -> List[IntentBlueprint]:
    return list(INTENT_BLUEPRINTS_REGISTRY.values())


def get_blueprints_by_category(category: str) -> List[IntentBlueprint]:
    return [bp for bp in INTENT_BLUEPRINTS_REGISTRY.values() if bp.intent_category == category]


register_intent_blueprint(IntentBlueprint(
    intent_id="INT-001",
    task="create_new_opportunity",
    intent_category="Opportunity Management",
    frequency=3683,
    expected_tool="create_new_opportunity",
    required_entities=("client",),
    argument_template={"name": "", "client_id": "", "amount": 0.0, "stage": ""},
    optional_fields=("probability", "close_date", "owner", "notes"),
    success_variants=(
        ArgumentVariation("Standard opportunity", {"stage": "Prospecting", "probability": 20}),
        ArgumentVariation("Qualified opportunity", {"stage": "Qualification", "probability": 40}),
        ArgumentVariation("Proposal stage", {"stage": "Proposal", "probability": 60}),
    ),
    failure_blueprint_ids=("CNO-BP-001", "CNO-BP-002", "CNO-BP-003", "CNO-BP-004"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-002",
    task="modify_opportunity",
    intent_category="Opportunity Management",
    frequency=1564,
    expected_tool="modify_opportunity",
    required_entities=("opportunity",),
    argument_template={"opportunity_id": "", "updates": {}},
    optional_fields=(),
    success_variants=(
        ArgumentVariation("Update stage", {"updates": {"stage": "Negotiation"}}),
        ArgumentVariation("Update amount", {"updates": {"amount": 50000.0}}),
        ArgumentVariation("Update probability", {"updates": {"probability": 75}}),
    ),
    failure_blueprint_ids=("MOP-BP-001", "MOP-BP-002", "MOP-BP-003"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-003",
    task="create_new_client",
    intent_category="Client Management",
    frequency=1226,
    expected_tool="create_new_client",
    required_entities=(),
    argument_template={"name": "", "email": "", "status": ""},
    optional_fields=("industry", "phone", "address", "owner"),
    success_variants=(
        ArgumentVariation("Active client", {"status": "Active"}),
        ArgumentVariation("Prospect client", {"status": "Prospect"}),
        ArgumentVariation("Full details", {"status": "Active", "industry": "Technology", "phone": "+1-555-0100"}),
    ),
    failure_blueprint_ids=("CNC-BP-001", "CNC-BP-002", "CNC-BP-003", "CNC-BP-004"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-004",
    task="create_quote",
    intent_category="Quote Management",
    frequency=1156,
    expected_tool="create_quote",
    required_entities=("opportunity",),
    argument_template={"opportunity_id": "", "amount": 0.0, "status": ""},
    optional_fields=("version", "valid_until", "quote_prefix"),
    success_variants=(
        ArgumentVariation("Draft quote", {"status": "Draft"}),
        ArgumentVariation("Sent quote", {"status": "Sent"}),
        ArgumentVariation("With version", {"status": "Draft", "version": "Q-001"}),
    ),
    failure_blueprint_ids=("CQT-BP-001", "CQT-BP-002", "CQT-BP-003"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-005",
    task="opportunity_search",
    intent_category="Opportunity Management",
    frequency=991,
    expected_tool="opportunity_search",
    required_entities=(),
    argument_template={"stage": None, "client_id": None, "owner": None, "amount": None},
    optional_fields=("stage", "client_id", "owner", "amount"),
    success_variants=(
        ArgumentVariation("By stage", {"stage": "Proposal"}),
        ArgumentVariation("By owner", {"owner": "stephanie.wong"}),
        ArgumentVariation("By client", {"client_id": "client-123"}),
    ),
    failure_blueprint_ids=("SROP-BP-001", "SROP-BP-002"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-006",
    task="client_search",
    intent_category="Client Management",
    frequency=871,
    expected_tool="client_search",
    required_entities=(),
    argument_template={"status": None, "industry": None, "owner": None, "name": None},
    optional_fields=("status", "industry", "owner", "name"),
    success_variants=(
        ArgumentVariation("By status", {"status": "Active"}),
        ArgumentVariation("By industry", {"industry": "Technology"}),
        ArgumentVariation("By name", {"name": "Acme"}),
    ),
    failure_blueprint_ids=("SRCL-BP-001", "SRCL-BP-002"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-007",
    task="view_opportunity_details",
    intent_category="Opportunity Management",
    frequency=851,
    expected_tool="view_opportunity_details",
    required_entities=("opportunity",),
    argument_template={"opportunity_id": ""},
    optional_fields=(),
    success_variants=(
        ArgumentVariation("View details", {}),
    ),
    failure_blueprint_ids=("VOP-BP-001", "VOP-BP-002"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-008",
    task="modify_quote",
    intent_category="Quote Management",
    frequency=703,
    expected_tool="modify_quote",
    required_entities=("quote",),
    argument_template={"quote_id": "", "updates": {}},
    optional_fields=(),
    success_variants=(
        ArgumentVariation("Update status", {"updates": {"status": "Approved"}}),
        ArgumentVariation("Update amount", {"updates": {"amount": 75000.0}}),
    ),
    failure_blueprint_ids=("MQT-BP-001", "MQT-BP-002", "MQT-BP-003", "MQT-BP-004"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-009",
    task="modify_client",
    intent_category="Client Management",
    frequency=620,
    expected_tool="modify_client",
    required_entities=("client",),
    argument_template={"client_id": "", "updates": {}},
    optional_fields=(),
    success_variants=(
        ArgumentVariation("Update status", {"updates": {"status": "Inactive"}}),
        ArgumentVariation("Update phone", {"updates": {"phone": "+1-555-0200"}}),
    ),
    failure_blueprint_ids=("MCL-BP-001", "MCL-BP-002", "MCL-BP-003", "MCL-BP-004"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-010",
    task="create_new_contact",
    intent_category="Contact Management",
    frequency=519,
    expected_tool="create_new_contact",
    required_entities=("client",),
    argument_template={"first_name": "", "last_name": "", "client_id": ""},
    optional_fields=("email", "phone", "title", "notes"),
    success_variants=(
        ArgumentVariation("Basic contact", {}),
        ArgumentVariation("With email and phone", {"email": "contact@example.com", "phone": "+1-555-0300"}),
    ),
    failure_blueprint_ids=("CCON-BP-001", "CCON-BP-002", "CCON-BP-003"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-011",
    task="contact_search",
    intent_category="Contact Management",
    frequency=460,
    expected_tool="contact_search",
    required_entities=(),
    argument_template={"client_id": None, "email": None, "first_name": None, "last_name": None},
    optional_fields=("client_id", "email", "first_name", "last_name"),
    success_variants=(
        ArgumentVariation("By client", {"client_id": "client-123"}),
        ArgumentVariation("By name", {"first_name": "John"}),
    ),
    failure_blueprint_ids=("SRCON-BP-001", "SRCON-BP-002"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-012",
    task="quote_details",
    intent_category="Quote Management",
    frequency=403,
    expected_tool="quote_details",
    required_entities=("quote",),
    argument_template={"quote_id": ""},
    optional_fields=(),
    success_variants=(
        ArgumentVariation("View details", {}),
    ),
    failure_blueprint_ids=("VQTD-BP-001", "VQTD-BP-002"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-013",
    task="compare_quotes",
    intent_category="Quote Management",
    frequency=367,
    expected_tool="compare_quotes",
    required_entities=("quote", "quote"),
    argument_template={"quote_ids": []},
    optional_fields=(),
    success_variants=(
        ArgumentVariation("Compare two", {"quote_ids": ["q1", "q2"]}),
        ArgumentVariation("Compare three", {"quote_ids": ["q1", "q2", "q3"]}),
    ),
    failure_blueprint_ids=("CMPQ-BP-001", "CMPQ-BP-002"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-014",
    task="modify_contact",
    intent_category="Contact Management",
    frequency=321,
    expected_tool="modify_contact",
    required_entities=("contact",),
    argument_template={"contact_id": "", "updates": {}},
    optional_fields=(),
    success_variants=(
        ArgumentVariation("Update email", {"updates": {"email": "newemail@example.com"}}),
        ArgumentVariation("Update title", {"updates": {"title": "VP of Sales"}}),
    ),
    failure_blueprint_ids=("MCN-BP-001", "MCN-BP-002", "MCN-BP-003"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-015",
    task="delete_opportunity",
    intent_category="Opportunity Management",
    frequency=300,
    expected_tool="delete_opportunity",
    required_entities=("opportunity",),
    argument_template={"opportunity_id": ""},
    optional_fields=(),
    success_variants=(
        ArgumentVariation("Delete opportunity", {}),
    ),
    failure_blueprint_ids=("DOP-BP-001", "DOP-BP-002"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-016",
    task="delete_quote",
    intent_category="Quote Management",
    frequency=282,
    expected_tool="delete_quote",
    required_entities=("quote",),
    argument_template={"quote_id": ""},
    optional_fields=(),
    success_variants=(
        ArgumentVariation("Delete quote", {}),
    ),
    failure_blueprint_ids=("DQT-BP-001",),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-017",
    task="compare_quote_details",
    intent_category="Quote Management",
    frequency=251,
    expected_tool="compare_quote_details",
    required_entities=("quote", "quote"),
    argument_template={"quote_ids": []},
    optional_fields=(),
    success_variants=(
        ArgumentVariation("Compare two", {"quote_ids": ["q1", "q2"]}),
    ),
    failure_blueprint_ids=("CQTD-BP-001",),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-018",
    task="create_contract",
    intent_category="Contract Management",
    frequency=228,
    expected_tool="create_contract",
    required_entities=("client",),
    argument_template={"client_id": ""},
    optional_fields=("opportunity_id", "status", "value", "start_date", "end_date"),
    success_variants=(
        ArgumentVariation("Basic contract", {"status": "Pending"}),
        ArgumentVariation("With opportunity", {"status": "Active", "opportunity_id": "opp-123"}),
    ),
    failure_blueprint_ids=("CCT-BP-001", "CCT-BP-002"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-019",
    task="cancel_quote",
    intent_category="Quote Management",
    frequency=208,
    expected_tool="cancel_quote",
    required_entities=("quote",),
    argument_template={"quote_id": ""},
    optional_fields=(),
    success_variants=(
        ArgumentVariation("Cancel quote", {}),
    ),
    failure_blueprint_ids=("CNQ-BP-001",),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-020",
    task="quote_search",
    intent_category="Quote Management",
    frequency=192,
    expected_tool="quote_search",
    required_entities=(),
    argument_template={"status": None, "opportunity_id": None, "amount": None},
    optional_fields=("status", "opportunity_id", "amount"),
    success_variants=(
        ArgumentVariation("By status", {"status": "Sent"}),
        ArgumentVariation("By opportunity", {"opportunity_id": "opp-123"}),
    ),
    failure_blueprint_ids=("SRQT-BP-001", "SRQT-BP-002"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-021",
    task="clone_opportunity",
    intent_category="Opportunity Management",
    frequency=176,
    expected_tool="clone_opportunity",
    required_entities=("opportunity",),
    argument_template={"opportunity_id": ""},
    optional_fields=(),
    success_variants=(
        ArgumentVariation("Clone opportunity", {}),
    ),
    failure_blueprint_ids=("CLOP-BP-001", "CLOP-BP-002"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-022",
    task="contract_search",
    intent_category="Contract Management",
    frequency=154,
    expected_tool="contract_search",
    required_entities=(),
    argument_template={"status": None, "client_id": None},
    optional_fields=("status", "client_id"),
    success_variants=(
        ArgumentVariation("By status", {"status": "Active"}),
        ArgumentVariation("By client", {"client_id": "client-123"}),
    ),
    failure_blueprint_ids=("SRCT-BP-001", "SRCT-BP-002"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-023",
    task="summarize_opportunities",
    intent_category="Opportunity Management",
    frequency=138,
    expected_tool="summarize_opportunities",
    required_entities=(),
    argument_template={"stage": None, "owner": None},
    optional_fields=("stage", "owner"),
    success_variants=(
        ArgumentVariation("All opportunities", {}),
        ArgumentVariation("By stage", {"stage": "Proposal"}),
    ),
    failure_blueprint_ids=("SUMOP-BP-001", "SUMOP-BP-002"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-024",
    task="company_search",
    intent_category="Company/Account Management",
    frequency=103,
    expected_tool="company_search",
    required_entities=(),
    argument_template={"type": None, "industry": None, "name": None},
    optional_fields=("type", "industry", "name"),
    success_variants=(
        ArgumentVariation("By type", {"type": "Partner"}),
        ArgumentVariation("By industry", {"industry": "Technology"}),
    ),
    failure_blueprint_ids=("SRCOM-BP-001", "SRCOM-BP-002"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-025",
    task="add_note",
    intent_category="Notes & Collaboration",
    frequency=78,
    expected_tool="add_note",
    required_entities=("entity",),
    argument_template={"entity_type": "", "entity_id": "", "content": ""},
    optional_fields=("created_by", "created_at"),
    success_variants=(
        ArgumentVariation("Client note", {"entity_type": "Client"}),
        ArgumentVariation("Opportunity note", {"entity_type": "Opportunity"}),
    ),
    failure_blueprint_ids=("ANT-BP-001", "ANT-BP-002"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-026",
    task="upload_document",
    intent_category="Document Management",
    frequency=42,
    expected_tool="upload_document",
    required_entities=("entity",),
    argument_template={"entity_type": "", "entity_id": "", "file_name": ""},
    optional_fields=("uploaded_by", "uploaded_at"),
    success_variants=(
        ArgumentVariation("PDF upload", {"file_name": "contract.pdf"}),
        ArgumentVariation("DOCX upload", {"file_name": "proposal.docx"}),
    ),
    failure_blueprint_ids=("UD-BP-001", "UD-BP-002", "UD-BP-003", "UD-BP-004", "UD-BP-005"),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-027",
    task="opportunity_details",
    intent_category="Opportunity Management",
    frequency=15,
    expected_tool="opportunity_details",
    required_entities=("opportunity",),
    argument_template={"opportunity_id": ""},
    optional_fields=(),
    success_variants=(
        ArgumentVariation("View details", {}),
    ),
    failure_blueprint_ids=("VOPD-BP-001",),
))

register_intent_blueprint(IntentBlueprint(
    intent_id="INT-028",
    task="quote_prefixes",
    intent_category="Quote Management",
    frequency=1,
    expected_tool="quote_prefixes",
    required_entities=(),
    argument_template={},  # No arguments for this task
    optional_fields=(),
    success_variants=(
        ArgumentVariation("List prefixes", {}),
    ),
    failure_blueprint_ids=(),
))
