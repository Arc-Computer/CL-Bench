"""Workflow templates for multi-turn conversation generation.

Defines 8 common CRM workflow patterns used to generate conversations with
realistic multi-turn interactions. Each template specifies:
- Complexity level (simple, medium, complex)
- Turn sequence (list of turn templates)
- Entity dependencies (what entities are needed/created)
- Cross-turn references (how turns reference previous entities)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence


@dataclass
class TurnTemplate:
    """Template for a single turn in a workflow.
    
    Attributes:
        turn_number: Sequential turn number (1-indexed)
        tool_name: Expected tool to be called
        argument_template: Template for expected arguments (may contain {{turn_N.field}})
        user_utterance_pattern: Pattern for generating natural language utterance
        references_previous_turns: List of turn numbers this turn references
        generation_params: Optional generation parameters for Curator (temperature, etc.)
    """
    turn_number: int
    tool_name: str
    argument_template: Dict[str, Any]
    user_utterance_pattern: str
    references_previous_turns: List[int] = field(default_factory=list)
    generation_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.references_previous_turns is None:
            self.references_previous_turns = []


@dataclass
class WorkflowTemplate:
    """Template for a complete CRM workflow conversation.
    
    Attributes:
        workflow_id: Unique identifier
        workflow_category: Category name (e.g., "Client Management")
        complexity_level: simple, medium, or complex
        turn_templates: List of TurnTemplate objects in order
        required_initial_entities: Entities that must exist before conversation starts
        entities_created: Entities created during conversation (for validation)
        description: Human-readable description of the workflow
    """
    workflow_id: str
    workflow_category: str
    complexity_level: Literal["simple", "medium", "complex"]
    turn_templates: List[TurnTemplate]
    required_initial_entities: List[str] = None
    entities_created: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.required_initial_entities is None:
            self.required_initial_entities = []
        if self.entities_created is None:
            self.entities_created = []
        
        # Validate turn numbers are sequential
        turn_numbers = [t.turn_number for t in self.turn_templates]
        expected_numbers = list(range(1, len(self.turn_templates) + 1))
        if turn_numbers != expected_numbers:
            raise ValueError(
                f"Turn numbers must be sequential starting at 1. "
                f"Got {turn_numbers}, expected {expected_numbers}"
            )


# 1. Client Management (simple, 1-3 turns)
CLIENT_MANAGEMENT = WorkflowTemplate(
    workflow_id="WF-001",
    workflow_category="Client Management",
    complexity_level="simple",
    description="Client search, creation, and modification workflows",
    turn_templates=[
        TurnTemplate(
            turn_number=1,
            tool_name="client_search",
            argument_template={"name": ""},
            user_utterance_pattern="Show me {entity_name}",
            references_previous_turns=[],
        ),
        TurnTemplate(
            turn_number=2,
            tool_name="modify_client",
            argument_template={"client_id": "{{turn_1.client_id}}", "updates": {"status": ""}},
            user_utterance_pattern="Update {entity_name}'s status to {status}",
            references_previous_turns=[1],
        ),
    ],
    required_initial_entities=["client"],
    entities_created=[],
)

# 2. Contact Management (simple, 2-3 turns)
CONTACT_MANAGEMENT = WorkflowTemplate(
    workflow_id="WF-002",
    workflow_category="Contact Management",
    complexity_level="simple",
    description="Search client → create contact for that client",
    turn_templates=[
        TurnTemplate(
            turn_number=1,
            tool_name="client_search",
            argument_template={"name": ""},
            user_utterance_pattern="Find {entity_name}",
            references_previous_turns=[],
        ),
        TurnTemplate(
            turn_number=2,
            tool_name="create_new_contact",
            argument_template={
                "client_id": "{{turn_1.client_id}}",
                "first_name": "",
                "last_name": "",
            },
            user_utterance_pattern="Create a contact for them",
            references_previous_turns=[1],
        ),
        TurnTemplate(
            turn_number=3,
            tool_name="modify_contact",
            argument_template={"contact_id": "{{turn_2.contact_id}}", "updates": {"email": ""}},
            user_utterance_pattern="Update their email to {email}",
            references_previous_turns=[2],
        ),
    ],
    required_initial_entities=["client"],
    entities_created=["contact"],
)

# 3. Opportunity Management (medium, 3-5 turns)
OPPORTUNITY_MANAGEMENT = WorkflowTemplate(
    workflow_id="WF-003",
    workflow_category="Opportunity Management",
    complexity_level="medium",
    description="Search client → create opportunity → modify opportunity",
    turn_templates=[
        TurnTemplate(
            turn_number=1,
            tool_name="client_search",
            argument_template={"name": ""},
            user_utterance_pattern="Show me {entity_name}",
            references_previous_turns=[],
        ),
        TurnTemplate(
            turn_number=2,
            tool_name="create_new_opportunity",
            argument_template={
                "client_id": "{{turn_1.client_id}}",
                "name": "",
                "amount": 0.0,
                "stage": "",
            },
            user_utterance_pattern="Create an opp for them",
            references_previous_turns=[1],
        ),
        TurnTemplate(
            turn_number=3,
            tool_name="modify_opportunity",
            argument_template={
                "opportunity_id": "{{turn_2.opportunity_id}}",
                "updates": {"stage": ""},
            },
            user_utterance_pattern="Update that opp to {stage} stage",
            references_previous_turns=[2],
        ),
        TurnTemplate(
            turn_number=4,
            tool_name="modify_opportunity",
            argument_template={
                "opportunity_id": "{{turn_2.opportunity_id}}",
                "updates": {"probability": 0},
            },
            user_utterance_pattern="Set probability to {probability}%",
            references_previous_turns=[2],
        ),
    ],
    required_initial_entities=["client"],
    entities_created=["opportunity"],
)

# 4. Quote Generation (medium, 4-6 turns)
QUOTE_GENERATION = WorkflowTemplate(
    workflow_id="WF-004",
    workflow_category="Quote Management",
    complexity_level="medium",
    description="Search opportunity → create quote → modify quote",
    turn_templates=[
        TurnTemplate(
            turn_number=1,
            tool_name="opportunity_search",
            argument_template={"name": ""},
            user_utterance_pattern="Find {entity_name} opportunity",
            references_previous_turns=[],
        ),
        TurnTemplate(
            turn_number=2,
            tool_name="create_quote",
            argument_template={
                "opportunity_id": "{{turn_1.opportunity_id}}",
                "amount": 0.0,
                "status": "",
            },
            user_utterance_pattern="Create a quote for that opp",
            references_previous_turns=[1],
        ),
        TurnTemplate(
            turn_number=3,
            tool_name="modify_quote",
            argument_template={
                "quote_id": "{{turn_2.quote_id}}",
                "updates": {"status": ""},
            },
            user_utterance_pattern="Update the quote status to {status}",
            references_previous_turns=[2],
        ),
        TurnTemplate(
            turn_number=4,
            tool_name="compare_quotes",
            argument_template={"quote_ids": ["{{turn_2.quote_id}}"]},
            user_utterance_pattern="Show me all quotes for that opp",
            references_previous_turns=[1],
        ),
    ],
    required_initial_entities=["opportunity"],
    entities_created=["quote"],
)

# 5. Client Onboarding (medium, 5-6 turns)
CLIENT_ONBOARDING = WorkflowTemplate(
    workflow_id="WF-005",
    workflow_category="Client Onboarding",
    complexity_level="medium",
    description="Create client → create contact → create opportunity",
    turn_templates=[
        TurnTemplate(
            turn_number=1,
            tool_name="create_new_client",
            argument_template={"name": "", "email": "", "status": ""},
            user_utterance_pattern="Create a new client: {entity_name}",
            references_previous_turns=[],
        ),
        TurnTemplate(
            turn_number=2,
            tool_name="create_new_contact",
            argument_template={
                "client_id": "{{turn_1.client_id}}",
                "first_name": "",
                "last_name": "",
            },
            user_utterance_pattern="Add a contact for them",
            references_previous_turns=[1],
        ),
        TurnTemplate(
            turn_number=3,
            tool_name="create_new_opportunity",
            argument_template={
                "client_id": "{{turn_1.client_id}}",
                "name": "",
                "amount": 0.0,
                "stage": "",
            },
            user_utterance_pattern="Create an opportunity for {entity_name}",
            references_previous_turns=[1],
        ),
        TurnTemplate(
            turn_number=4,
            tool_name="add_note",
            argument_template={
                "entity_type": "Client",
                "entity_id": "{{turn_1.client_id}}",
                "content": "",
            },
            user_utterance_pattern="Add a note about {entity_name}",
            references_previous_turns=[1],
        ),
    ],
    required_initial_entities=[],
    entities_created=["client", "contact", "opportunity"],
)

# 6. Deal Pipeline (complex, 7-10 turns)
DEAL_PIPELINE = WorkflowTemplate(
    workflow_id="WF-006",
    workflow_category="Deal Pipeline",
    complexity_level="complex",
    description="Full deal pipeline: search → qualify → quote → negotiate → contract",
    turn_templates=[
        TurnTemplate(
            turn_number=1,
            tool_name="client_search",
            argument_template={"name": ""},
            user_utterance_pattern="Show me {entity_name}",
            references_previous_turns=[],
        ),
        TurnTemplate(
            turn_number=2,
            tool_name="create_new_opportunity",
            argument_template={
                "client_id": "{{turn_1.client_id}}",
                "name": "",
                "amount": 0.0,
                "stage": "Prospecting",
            },
            user_utterance_pattern="Create a new opp for them",
            references_previous_turns=[1],
        ),
        TurnTemplate(
            turn_number=3,
            tool_name="modify_opportunity",
            argument_template={
                "opportunity_id": "{{turn_2.opportunity_id}}",
                "updates": {"stage": "Qualification"},
            },
            user_utterance_pattern="Move that opp to Qualification",
            references_previous_turns=[2],
        ),
        TurnTemplate(
            turn_number=4,
            tool_name="create_quote",
            argument_template={
                "opportunity_id": "{{turn_2.opportunity_id}}",
                "amount": 0.0,
                "status": "Draft",
            },
            user_utterance_pattern="Generate a quote for that opp",
            references_previous_turns=[2],
        ),
        TurnTemplate(
            turn_number=5,
            tool_name="modify_quote",
            argument_template={
                "quote_id": "{{turn_4.quote_id}}",
                "updates": {"status": "Sent"},
            },
            user_utterance_pattern="Send the quote",
            references_previous_turns=[4],
        ),
        TurnTemplate(
            turn_number=6,
            tool_name="modify_opportunity",
            argument_template={
                "opportunity_id": "{{turn_2.opportunity_id}}",
                "updates": {"stage": "Negotiation"},
            },
            user_utterance_pattern="Move the opp to Negotiation",
            references_previous_turns=[2],
        ),
        TurnTemplate(
            turn_number=7,
            tool_name="modify_opportunity",
            argument_template={
                "opportunity_id": "{{turn_2.opportunity_id}}",
                "updates": {"stage": "Closed-Won"},
            },
            user_utterance_pattern="Mark it as Closed-Won",
            references_previous_turns=[2],
        ),
    ],
    required_initial_entities=["client"],
    entities_created=["opportunity", "quote"],
)

# 7. Document Workflow (simple, 2-3 turns)
DOCUMENT_WORKFLOW = WorkflowTemplate(
    workflow_id="WF-007",
    workflow_category="Document Management",
    complexity_level="simple",
    description="Search entity → upload document",
    turn_templates=[
        TurnTemplate(
            turn_number=1,
            tool_name="client_search",
            argument_template={"name": ""},
            user_utterance_pattern="Find {entity_name}",
            references_previous_turns=[],
        ),
        TurnTemplate(
            turn_number=2,
            tool_name="upload_document",
            argument_template={
                "entity_type": "Client",
                "entity_id": "{{turn_1.client_id}}",
                "file_name": "",
            },
            user_utterance_pattern="Upload a document for them",
            references_previous_turns=[1],
        ),
        TurnTemplate(
            turn_number=3,
            tool_name="add_note",
            argument_template={
                "entity_type": "Client",
                "entity_id": "{{turn_1.client_id}}",
                "content": "",
            },
            user_utterance_pattern="Add a note about the document",
            references_previous_turns=[1],
        ),
    ],
    required_initial_entities=["client"],
    entities_created=["document"],
)

# 8. Multi-Entity Search (medium, 4-5 turns)
MULTI_ENTITY_SEARCH = WorkflowTemplate(
    workflow_id="WF-008",
    workflow_category="Multi-Entity Search",
    complexity_level="medium",
    description="Search client → search opportunities → search quotes",
    turn_templates=[
        TurnTemplate(
            turn_number=1,
            tool_name="client_search",
            argument_template={"name": ""},
            user_utterance_pattern="Show me {entity_name}",
            references_previous_turns=[],
        ),
        TurnTemplate(
            turn_number=2,
            tool_name="opportunity_search",
            argument_template={"client_id": "{{turn_1.client_id}}"},
            user_utterance_pattern="Show me all opportunities for them",
            references_previous_turns=[1],
        ),
        TurnTemplate(
            turn_number=3,
            tool_name="view_opportunity_details",
            argument_template={"opportunity_id": "{{turn_2.opportunity_id}}"},
            user_utterance_pattern="Show me details for {opportunity_name}",
            references_previous_turns=[2],
        ),
        TurnTemplate(
            turn_number=4,
            tool_name="quote_search",
            argument_template={"opportunity_id": "{{turn_2.opportunity_id}}"},
            user_utterance_pattern="Show me quotes for that opp",
            references_previous_turns=[2],
        ),
    ],
    required_initial_entities=["client", "opportunity", "quote"],
    entities_created=[],
)

# 9. Opportunity Summary (medium, 4 turns)
OPPORTUNITY_SUMMARY = WorkflowTemplate(
    workflow_id="WF-009",
    workflow_category="Opportunity Insights",
    complexity_level="medium",
    description="Summarize a client's pipeline and inspect key opportunities",
    turn_templates=[
        TurnTemplate(
            turn_number=1,
            tool_name="client_search",
            argument_template={"name": ""},
            user_utterance_pattern="Pull up {entity_name}",
            references_previous_turns=[],
        ),
        TurnTemplate(
            turn_number=2,
            tool_name="opportunity_search",
            argument_template={"client_id": "{{turn_1.client_id}}", "stage": ""},
            user_utterance_pattern="Show their opportunities in {stage}",
            references_previous_turns=[1],
        ),
        TurnTemplate(
            turn_number=3,
            tool_name="summarize_opportunities",
            argument_template={
                "client_id": "{{turn_1.client_id}}",
                "stage": "",
                "owner": "",
            },
            user_utterance_pattern="Summarize those opps for {stage} owned by {owner}",
            references_previous_turns=[1],
        ),
        TurnTemplate(
            turn_number=4,
            tool_name="view_opportunity_details",
            argument_template={"opportunity_id": "{{turn_2.opportunity_id}}"},
            user_utterance_pattern="Open details for {opportunity_name}",
            references_previous_turns=[2],
        ),
    ],
    required_initial_entities=["client", "opportunity"],
    entities_created=[],
)

# 10. Quote Cleanup (medium, 3 turns)
QUOTE_CLEANUP = WorkflowTemplate(
    workflow_id="WF-010",
    workflow_category="Quote Management",
    complexity_level="medium",
    description="Review active quotes, cancel outdated ones, and remove them",
    turn_templates=[
        TurnTemplate(
            turn_number=1,
            tool_name="quote_search",
            argument_template={"opportunity_id": "", "status": ""},
            user_utterance_pattern="Find quotes for {opportunity_name} in {status} status",
            references_previous_turns=[],
        ),
        TurnTemplate(
            turn_number=2,
            tool_name="cancel_quote",
            argument_template={"quote_id": ""},
            user_utterance_pattern="Cancel that quote",
            references_previous_turns=[1],
        ),
        TurnTemplate(
            turn_number=3,
            tool_name="delete_quote",
            argument_template={"quote_id": ""},
            user_utterance_pattern="Delete the canceled quote",
            references_previous_turns=[1],
        ),
    ],
    required_initial_entities=["quote", "opportunity"],
    entities_created=[],
)

# 11. Contract Review (medium, 3 turns)
CONTRACT_REVIEW = WorkflowTemplate(
    workflow_id="WF-011",
    workflow_category="Contract Management",
    complexity_level="medium",
    description="Locate contracts and attach supporting collateral or notes",
    turn_templates=[
        TurnTemplate(
            turn_number=1,
            tool_name="contract_search",
            argument_template={"client_id": "", "status": ""},
            user_utterance_pattern="Find contracts for {entity_name} in {status}",
            references_previous_turns=[],
        ),
        TurnTemplate(
            turn_number=2,
            tool_name="upload_document",
            argument_template={
                "entity_type": "Contract",
                "entity_id": "{{turn_1.contract_id}}",
                "file_name": "",
            },
            user_utterance_pattern="Upload supporting documents to that contract",
            references_previous_turns=[1],
        ),
        TurnTemplate(
            turn_number=3,
            tool_name="add_note",
            argument_template={
                "entity_type": "Contract",
                "entity_id": "{{turn_1.contract_id}}",
                "content": "",
            },
            user_utterance_pattern="Add an internal note about the contract",
            references_previous_turns=[1],
        ),
    ],
    required_initial_entities=["contract", "client"],
    entities_created=["document"],
)

# 12. Opportunity Clone (medium, 4 turns)
OPPORTUNITY_CLONE = WorkflowTemplate(
    workflow_id="WF-012",
    workflow_category="Opportunity Management",
    complexity_level="medium",
    description="Clone an existing opportunity and advance the duplicate",
    turn_templates=[
        TurnTemplate(
            turn_number=1,
            tool_name="opportunity_search",
            argument_template={"client_id": "", "name": ""},
            user_utterance_pattern="Find {opportunity_name} for {entity_name}",
            references_previous_turns=[],
        ),
        TurnTemplate(
            turn_number=2,
            tool_name="clone_opportunity",
            argument_template={"opportunity_id": "{{turn_1.opportunity_id}}"},
            user_utterance_pattern="Clone that opportunity",
            references_previous_turns=[1],
        ),
        TurnTemplate(
            turn_number=3,
            tool_name="modify_opportunity",
            argument_template={
                "opportunity_id": "{{turn_2.opportunity_id}}",
                "updates": {"stage": "", "probability": 0},
            },
            user_utterance_pattern="Update the cloned opportunity to {stage} at {probability}%",
            references_previous_turns=[2],
        ),
        TurnTemplate(
            turn_number=4,
            tool_name="upload_document",
            argument_template={
                "entity_type": "Opportunity",
                "entity_id": "{{turn_2.opportunity_id}}",
                "file_name": "",
            },
            user_utterance_pattern="Attach supporting docs to the cloned opportunity",
            references_previous_turns=[2],
        ),
    ],
    required_initial_entities=["client", "opportunity"],
    entities_created=["opportunity", "document"],
)


# Registry of all workflow templates
WORKFLOW_TEMPLATES: Dict[str, WorkflowTemplate] = {
    "client_management": CLIENT_MANAGEMENT,
    "contact_management": CONTACT_MANAGEMENT,
    "opportunity_management": OPPORTUNITY_MANAGEMENT,
    "quote_generation": QUOTE_GENERATION,
    "client_onboarding": CLIENT_ONBOARDING,
    "deal_pipeline": DEAL_PIPELINE,
    "document_workflow": DOCUMENT_WORKFLOW,
    "multi_entity_search": MULTI_ENTITY_SEARCH,
    "opportunity_summary": OPPORTUNITY_SUMMARY,
    "quote_cleanup": QUOTE_CLEANUP,
    "contract_review": CONTRACT_REVIEW,
    "opportunity_clone": OPPORTUNITY_CLONE,
}


def get_workflow_template(workflow_id: str) -> Optional[WorkflowTemplate]:
    """Get workflow template by ID."""
    return WORKFLOW_TEMPLATES.get(workflow_id)


def get_templates_by_complexity(complexity: Literal["simple", "medium", "complex"]) -> List[WorkflowTemplate]:
    """Get all templates for a given complexity level."""
    return [t for t in WORKFLOW_TEMPLATES.values() if t.complexity_level == complexity]


def get_all_templates() -> List[WorkflowTemplate]:
    """Get all workflow templates."""
    return list(WORKFLOW_TEMPLATES.values())


@dataclass
class WorkflowChain:
    """Defines a sequence of workflow templates that form a multi-segment conversation.
    
    Attributes:
        chain_id: Unique identifier for this chain
        workflow_sequence: List of workflow template IDs (keys from WORKFLOW_TEMPLATES)
        success_pattern: List of booleans indicating per-segment success/failure expectations
        entity_handoff_rules: Dictionary mapping entity types to handoff rules
            (e.g., {"client_id": "propagate", "opportunity_id": "create_in_segment_2"})
        description: Human-readable description of the chain
    """
    chain_id: str
    workflow_sequence: List[str]
    success_pattern: List[bool]
    entity_handoff_rules: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    
    def __post_init__(self):
        if len(self.workflow_sequence) != len(self.success_pattern):
            raise ValueError(
                f"workflow_sequence length ({len(self.workflow_sequence)}) must match "
                f"success_pattern length ({len(self.success_pattern)})"
            )
        
        # Validate all workflow IDs exist
        for workflow_id in self.workflow_sequence:
            if workflow_id not in WORKFLOW_TEMPLATES:
                raise ValueError(f"Workflow template '{workflow_id}' not found in WORKFLOW_TEMPLATES")


# Define realistic workflow chains (success + failure variants)

# Chain 1: Client Onboarding → Deal Pipeline → Contract Review
CHAIN_ONBOARDING_PIPELINE_CONTRACT_SUCCESS = WorkflowChain(
    chain_id="CHAIN-001A",
    workflow_sequence=["client_onboarding", "deal_pipeline", "quote_generation"],
    success_pattern=[True, True, True],
    entity_handoff_rules={
        "client_id": "propagate",  # Client from onboarding used in pipeline
        "opportunity_id": "propagate",  # Opportunity from onboarding used in pipeline
        "quote_id": "propagate",  # Quote from pipeline used in contract review
    },
    description="Full client lifecycle: onboard new client, manage deal pipeline, generate contract",
)

CHAIN_ONBOARDING_PIPELINE_CONTRACT_FAILURE = WorkflowChain(
    chain_id="CHAIN-001B",
    workflow_sequence=["client_onboarding", "deal_pipeline", "quote_generation"],
    success_pattern=[True, False, True],
    entity_handoff_rules={
        "client_id": "propagate",
        "opportunity_id": "propagate",
        "quote_id": "propagate",
    },
    description="Client lifecycle with expected failure during pipeline execution",
)

# Chain 2: Client Management → Opportunity Management → Quote Generation
CHAIN_CLIENT_OPP_QUOTE_SUCCESS = WorkflowChain(
    chain_id="CHAIN-002A",
    workflow_sequence=["client_management", "opportunity_management", "quote_generation"],
    success_pattern=[True, True, True],
    entity_handoff_rules={
        "client_id": "propagate",
        "opportunity_id": "propagate",
    },
    description="Manage existing client, create/modify opportunity, generate quote",
)

CHAIN_CLIENT_OPP_QUOTE_FAILURE = WorkflowChain(
    chain_id="CHAIN-002B",
    workflow_sequence=["client_management", "opportunity_management", "quote_generation"],
    success_pattern=[True, False, True],
    entity_handoff_rules={
        "client_id": "propagate",
        "opportunity_id": "propagate",
    },
    description="Opportunity management chain with a deliberate failure segment",
)

# Chain 3: Contact Management → Document Workflow → Notes
# Note: We'll need to create an add_note workflow template or use existing one
CHAIN_CONTACT_DOCUMENT_NOTE_SUCCESS = WorkflowChain(
    chain_id="CHAIN-003A",
    workflow_sequence=["contact_management", "document_workflow", "client_management"],
    success_pattern=[True, True, True],
    entity_handoff_rules={
        "client_id": "propagate",
        "contact_id": "propagate",
        "document_id": "propagate",
    },
    description="Manage contact, upload documents, add notes",
)

CHAIN_CONTACT_DOCUMENT_NOTE_FAILURE = WorkflowChain(
    chain_id="CHAIN-003B",
    workflow_sequence=["contact_management", "document_workflow", "client_management"],
    success_pattern=[True, False, True],
    entity_handoff_rules={
        "client_id": "propagate",
        "contact_id": "propagate",
        "document_id": "propagate",
    },
    description="Contact/document workflow with expected document upload failure",
)

# Chain 4: Opportunity Search → Quote Generation → Multi-Entity Search
CHAIN_SEARCH_QUOTE_REVIEW_SUCCESS = WorkflowChain(
    chain_id="CHAIN-004A",
    workflow_sequence=["multi_entity_search", "quote_generation", "multi_entity_search"],
    success_pattern=[True, True, True],
    entity_handoff_rules={
        "client_id": "propagate",
        "opportunity_id": "propagate",
        "quote_id": "propagate",
    },
    description="Search opportunities, generate quote, review all related entities",
)

CHAIN_SEARCH_QUOTE_REVIEW_FAILURE = WorkflowChain(
    chain_id="CHAIN-004B",
    workflow_sequence=["multi_entity_search", "quote_generation", "multi_entity_search"],
    success_pattern=[True, False, True],
    entity_handoff_rules={
        "client_id": "propagate",
        "opportunity_id": "propagate",
        "quote_id": "propagate",
    },
    description="Search/quote workflow with expected quote segment failure",
)

# Chain 5: Client Onboarding → Opportunity Management → Deal Pipeline
CHAIN_ONBOARDING_OPP_DEAL_SUCCESS = WorkflowChain(
    chain_id="CHAIN-005A",
    workflow_sequence=["client_onboarding", "opportunity_management", "deal_pipeline"],
    success_pattern=[True, True, True],
    entity_handoff_rules={
        "client_id": "propagate",
        "opportunity_id": "propagate",
        "contact_id": "propagate",
    },
    description="Onboard client, manage opportunities, track deal pipeline",
)

CHAIN_ONBOARDING_OPP_DEAL_FAILURE = WorkflowChain(
    chain_id="CHAIN-005B",
    workflow_sequence=["client_onboarding", "opportunity_management", "deal_pipeline"],
    success_pattern=[True, True, False],
    entity_handoff_rules={
        "client_id": "propagate",
        "opportunity_id": "propagate",
        "contact_id": "propagate",
    },
    description="Onboarding/opp/deal workflow with expected failure in deal pipeline",
)

# Chain 6 (simple): Client management spot check
CHAIN_CLIENT_MANAGEMENT_SIMPLE_SUCCESS = WorkflowChain(
    chain_id="CHAIN-006A",
    workflow_sequence=["client_management"],
    success_pattern=[True],
    entity_handoff_rules={"client_id": "propagate"},
    description="Single-segment client management refresh (success)",
)

CHAIN_CLIENT_MANAGEMENT_SIMPLE_FAILURE = WorkflowChain(
    chain_id="CHAIN-006B",
    workflow_sequence=["client_management"],
    success_pattern=[False],
    entity_handoff_rules={"client_id": "propagate"},
    description="Single-segment client management refresh (expected failure)",
)

# Chain 7 (medium): Contact management followed by document workflow
CHAIN_CONTACT_DOC_MEDIUM_SUCCESS = WorkflowChain(
    chain_id="CHAIN-007A",
    workflow_sequence=["contact_management", "document_workflow"],
    success_pattern=[True, True],
    entity_handoff_rules={
        "client_id": "propagate",
        "contact_id": "propagate",
        "document_id": "propagate",
    },
    description="Contact creation and document workflow without failures",
)

CHAIN_CONTACT_DOC_MEDIUM_FAILURE = WorkflowChain(
    chain_id="CHAIN-007B",
    workflow_sequence=["contact_management", "document_workflow"],
    success_pattern=[True, False],
    entity_handoff_rules={
        "client_id": "propagate",
        "contact_id": "propagate",
        "document_id": "propagate",
    },
    description="Contact/document workflow with expected document upload failure",
)

# Chain 8: Quote remediation after generation
CHAIN_QUOTE_REMEDIATION_SUCCESS = WorkflowChain(
    chain_id="CHAIN-008A",
    workflow_sequence=["quote_generation", "quote_cleanup"],
    success_pattern=[True, True],
    entity_handoff_rules={
        "client_id": "propagate",
        "opportunity_id": "propagate",
        "quote_id": "propagate",
    },
    description="Generate quotes and clean up stale or duplicate artifacts",
)

CHAIN_QUOTE_REMEDIATION_FAILURE = WorkflowChain(
    chain_id="CHAIN-008B",
    workflow_sequence=["quote_generation", "quote_cleanup"],
    success_pattern=[True, False],
    entity_handoff_rules={
        "client_id": "propagate",
        "opportunity_id": "propagate",
        "quote_id": "propagate",
    },
    description="Quote lifecycle with expected failure during cleanup",
)

# Chain 9: Opportunity insights followed by contract review
CHAIN_SUMMARY_CONTRACT_SUCCESS = WorkflowChain(
    chain_id="CHAIN-009A",
    workflow_sequence=["opportunity_summary", "contract_review"],
    success_pattern=[True, True],
    entity_handoff_rules={
        "client_id": "propagate",
        "opportunity_id": "propagate",
        "contract_id": "propagate",
        "document_id": "propagate",
    },
    description="Summarize pipeline health then review associated contracts",
)

CHAIN_SUMMARY_CONTRACT_FAILURE = WorkflowChain(
    chain_id="CHAIN-009B",
    workflow_sequence=["opportunity_summary", "contract_review"],
    success_pattern=[True, False],
    entity_handoff_rules={
        "client_id": "propagate",
        "opportunity_id": "propagate",
        "contract_id": "propagate",
        "document_id": "propagate",
    },
    description="Contract review workflow with expected collateral upload failure",
)

# Chain 10: Opportunity clone and expansion
CHAIN_CLONE_EXPANSION_SUCCESS = WorkflowChain(
    chain_id="CHAIN-010A",
    workflow_sequence=["opportunity_management", "opportunity_clone"],
    success_pattern=[True, True],
    entity_handoff_rules={
        "client_id": "propagate",
        "opportunity_id": "propagate",
        "contact_id": "propagate",
        "document_id": "propagate",
    },
    description="Advance an opportunity then clone and enrich the duplicate",
)

CHAIN_CLONE_EXPANSION_FAILURE = WorkflowChain(
    chain_id="CHAIN-010B",
    workflow_sequence=["opportunity_management", "opportunity_clone"],
    success_pattern=[True, False],
    entity_handoff_rules={
        "client_id": "propagate",
        "opportunity_id": "propagate",
        "contact_id": "propagate",
        "document_id": "propagate",
    },
    description="Opportunity clone workflow with expected failure during expansion",
)

# Registry of all workflow chains
WORKFLOW_CHAINS: Dict[str, WorkflowChain] = {
    "onboarding_pipeline_contract_success": CHAIN_ONBOARDING_PIPELINE_CONTRACT_SUCCESS,
    "onboarding_pipeline_contract_failure": CHAIN_ONBOARDING_PIPELINE_CONTRACT_FAILURE,
    "client_opp_quote_success": CHAIN_CLIENT_OPP_QUOTE_SUCCESS,
    "client_opp_quote_failure": CHAIN_CLIENT_OPP_QUOTE_FAILURE,
    "contact_document_note_success": CHAIN_CONTACT_DOCUMENT_NOTE_SUCCESS,
    "contact_document_note_failure": CHAIN_CONTACT_DOCUMENT_NOTE_FAILURE,
    "search_quote_review_success": CHAIN_SEARCH_QUOTE_REVIEW_SUCCESS,
    "search_quote_review_failure": CHAIN_SEARCH_QUOTE_REVIEW_FAILURE,
    "onboarding_opp_deal_success": CHAIN_ONBOARDING_OPP_DEAL_SUCCESS,
    "onboarding_opp_deal_failure": CHAIN_ONBOARDING_OPP_DEAL_FAILURE,
    "client_management_chain_success": CHAIN_CLIENT_MANAGEMENT_SIMPLE_SUCCESS,
    "client_management_chain_failure": CHAIN_CLIENT_MANAGEMENT_SIMPLE_FAILURE,
    "contact_document_medium_success": CHAIN_CONTACT_DOC_MEDIUM_SUCCESS,
    "contact_document_medium_failure": CHAIN_CONTACT_DOC_MEDIUM_FAILURE,
    "quote_remediation_success": CHAIN_QUOTE_REMEDIATION_SUCCESS,
    "quote_remediation_failure": CHAIN_QUOTE_REMEDIATION_FAILURE,
    "summary_contract_success": CHAIN_SUMMARY_CONTRACT_SUCCESS,
    "summary_contract_failure": CHAIN_SUMMARY_CONTRACT_FAILURE,
    "clone_expansion_success": CHAIN_CLONE_EXPANSION_SUCCESS,
    "clone_expansion_failure": CHAIN_CLONE_EXPANSION_FAILURE,
}

CHAIN_ID_TO_KEY: Dict[str, str] = {
    chain.chain_id: key for key, chain in WORKFLOW_CHAINS.items()
}

CHAIN_ALIAS_MAP: Dict[str, List[str]] = {
    "onboarding_pipeline_contract": [
        "onboarding_pipeline_contract_success",
        "onboarding_pipeline_contract_failure",
    ],
    "client_opp_quote": [
        "client_opp_quote_success",
        "client_opp_quote_failure",
    ],
    "contact_document_note": [
        "contact_document_note_success",
        "contact_document_note_failure",
    ],
    "search_quote_review": [
        "search_quote_review_success",
        "search_quote_review_failure",
    ],
    "onboarding_opp_deal": [
        "onboarding_opp_deal_success",
        "onboarding_opp_deal_failure",
    ],
    "client_management_chain": [
        "client_management_chain_success",
        "client_management_chain_failure",
    ],
    "contact_document_medium": [
        "contact_document_medium_success",
        "contact_document_medium_failure",
    ],
    "quote_remediation": [
        "quote_remediation_success",
        "quote_remediation_failure",
    ],
    "summary_contract": [
        "summary_contract_success",
        "summary_contract_failure",
    ],
    "clone_expansion": [
        "clone_expansion_success",
        "clone_expansion_failure",
    ],
}


def get_workflow_chain(chain_id: str) -> Optional[WorkflowChain]:
    """Get workflow chain by ID."""
    return WORKFLOW_CHAINS.get(chain_id)


def expand_chain_ids(requested_keys: Sequence[str]) -> List[str]:
    """Expand legacy chain aliases into explicit success/failure variants."""
    expanded: List[str] = []
    for key in requested_keys:
        variants = CHAIN_ALIAS_MAP.get(key)
        if variants:
            expanded.extend(variants)
        elif key in WORKFLOW_CHAINS:
            expanded.append(key)
        elif key in CHAIN_ID_TO_KEY:
            expanded.append(CHAIN_ID_TO_KEY[key])
        else:
            raise KeyError(f"Unknown workflow chain key '{key}'.")

    seen: set[str] = set()
    ordered: List[str] = []
    for key in expanded:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered
CHAIN_SUCCESS_RATIO: float = 0.6
CHAIN_FAILURE_RATIO: float = 1.0 - CHAIN_SUCCESS_RATIO
CHAIN_RATIO_TOLERANCE: float = 0.02
