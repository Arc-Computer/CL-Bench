"""Workflow templates for multi-turn conversation generation.

Defines 8 common CRM workflow patterns used to generate conversations with
realistic multi-turn interactions. Each template specifies:
- Complexity level (simple, medium, complex)
- Turn sequence (list of turn templates)
- Entity dependencies (what entities are needed/created)
- Cross-turn references (how turns reference previous entities)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


@dataclass
class TurnTemplate:
    """Template for a single turn in a workflow.
    
    Attributes:
        turn_number: Sequential turn number (1-indexed)
        tool_name: Expected tool to be called
        argument_template: Template for expected arguments (may contain {{turn_N.field}})
        user_utterance_pattern: Pattern for generating natural language utterance
        references_previous_turns: List of turn numbers this turn references
    """
    turn_number: int
    tool_name: str
    argument_template: Dict[str, Any]
    user_utterance_pattern: str
    references_previous_turns: List[int] = None
    
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
            argument_template={"opportunity_id": "{{turn_1.opportunity_id}}"},
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

