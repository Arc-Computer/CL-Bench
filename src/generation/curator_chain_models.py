"""Structured output models for Curator chain generation.

This module defines Pydantic models following Bespoke Curator patterns for:
- Scenario selection across multi-turn segments
- Turn-level utterance generation
- Segment-level summaries for handoff context
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TurnMetadata(BaseModel):
    """Curator hint describing the intent of a turn inside a segment."""

    turn_number: int = Field(description="Turn number within the segment (1-indexed).")
    tool_name: str = Field(description="Tool the assistant will execute on this turn.")
    desired_outcome: Literal["success", "failure"] = Field(
        description="Whether this turn should succeed or surface a controlled failure."
    )
    stage_hint: Optional[str] = Field(
        default=None,
        description="Optional pipeline or workflow stage hint (e.g., 'Proposal').",
    )
    persona_hint: Optional[str] = Field(
        default=None,
        description="Optional description of the end-user persona/tone.",
    )
    handoff_dependencies: List[str] = Field(
        default_factory=list,
        description="Entity identifiers required from earlier turns or segments "
        "(e.g., ['client_id', 'opportunity_id']).",
    )


class ScenarioSelection(BaseModel):
    """Single scenario selection for a turn."""

    scenario_id: str = Field(description="Scenario ID to use for this turn.")
    tool_name: str = Field(description="Tool name this scenario targets.")
    turn_number: int = Field(description="Turn number within the segment.")
    justification: str = Field(
        description="Short rationale tying the scenario to the turn metadata."
    )
    handoff_actions: Dict[str, Literal["propagate", "create", "reference", "none"]] = Field(
        default_factory=dict,
        description="Mapping of entity types (client_id, opportunity_id, etc.) to how this turn handles them.",
    )


class ScenarioSelectionResponse(BaseModel):
    """Structured output containing scenario selections for a segment."""

    selections: List[ScenarioSelection] = Field(
        description="List of scenario selections, one per turn in the segment."
    )


class TurnUtterance(BaseModel):
    """Single user utterance for a conversation turn."""

    turn_number: int = Field(description="Turn number within the segment (1-indexed).")
    user_utterance: str = Field(
        description="Natural language user request for the CRM assistant. "
        "Use concise, conversational language with pronouns and implicit references."
    )
    persona_hint: Optional[str] = Field(
        default=None,
        description="Persona or tone reflected in this utterance (e.g., 'Sales Manager, decisive').",
    )
    stage_focus: Optional[str] = Field(
        default=None,
        description="Pipeline stage or workflow focus emphasized by the utterance.",
    )
    referenced_entities: List[str] = Field(
        default_factory=list,
        description="List of entity identifiers referenced (e.g., ['client_id', 'quote_id']).",
    )
    handoff_summary: Optional[str] = Field(
        default=None,
        description="Summary of how the utterance references prior segment state or sets up next handoff.",
    )


class TurnUtteranceResponse(BaseModel):
    """Structured output containing multiple user utterances."""

    utterances: List[TurnUtterance] = Field(
        description="List of user utterances, one per turn in the conversation segment."
    )


class ChainSegmentSummary(BaseModel):
    """Summary of a chain segment for context propagation."""

    segment_id: int = Field(description="Segment number (1-indexed).")
    workflow_category: str = Field(description="Workflow category name.")
    expected_outcome: Literal["success", "failure"] = Field(
        description="Expected outcome of this segment."
    )
    cumulative_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context accumulated from previous segments (entity IDs, state, etc.).",
    )
    handoff_trace: Dict[str, str] = Field(
        default_factory=dict,
        description="Entity identifiers made available to later segments (e.g., {'client_id': 'abc-123'}).",
    )


class SegmentContext(BaseModel):
    """Context passed between chain segments."""

    segment_number: int = Field(description="Segment number (1-indexed).")
    expected_outcome: Literal["success", "failure"] = Field(
        description="Expected outcome of the segment."
    )
    entities_created: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Entities created in this segment, keyed by entity type.",
    )
    entities_referenced: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Entities referenced in this segment, keyed by entity type.",
    )
    cumulative_entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="All entities available after this segment (cumulative).",
    )
