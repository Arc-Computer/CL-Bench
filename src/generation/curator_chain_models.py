"""Structured output models for Curator chain generation.

This module defines Pydantic models following Bespoke Curator patterns for:
- Turn utterance generation
- Chain segment summaries
- Scenario selection
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class TurnUtterance(BaseModel):
    """Single user utterance for a conversation turn."""

    user_utterance: str = Field(
        description="Natural language user request for the CRM assistant. "
        "Use concise, conversational language with pronouns and implicit references "
        "that reference previous turns (e.g., 'Create an opp for them')."
    )


class TurnUtteranceResponse(BaseModel):
    """Structured output containing multiple user utterances."""

    utterances: List[TurnUtterance] = Field(
        description="List of user utterances, one per turn in the conversation segment."
    )


class ChainSegmentSummary(BaseModel):
    """Summary of a chain segment for context propagation."""

    segment_id: int = Field(description="Segment number (1-indexed)")
    workflow_category: str = Field(description="Workflow category name")
    expected_outcome: str = Field(description="Expected outcome of this segment")
    cumulative_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context accumulated from previous segments (entity IDs, state, etc.)",
    )


class ScenarioSelection(BaseModel):
    """Single scenario selection for a turn."""

    scenario_id: str = Field(description="Scenario ID to use for this turn")
    tool_name: str = Field(description="Tool name this scenario targets")
    turn_number: int = Field(description="Turn number within the segment")


class ScenarioSelectionResponse(BaseModel):
    """Structured output containing scenario selections for a segment."""

    selections: List[ScenarioSelection] = Field(
        description="List of scenario selections, one per turn in the segment."
    )


class SegmentContext(BaseModel):
    """Context passed between chain segments."""

    segment_number: int = Field(description="Segment number (1-indexed)")
    entities_created: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Entities created in this segment, keyed by entity type",
    )
    entities_referenced: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Entities referenced in this segment, keyed by entity type",
    )
    cumulative_entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="All entities available after this segment (cumulative)",
    )

