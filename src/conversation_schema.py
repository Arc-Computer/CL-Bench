"""Conversation schema for multi-turn CRM benchmark evaluation.

This module defines the core data structures for multi-turn conversations:
- ConversationTurn: Single turn in a conversation with user utterance and expected tool call
- Conversation: Complete conversation with multiple turns, initial state, and success criteria
- ConversationResult: Evaluation result from executing a conversation

Key features:
- Template references: {{turn_N.field}} syntax for cross-turn entity references
- Complexity tiers: Simple (1-3 turns), Medium (4-6 turns), Complex (7-10 turns)
- Backward compatibility: Single-turn conversations compatible with existing Scenario objects
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Union
from enum import Enum

from src.evaluation.verification import VerificationMode


class ComplexityLevel(str, Enum):
    """Complexity levels for conversations."""
    SIMPLE = "simple"  # 1-3 turns
    MEDIUM = "medium"  # 4-6 turns
    COMPLEX = "complex"  # 7-10 turns


class SuccessCriteria(str, Enum):
    """Success criteria for conversation evaluation."""
    ALL_TURNS = "all_turns"  # All turns must succeed
    FINAL_STATE = "final_state"  # Only final state matters
    BOTH = "both"  # Both all turns and final state must pass


@dataclass
class ExpectedResponse:
    """Ground-truth assistant response associated with a turn."""

    text: str
    evaluation: Literal["structured", "judge"] = "structured"
    answers: List[str] = field(default_factory=list)
    requires_judge: bool = False

    def __post_init__(self) -> None:
        self.text = (self.text or "").strip()
        if not self.evaluation:
            self.evaluation = "structured"
        self.evaluation = self.evaluation.lower()
        if self.requires_judge:
            self.evaluation = "judge"
        if self.evaluation == "structured" and not self.answers and self.text:
            self.answers.append(self.text)
        self.answers = [str(answer).strip() for answer in self.answers if str(answer).strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "text": self.text,
            "evaluation": self.evaluation,
            "answers": list(self.answers),
            "requires_judge": self.requires_judge,
        }

    @classmethod
    def from_payload(cls, payload: Optional[Union["ExpectedResponse", Mapping[str, Any]]]) -> Optional["ExpectedResponse"]:
        if payload is None:
            return None
        if isinstance(payload, ExpectedResponse):
            return payload
        if not isinstance(payload, dict):
            raise TypeError(f"ExpectedResponse payload must be dict, got {type(payload)}")
        return cls(
            text=str(payload.get("text", "")),
            evaluation=str(payload.get("evaluation", "structured")),
            answers=list(payload.get("answers") or []),
            requires_judge=bool(payload.get("requires_judge", False)),
        )


@dataclass
class ConversationTurn:
    """Single turn in a conversation.
    
    Attributes:
        turn_id: Sequential turn number (1-indexed)
        user_utterance: Natural language user input for this turn
        expected_tool: Tool name expected to be called
        expected_args: Dictionary of expected arguments (may contain {{turn_N.field}} templates)
        references_previous_turns: List of turn IDs this turn references (for validation)
        expect_success: Whether this turn is expected to succeed (True) or fail (False)
        expected_error_substring: If expect_success=False, substring to match in error message
        failure_category: Category of failure if this is a failure scenario
        expected_response: Structured description of the agent's natural-language reply
    """
    turn_id: int
    user_utterance: str
    expected_tool: str
    expected_args: Dict[str, Any]
    references_previous_turns: List[int] = field(default_factory=list)
    expect_success: bool = True
    expected_error_substring: Optional[str] = None
    failure_category: Optional[str] = None
    expected_response: Optional[ExpectedResponse] = None

    def __post_init__(self) -> None:
        if self.expected_response is not None and not isinstance(self.expected_response, ExpectedResponse):
            self.expected_response = ExpectedResponse.from_payload(self.expected_response)


@dataclass
class Conversation:
    """Complete multi-turn conversation.
    
    Attributes:
        conversation_id: Unique identifier for this conversation
        workflow_category: Category of workflow (e.g., "Client Management", "Opportunity Pipeline")
        complexity_level: Complexity tier (simple, medium, complex)
        turns: List of ConversationTurn objects in order
        initial_entities: Dictionary of entities that exist before conversation starts
        final_expected_state: Expected state after all turns complete (for validation)
        success_criteria: How to evaluate success (all_turns, final_state, or both)
        contains_failure: Whether this conversation contains a failure scenario
        failure_turn: Turn number where failure is expected (if contains_failure=True)
        verification_mode: How to verify conversation success
        chain_id: Optional chain identifier if this conversation is part of a workflow chain
        segment_number: Optional segment number within a chain (1-indexed)
        segment_boundaries: Optional list of turn numbers where segments end (for chained conversations)
        expected_outcome: Optional expected outcome description for the conversation
        cumulative_context: Optional dictionary of context accumulated from previous segments (for chains)
    """
    conversation_id: str
    workflow_category: str
    complexity_level: Literal["simple", "medium", "complex"]
    turns: List[ConversationTurn]
    initial_entities: Dict[str, Any] = field(default_factory=dict)
    final_expected_state: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Literal["all_turns", "final_state", "both"] = "all_turns"
    contains_failure: bool = False
    failure_turn: Optional[int] = None
    verification_mode: VerificationMode = VerificationMode.DATABASE
    chain_id: Optional[str] = None
    segment_number: Optional[int] = None
    segment_boundaries: Optional[List[int]] = None
    expected_outcome: Optional[str] = None
    cumulative_context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate conversation structure."""
        if not self.turns:
            raise ValueError("Conversation must have at least one turn")
        
        # Validate turn IDs are sequential and start at 1
        turn_ids = [turn.turn_id for turn in self.turns]
        expected_ids = list(range(1, len(self.turns) + 1))
        if turn_ids != expected_ids:
            raise ValueError(
                f"Turn IDs must be sequential starting at 1. Got {turn_ids}, expected {expected_ids}"
            )
        
        # Validate complexity level matches turn count
        turn_count = len(self.turns)
        is_chain = self.chain_id is not None

        if self.complexity_level == "simple" and not (1 <= turn_count <= 3):
            raise ValueError(
                f"Simple conversations must have 1-3 turns, got {turn_count}"
            )
        elif self.complexity_level == "medium" and not (4 <= turn_count <= 6):
            raise ValueError(
                f"Medium conversations must have 4-6 turns, got {turn_count}"
            )
        elif self.complexity_level == "complex":
            if not is_chain and not (7 <= turn_count <= 10):
                raise ValueError(
                    f"Complex conversations must have 7-10 turns, got {turn_count}"
                )
        
        # Validate failure turn if contains_failure=True
        if self.contains_failure and self.failure_turn is None:
            raise ValueError("failure_turn must be set when contains_failure=True")
        if self.failure_turn is not None and not (1 <= self.failure_turn <= turn_count):
            raise ValueError(
                f"failure_turn ({self.failure_turn}) must be between 1 and {turn_count}"
            )


@dataclass
class ConversationResult:
    """Result from executing a conversation.
    
    Attributes:
        conversation_id: ID of the conversation that was executed
        overall_success: Whether the conversation succeeded overall
        turns_executed: Number of turns successfully executed before failure (or all if success)
        failed_at_turn: Turn number where execution failed (None if successful)
        per_turn_results: List of results for each turn executed
        reward_signal: Reward signal for Atlas (0.0-1.0)
        error_message: Error message if conversation failed
        metadata: Additional metadata about the execution
        per_segment_results: Optional list of results for each segment (for chained conversations)
        chain_success: Optional overall chain success status (for chained conversations)
    """
    conversation_id: str
    overall_success: bool
    turns_executed: int
    failed_at_turn: Optional[int] = None
    per_turn_results: List[Dict[str, Any]] = field(default_factory=list)
    reward_signal: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    per_segment_results: List[Dict[str, Any]] = field(default_factory=list)
    chain_success: Optional[bool] = None

    def __post_init__(self):
        """Validate result structure."""
        if self.overall_success and self.failed_at_turn is not None:
            raise ValueError("failed_at_turn must be None when overall_success=True")
        if not self.overall_success and self.failed_at_turn is None:
            raise ValueError("failed_at_turn must be set when overall_success=False")
        
        if self.turns_executed < 0:
            raise ValueError(f"turns_executed must be >= 0, got {self.turns_executed}")
        
        if not (0.0 <= self.reward_signal <= 1.0):
            raise ValueError(
                f"reward_signal must be between 0.0 and 1.0, got {self.reward_signal}"
            )
        
        if self.failed_at_turn is not None and self.turns_executed < self.failed_at_turn:
            raise ValueError(
                f"turns_executed ({self.turns_executed}) must be >= failed_at_turn ({self.failed_at_turn}) "
                "when continuing through all turns"
            )
