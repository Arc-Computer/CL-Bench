"""Tests for conversation schema (ConversationTurn, Conversation, ConversationResult)."""

import pytest
from src.conversation_schema import (
    Conversation,
    ConversationTurn,
    ConversationResult,
    ComplexityLevel,
    SuccessCriteria,
)
from src.evaluation.verification import VerificationMode

try:
    from src.scenario_generator import Scenario, ScenarioGenerator
    from src.entity_sampler import EntitySampler, SamplerConfig
    from src.crm_sandbox import MockCrmApi
    SCENARIO_SUPPORT = True
except ImportError:  # pragma: no cover - legacy modules removed
    Scenario = ScenarioGenerator = EntitySampler = SamplerConfig = MockCrmApi = None
    SCENARIO_SUPPORT = False


class TestConversationTurn:
    """Test ConversationTurn dataclass."""

    def test_create_simple_turn(self):
        """Test creating a simple turn."""
        turn = ConversationTurn(
            turn_id=1,
            user_utterance="Show me Acme Corp",
            expected_tool="client_search",
            expected_args={"name": "Acme Corp"},
        )
        assert turn.turn_id == 1
        assert turn.user_utterance == "Show me Acme Corp"
        assert turn.expected_tool == "client_search"
        assert turn.expect_success is True
        assert turn.references_previous_turns == []

    def test_create_turn_with_references(self):
        """Test creating a turn that references previous turns."""
        turn = ConversationTurn(
            turn_id=2,
            user_utterance="Create an opp for them",
            expected_tool="create_new_opportunity",
            expected_args={"client_id": "{{turn_1.client_id}}", "name": "Migration"},
            references_previous_turns=[1],
        )
        assert turn.references_previous_turns == [1]

    def test_create_failure_turn(self):
        """Test creating a failure scenario turn."""
        turn = ConversationTurn(
            turn_id=1,
            user_utterance="Create client with invalid email",
            expected_tool="create_new_client",
            expected_args={"email": "invalid-email"},
            expect_success=False,
            expected_error_substring="Validation error",
            failure_category="MALFORMED_EMAIL",
        )
        assert turn.expect_success is False
        assert turn.expected_error_substring == "Validation error"
        assert turn.failure_category == "MALFORMED_EMAIL"


class TestConversation:
    """Test Conversation dataclass."""

    def test_create_simple_conversation(self):
        """Test creating a simple 1-turn conversation."""
        turn = ConversationTurn(
            turn_id=1,
            user_utterance="Show me Acme Corp",
            expected_tool="client_search",
            expected_args={"name": "Acme Corp"},
        )
        conv = Conversation(
            conversation_id="CONV-001",
            workflow_category="Client Management",
            complexity_level="simple",
            turns=[turn],
        )
        assert conv.conversation_id == "CONV-001"
        assert len(conv.turns) == 1
        assert conv.complexity_level == "simple"

    def test_create_medium_conversation(self):
        """Test creating a medium 4-turn conversation."""
        turns = [
            ConversationTurn(
                turn_id=i,
                user_utterance=f"Turn {i}",
                expected_tool="client_search",
                expected_args={},
            )
            for i in range(1, 5)
        ]
        conv = Conversation(
            conversation_id="CONV-002",
            workflow_category="Client Onboarding",
            complexity_level="medium",
            turns=turns,
        )
        assert len(conv.turns) == 4
        assert conv.complexity_level == "medium"

    def test_create_complex_conversation(self):
        """Test creating a complex 7-turn conversation."""
        turns = [
            ConversationTurn(
                turn_id=i,
                user_utterance=f"Turn {i}",
                expected_tool="client_search",
                expected_args={},
            )
            for i in range(1, 8)
        ]
        conv = Conversation(
            conversation_id="CONV-003",
            workflow_category="Deal Pipeline",
            complexity_level="complex",
            turns=turns,
        )
        assert len(conv.turns) == 7
        assert conv.complexity_level == "complex"

    def test_conversation_validation_sequential_turn_ids(self):
        """Test that turn IDs must be sequential starting at 1."""
        # Valid: sequential IDs
        turns = [
            ConversationTurn(turn_id=1, user_utterance="Turn 1", expected_tool="client_search", expected_args={}),
            ConversationTurn(turn_id=2, user_utterance="Turn 2", expected_tool="modify_client", expected_args={}),
        ]
        conv = Conversation(
            conversation_id="CONV-004",
            workflow_category="Test",
            complexity_level="simple",
            turns=turns,
        )
        assert len(conv.turns) == 2

        # Invalid: non-sequential IDs
        bad_turns = [
            ConversationTurn(turn_id=1, user_utterance="Turn 1", expected_tool="client_search", expected_args={}),
            ConversationTurn(turn_id=3, user_utterance="Turn 3", expected_tool="modify_client", expected_args={}),
        ]
        with pytest.raises(ValueError, match="Turn IDs must be sequential"):
            Conversation(
                conversation_id="CONV-005",
                workflow_category="Test",
                complexity_level="simple",
                turns=bad_turns,
            )

    def test_conversation_validation_complexity_mismatch(self):
        """Test that complexity level must match turn count."""
        # Simple must have 1-3 turns
        turns = [
            ConversationTurn(turn_id=i, user_utterance=f"Turn {i}", expected_tool="client_search", expected_args={})
            for i in range(1, 5)
        ]
        with pytest.raises(ValueError, match="Simple conversations must have 1-3 turns"):
            Conversation(
                conversation_id="CONV-006",
                workflow_category="Test",
                complexity_level="simple",
                turns=turns,
            )

        # Medium must have 4-6 turns
        turns = [
            ConversationTurn(turn_id=i, user_utterance=f"Turn {i}", expected_tool="client_search", expected_args={})
            for i in range(1, 3)
        ]
        with pytest.raises(ValueError, match="Medium conversations must have 4-6 turns"):
            Conversation(
                conversation_id="CONV-007",
                workflow_category="Test",
                complexity_level="medium",
                turns=turns,
            )

        # Complex must have 7-10 turns
        turns = [
            ConversationTurn(turn_id=i, user_utterance=f"Turn {i}", expected_tool="client_search", expected_args={})
            for i in range(1, 6)
        ]
        with pytest.raises(ValueError, match="Complex conversations must have 7-10 turns"):
            Conversation(
                conversation_id="CONV-008",
                workflow_category="Test",
                complexity_level="complex",
                turns=turns,
            )

    def test_conversation_failure_turn_validation(self):
        """Test that failure_turn must be set when contains_failure=True."""
        turn = ConversationTurn(
            turn_id=1,
            user_utterance="Test",
            expected_tool="client_search",
            expected_args={},
            expect_success=False,
        )
        with pytest.raises(ValueError, match="failure_turn must be set when contains_failure=True"):
            Conversation(
                conversation_id="CONV-009",
                workflow_category="Test",
                complexity_level="simple",
                turns=[turn],
                contains_failure=True,
                failure_turn=None,
            )

        # Valid: failure_turn set
        conv = Conversation(
            conversation_id="CONV-010",
            workflow_category="Test",
            complexity_level="simple",
            turns=[turn],
            contains_failure=True,
            failure_turn=1,
        )
        assert conv.failure_turn == 1

    def test_conversation_failure_turn_range(self):
        """Test that failure_turn must be within valid range."""
        turn = ConversationTurn(
            turn_id=1,
            user_utterance="Test",
            expected_tool="client_search",
            expected_args={},
        )
        with pytest.raises(ValueError, match="failure_turn.*must be between"):
            Conversation(
                conversation_id="CONV-011",
                workflow_category="Test",
                complexity_level="simple",
                turns=[turn],
                contains_failure=True,
                failure_turn=2,  # Only 1 turn exists
            )

    def test_conversation_empty_turns(self):
        """Test that conversation must have at least one turn."""
        with pytest.raises(ValueError, match="Conversation must have at least one turn"):
            Conversation(
                conversation_id="CONV-012",
                workflow_category="Test",
                complexity_level="simple",
                turns=[],
            )


class TestConversationResult:
    """Test ConversationResult dataclass."""

    def test_create_success_result(self):
        """Test creating a successful conversation result."""
        result = ConversationResult(
            conversation_id="CONV-001",
            overall_success=True,
            turns_executed=3,
            reward_signal=0.95,
        )
        assert result.overall_success is True
        assert result.turns_executed == 3
        assert result.failed_at_turn is None
        assert result.reward_signal == 0.95

    def test_create_failure_result(self):
        """Test creating a failed conversation result."""
        result = ConversationResult(
            conversation_id="CONV-002",
            overall_success=False,
            turns_executed=2,
            failed_at_turn=3,
            error_message="Tool execution failed",
        )
        assert result.overall_success is False
        assert result.turns_executed == 2
        assert result.failed_at_turn == 3
        assert result.error_message == "Tool execution failed"

    def test_result_validation_success_has_no_failure_turn(self):
        """Test that successful results cannot have failed_at_turn."""
        with pytest.raises(ValueError, match="failed_at_turn must be None when overall_success=True"):
            ConversationResult(
                conversation_id="CONV-003",
                overall_success=True,
                turns_executed=3,
                failed_at_turn=2,
            )

    def test_result_validation_failure_has_failure_turn(self):
        """Test that failed results must have failed_at_turn."""
        with pytest.raises(ValueError, match="failed_at_turn must be set when overall_success=False"):
            ConversationResult(
                conversation_id="CONV-004",
                overall_success=False,
                turns_executed=2,
                failed_at_turn=None,
            )

    def test_result_validation_reward_signal_range(self):
        """Test that reward_signal must be between 0.0 and 1.0."""
        # Valid: 0.0
        result = ConversationResult(
            conversation_id="CONV-005",
            overall_success=True,
            turns_executed=1,
            reward_signal=0.0,
        )
        assert result.reward_signal == 0.0

        # Valid: 1.0
        result = ConversationResult(
            conversation_id="CONV-006",
            overall_success=True,
            turns_executed=1,
            reward_signal=1.0,
        )
        assert result.reward_signal == 1.0

        # Invalid: < 0.0
        with pytest.raises(ValueError, match="reward_signal must be between 0.0 and 1.0"):
            ConversationResult(
                conversation_id="CONV-007",
                overall_success=True,
                turns_executed=1,
                reward_signal=-0.1,
            )

        # Invalid: > 1.0
        with pytest.raises(ValueError, match="reward_signal must be between 0.0 and 1.0"):
            ConversationResult(
                conversation_id="CONV-008",
                overall_success=True,
                turns_executed=1,
                reward_signal=1.1,
            )

    def test_result_validation_turns_executed_vs_failed_at(self):
        """Test that turns_executed must be < failed_at_turn."""
        with pytest.raises(ValueError, match="turns_executed.*must be < failed_at_turn"):
            ConversationResult(
                conversation_id="CONV-009",
                overall_success=False,
                turns_executed=3,
                failed_at_turn=3,  # Must be > turns_executed
            )


class TestBackwardCompatibility:
    """Test backward compatibility with Scenario objects."""

    def test_scenario_to_conversation_single_tool(self):
        """Test converting single-tool scenario to conversation."""
        if not SCENARIO_SUPPORT:
            pytest.skip("Scenario generator not available in current codebase.")

        scenario = Scenario(
            scenario_id="SC-001",
            task="create_new_client",
            intent="Client Management",
            utterance="Create a new client",
            expected_tool="create_new_client",
            setup_entities={"client_id": "test-123"},
            expected_args={"name": "Acme Corp", "email": "test@example.com", "status": "Active"},
            expect_success=True,
            expected_error_substring=None,
            failure_category=None,
            verification_mode=VerificationMode.DATABASE,
        )

        api = MockCrmApi()
        sampler = EntitySampler(api, SamplerConfig(seed=42))
        generator = ScenarioGenerator(api, sampler)

        conv = generator.to_conversation(scenario)

        assert conv.conversation_id == "CONV-SC-001"
        assert len(conv.turns) == 1
        assert conv.turns[0].turn_id == 1
        assert conv.turns[0].user_utterance == "Create a new client"
        assert conv.turns[0].expected_tool == "create_new_client"
        assert conv.complexity_level == "simple"
        assert conv.initial_entities == {"client_id": "test-123"}

    def test_scenario_to_conversation_multi_tool(self):
        """Test converting multi-tool scenario to conversation."""
        if not SCENARIO_SUPPORT:
            pytest.skip("Scenario generator not available in current codebase.")

        scenario = Scenario(
            scenario_id="SC-002",
            task="create_contact_and_opp",
            intent="Client Onboarding",
            utterance="Create contact and opportunity",
            expected_tool=["create_new_contact", "create_new_opportunity"],
            setup_entities={"client_id": "test-123"},
            expected_args=[
                {"client_id": "test-123", "first_name": "John", "last_name": "Doe"},
                {"client_id": "test-123", "name": "Migration", "amount": 50000, "stage": "Prospecting"},
            ],
            expect_success=True,
            expected_error_substring=None,
            failure_category=None,
            verification_mode=VerificationMode.DATABASE,
        )

        api = MockCrmApi()
        sampler = EntitySampler(api, SamplerConfig(seed=42))
        generator = ScenarioGenerator(api, sampler)

        conv = generator.to_conversation(scenario)

        assert len(conv.turns) == 2
        assert conv.turns[0].turn_id == 1
        assert conv.turns[0].user_utterance == "Create contact and opportunity"
        assert conv.turns[0].expected_tool == "create_new_contact"
        assert conv.turns[1].turn_id == 2
        assert conv.turns[1].expected_tool == "create_new_opportunity"
        assert conv.complexity_level == "simple"

    def test_scenario_to_conversation_failure(self):
        """Test converting failure scenario to conversation."""
        if not SCENARIO_SUPPORT:
            pytest.skip("Scenario generator not available in current codebase.")

        scenario = Scenario(
            scenario_id="SC-003",
            task="create_new_client",
            intent="Client Management",
            utterance="Create client with invalid email",
            expected_tool="create_new_client",
            setup_entities={},
            expected_args={"email": "invalid-email"},
            expect_success=False,
            expected_error_substring="Validation error",
            failure_category="MALFORMED_EMAIL",
            verification_mode=VerificationMode.DATABASE,
        )

        api = MockCrmApi()
        sampler = EntitySampler(api, SamplerConfig(seed=42))
        generator = ScenarioGenerator(api, sampler)

        conv = generator.to_conversation(scenario)

        assert conv.contains_failure is True
        assert conv.failure_turn == 1
        assert conv.turns[0].expect_success is False
        assert conv.turns[0].expected_error_substring == "Validation error"
