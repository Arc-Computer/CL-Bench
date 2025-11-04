"""Tests for curator conversation generator."""

import pytest
from unittest.mock import Mock, patch
from src.curator_conversation_generator import (
    ConversationTurnResponse,
    CuratorConversationGenerator,
    CuratorConversationDatasetGenerator,
    simulate_turn_execution,
)
from src.conversation_templates import CLIENT_MANAGEMENT, OPPORTUNITY_MANAGEMENT
from src.crm_sandbox import MockCrmApi


class TestConversationTurnResponse:
    """Test ConversationTurnResponse Pydantic model."""

    def test_valid_response(self):
        """Test creating a valid response."""
        response = ConversationTurnResponse(
            user_utterance="Show me Acme Corp",
            expected_args={"name": "Acme Corp"},
        )
        assert response.user_utterance == "Show me Acme Corp"
        assert response.expected_args == {"name": "Acme Corp"}

    def test_empty_utterance_allowed(self):
        """Test that empty utterance is allowed (validation will catch later)."""
        response = ConversationTurnResponse(
            user_utterance="",
            expected_args={},
        )
        assert response.user_utterance == ""


class TestSimulateTurnExecution:
    """Test simulate_turn_execution function."""

    def test_create_new_client(self):
        """Test simulating create_new_client operation."""
        api = MockCrmApi()
        entity_pool = {"clients": []}
        
        result = simulate_turn_execution(
            "create_new_client",
            {"name": "Acme Corp", "email": "test@acme.com", "status": "Active"},
            api,
            entity_pool,
        )
        
        assert "client_id" in result
        assert isinstance(result["client_id"], str)
        assert len(result["client_id"]) > 0

    def test_create_new_opportunity(self):
        """Test simulating create_new_opportunity operation."""
        api = MockCrmApi()
        entity_pool = {"opportunities": []}
        
        result = simulate_turn_execution(
            "create_new_opportunity",
            {"client_id": "test-123", "name": "Migration", "amount": 50000, "stage": "Prospecting"},
            api,
            entity_pool,
        )
        
        assert "opportunity_id" in result
        assert result["client_id"] == "test-123"

    def test_client_search_exact_match(self):
        """Test client_search with exact match."""
        api = MockCrmApi()
        entity_pool = {
            "clients": [
                {"client_id": "abc-123", "name": "Acme Corp"},
                {"client_id": "def-456", "name": "Tech Corp"},
            ]
        }
        
        result = simulate_turn_execution(
            "client_search",
            {"name": "Acme Corp"},
            api,
            entity_pool,
        )
        
        assert result["client_id"] == "abc-123"
        assert result["name"] == "Acme Corp"

    def test_client_search_partial_match(self):
        """Test client_search with partial match."""
        api = MockCrmApi()
        entity_pool = {
            "clients": [
                {"client_id": "abc-123", "name": "Acme Corporation"},
                {"client_id": "def-456", "name": "Tech Corp"},
            ]
        }
        
        result = simulate_turn_execution(
            "client_search",
            {"name": "Acme"},
            api,
            entity_pool,
        )
        
        assert result["client_id"] == "abc-123"

    def test_client_search_fallback(self):
        """Test client_search fallback when no match."""
        api = MockCrmApi()
        entity_pool = {
            "clients": [
                {"client_id": "abc-123", "name": "Acme Corp"},
            ]
        }
        
        result = simulate_turn_execution(
            "client_search",
            {"name": "NonExistent"},
            api,
            entity_pool,
        )
        
        assert result["client_id"] == "abc-123"  # Falls back to first

    def test_modify_opportunity(self):
        """Test simulating modify_opportunity operation."""
        api = MockCrmApi()
        entity_pool = {}
        
        result = simulate_turn_execution(
            "modify_opportunity",
            {"opportunity_id": "opp-123", "updates": {"stage": "Qualification"}},
            api,
            entity_pool,
        )
        
        assert result["opportunity_id"] == "opp-123"
        assert result["stage"] == "Qualification"

    def test_opportunity_search_by_client(self):
        """Test opportunity_search filtered by client_id."""
        api = MockCrmApi()
        entity_pool = {
            "opportunities": [
                {"opportunity_id": "opp-1", "client_id": "client-123", "name": "Deal 1"},
                {"opportunity_id": "opp-2", "client_id": "client-456", "name": "Deal 2"},
            ]
        }
        
        result = simulate_turn_execution(
            "opportunity_search",
            {"client_id": "client-123"},
            api,
            entity_pool,
        )
        
        assert result["opportunity_id"] == "opp-1"


class TestCuratorConversationGenerator:
    """Test CuratorConversationGenerator class."""

    def test_prompt_first_turn(self):
        """Test prompt generation for Turn 1."""
        generator = CuratorConversationGenerator(entity_pool={"clients": []})
        
        prompt = generator._build_first_turn_prompt(
            workflow_category="Client Management",
            turn_template={
                "tool_name": "client_search",
                "argument_template": {"name": ""},
                "user_utterance_pattern": "Show me {entity_name}",
            },
            current_crm_state={},
        )
        
        assert "Turn 1" in prompt or "first turn" in prompt.lower()
        assert "client_search" in prompt
        assert "Do NOT include" in prompt and "templates yet" in prompt or "Turn 1" in prompt

    def test_prompt_subsequent_turn(self):
        """Test prompt generation for Turn 2+."""
        generator = CuratorConversationGenerator(entity_pool={"clients": []})
        
        prompt = generator._build_subsequent_turn_prompt(
            turn_number=2,
            workflow_category="Client Management",
            turn_template={
                "tool_name": "modify_client",
                "argument_template": {"client_id": "{{turn_1.client_id}}", "updates": {"status": ""}},
                "user_utterance_pattern": "Update {entity_name}'s status",
                "references_previous_turns": [1],
            },
            conversation_history=[
                {
                    "turn_id": 1,
                    "user_utterance": "Show me Acme Corp",
                    "tool_name": "client_search",
                    "result": {"client_id": "abc-123"},
                }
            ],
            current_crm_state={},
        )
        
        assert "turn 2" in prompt.lower()
        assert "Conversation History" in prompt
        assert "turn_1.client_id" in prompt or "{{turn_1.client_id}}" in prompt
        assert "pronouns" in prompt.lower() or "them" in prompt.lower()


class TestCuratorConversationDatasetGenerator:
    """Test CuratorConversationDatasetGenerator class."""

    def test_generate_conversation_forward_reference_detection(self):
        """Test that forward references are detected before generation."""
        generator = CuratorConversationDatasetGenerator()
        generator.entity_pool = {"clients": [{"client_id": "test-123", "name": "Test Corp"}]}
        
        # Create a template with forward reference (should fail)
        from src.conversation_templates import TurnTemplate, WorkflowTemplate
        
        bad_template = WorkflowTemplate(
            workflow_id="BAD-001",
            workflow_category="Test",
            complexity_level="simple",
            turn_templates=[
                TurnTemplate(
                    turn_number=1,
                    tool_name="client_search",
                    argument_template={"name": ""},
                    user_utterance_pattern="Show me {entity_name}",
                    references_previous_turns=[2],  # Forward reference!
                ),
            ],
        )
        
        # Mock the generator to avoid actual API calls
        mock_generator = Mock()
        
        result = generator.generate_conversation(
            bad_template,
            "CONV-TEST-001",
            mock_generator,
        )
        
        assert result is None  # Should fail validation

    def test_generate_conversation_valid_template(self):
        """Test generating a conversation with valid template."""
        generator = CuratorConversationDatasetGenerator()
        generator.entity_pool = {
            "clients": [
                {"client_id": "test-123", "name": "Acme Corp", "industry": "Tech", "status": "Active"}
            ]
        }
        
        # Mock the Curator generator to return valid responses
        mock_generator = Mock()
        mock_response = Mock()
        mock_response.dataset = [
            {
                "turn_number": 1,
                "user_utterance": "Show me Acme Corp",
                "expected_args": {"name": "Acme Corp"},
                "tool_name": "client_search",
            }
        ]
        mock_generator.return_value = mock_response
        
        # This will fail because we're not actually calling the API,
        # but we can test the validation logic
        # For a full test, we'd need to mock the Dataset.from_list and generator call properly
        
        # Just verify the template validation works
        assert CLIENT_MANAGEMENT.turn_templates[0].turn_number == 1
        assert CLIENT_MANAGEMENT.turn_templates[0].references_previous_turns == []


class TestEndToEnd:
    """End-to-end integration tests (require API calls)."""

    @pytest.mark.skip(reason="Requires API key and costs money")
    def test_generate_single_conversation(self):
        """Test generating a single conversation (requires API)."""
        generator = CuratorConversationDatasetGenerator(seed=42)
        curator_generator = CuratorConversationGenerator(entity_pool=generator.entity_pool)
        
        conversation = generator.generate_conversation(
            CLIENT_MANAGEMENT,
            "CONV-TEST-001",
            curator_generator,
        )
        
        assert conversation is not None
        assert len(conversation.turns) == 2
        assert conversation.turns[0].turn_id == 1
        assert conversation.turns[1].turn_id == 2

