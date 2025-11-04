"""Tests for reference resolver (template resolution and validation)."""

import pytest
from src.reference_resolver import (
    resolve_template,
    extract_template_references,
    validate_template_references,
    find_circular_dependencies,
    TemplateResolutionError,
)


class TestExtractTemplateReferences:
    """Test extracting template references from dictionaries."""

    def test_extract_simple_reference(self):
        """Test extracting a single template reference."""
        args = {"client_id": "{{turn_1.client_id}}", "name": "Acme Corp"}
        references = extract_template_references(args)
        assert references == [(1, "client_id")]

    def test_extract_multiple_references(self):
        """Test extracting multiple template references."""
        args = {
            "client_id": "{{turn_1.client_id}}",
            "opportunity_id": "{{turn_2.opportunity_id}}",
            "amount": 50000,
        }
        references = extract_template_references(args)
        assert set(references) == {(1, "client_id"), (2, "opportunity_id")}

    def test_extract_nested_references(self):
        """Test extracting references from nested dictionaries."""
        args = {
            "updates": {
                "stage": "{{turn_2.stage}}",
                "probability": 75,
            },
            "opportunity_id": "{{turn_1.opportunity_id}}",
        }
        references = extract_template_references(args)
        assert set(references) == {(1, "opportunity_id"), (2, "stage")}

    def test_extract_no_references(self):
        """Test extracting from dictionary with no templates."""
        args = {"name": "Acme Corp", "amount": 50000}
        references = extract_template_references(args)
        assert references == []

    def test_extract_from_list(self):
        """Test extracting references from list values."""
        args = {
            "ids": ["{{turn_1.client_id}}", "{{turn_2.contact_id}}"],
            "name": "Test",
        }
        references = extract_template_references(args)
        assert set(references) == {(1, "client_id"), (2, "contact_id")}


class TestResolveTemplate:
    """Test resolving template references to actual values."""

    def test_resolve_simple_reference(self):
        """Test resolving a single template reference."""
        args = {"client_id": "{{turn_1.client_id}}", "name": "Acme Corp"}
        previous_turns = {1: {"client_id": "abc-123"}}
        resolved = resolve_template(args, previous_turns, turn_number=2)
        assert resolved == {"client_id": "abc-123", "name": "Acme Corp"}

    def test_resolve_multiple_references(self):
        """Test resolving multiple template references."""
        args = {
            "client_id": "{{turn_1.client_id}}",
            "opportunity_id": "{{turn_2.opportunity_id}}",
            "amount": 50000,
        }
        previous_turns = {
            1: {"client_id": "abc-123"},
            2: {"opportunity_id": "def-456"},
        }
        resolved = resolve_template(args, previous_turns, turn_number=3)
        assert resolved == {
            "client_id": "abc-123",
            "opportunity_id": "def-456",
            "amount": 50000,
        }

    def test_resolve_nested_references(self):
        """Test resolving references in nested dictionaries."""
        args = {
            "updates": {
                "stage": "{{turn_2.stage}}",
                "probability": 75,
            },
            "opportunity_id": "{{turn_1.opportunity_id}}",
        }
        previous_turns = {
            1: {"opportunity_id": "opp-123"},
            2: {"stage": "Negotiation"},
        }
        resolved = resolve_template(args, previous_turns, turn_number=3)
        assert resolved == {
            "updates": {
                "stage": "Negotiation",
                "probability": 75,
            },
            "opportunity_id": "opp-123",
        }

    def test_resolve_non_string_values(self):
        """Test that non-string values are left unchanged."""
        args = {
            "client_id": "{{turn_1.client_id}}",
            "amount": 50000,
            "probability": 75,
            "active": True,
        }
        previous_turns = {1: {"client_id": "abc-123"}}
        resolved = resolve_template(args, previous_turns, turn_number=2)
        assert resolved == {
            "client_id": "abc-123",
            "amount": 50000,
            "probability": 75,
            "active": True,
        }

    def test_resolve_strict_mode_missing_turn(self):
        """Test that strict mode raises error for missing turn."""
        args = {"client_id": "{{turn_1.client_id}}"}
        previous_turns = {}  # Turn 1 doesn't exist
        with pytest.raises(TemplateResolutionError, match="references turn 1 which doesn't exist"):
            resolve_template(args, previous_turns, turn_number=2, strict=True)

    def test_resolve_strict_mode_missing_field(self):
        """Test that strict mode raises error for missing field."""
        args = {"client_id": "{{turn_1.client_id}}"}
        previous_turns = {1: {"name": "Acme Corp"}}  # client_id doesn't exist
        with pytest.raises(TemplateResolutionError, match="field 'client_id' which doesn't exist"):
            resolve_template(args, previous_turns, turn_number=2, strict=True)

    def test_resolve_strict_mode_forward_reference(self):
        """Test that strict mode raises error for forward references."""
        args = {"client_id": "{{turn_2.client_id}}"}
        previous_turns = {1: {"client_id": "abc-123"}}
        with pytest.raises(TemplateResolutionError, match="references turn 2 which is >= current turn 2"):
            resolve_template(args, previous_turns, turn_number=2, strict=True)

    def test_resolve_non_strict_mode(self):
        """Test that non-strict mode leaves unresolved templates as-is."""
        args = {"client_id": "{{turn_1.client_id}}"}
        previous_turns = {}  # Turn 1 doesn't exist
        resolved = resolve_template(args, previous_turns, turn_number=2, strict=False)
        assert resolved == {"client_id": "{{turn_1.client_id}}"}


class TestValidateTemplateReferences:
    """Test validating template references."""

    def test_validate_valid_references(self):
        """Test validation passes for valid references."""
        args = {"client_id": "{{turn_1.client_id}}"}
        previous_turns = {1: {"client_id": "abc-123"}}
        errors = validate_template_references(args, previous_turns, turn_number=2)
        assert errors == []

    def test_validate_missing_turn(self):
        """Test validation catches missing turn."""
        args = {"client_id": "{{turn_1.client_id}}"}
        previous_turns = {}
        errors = validate_template_references(args, previous_turns, turn_number=2)
        assert len(errors) == 1
        assert "doesn't exist" in errors[0]

    def test_validate_missing_field(self):
        """Test validation catches missing field."""
        args = {"client_id": "{{turn_1.client_id}}"}
        previous_turns = {1: {"name": "Acme Corp"}}
        errors = validate_template_references(args, previous_turns, turn_number=2)
        assert len(errors) == 1
        assert "field 'client_id' which doesn't exist" in errors[0]

    def test_validate_forward_reference(self):
        """Test validation catches forward references."""
        args = {"client_id": "{{turn_2.client_id}}"}
        previous_turns = {1: {"client_id": "abc-123"}}
        errors = validate_template_references(args, previous_turns, turn_number=2)
        assert len(errors) == 1
        assert ">= current turn 2" in errors[0]

    def test_validate_multiple_errors(self):
        """Test validation catches multiple errors."""
        args = {
            "client_id": "{{turn_1.client_id}}",
            "opportunity_id": "{{turn_3.opportunity_id}}",
        }
        previous_turns = {}
        errors = validate_template_references(args, previous_turns, turn_number=2)
        assert len(errors) >= 2


class TestFindCircularDependencies:
    """Test finding circular dependencies."""

    def test_no_circular_dependencies(self):
        """Test that linear dependencies don't trigger errors."""
        templates = [
            {},  # Turn 1 references nothing
            {"opportunity_id": "{{turn_1.client_id}}"},  # Turn 2 references turn 1
            {"quote_id": "{{turn_2.opportunity_id}}"},  # Turn 3 references turn 2
        ]
        errors = find_circular_dependencies(templates)
        assert errors == []

    def test_forward_reference_detection(self):
        """Test that forward references are detected."""
        templates = [
            {"client_id": "{{turn_2.client_id}}"},  # Turn 1 references turn 2 (forward!)
            {"client_id": "abc-123"},
        ]
        errors = find_circular_dependencies(templates)
        assert len(errors) == 1
        assert "forward reference" in errors[0]

    def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected."""
        templates = [
            {"client_id": "{{turn_2.client_id}}"},  # Turn 1 references turn 2 (forward reference)
            {"client_id": "{{turn_1.client_id}}"},  # Turn 2 references turn 1
        ]
        errors = find_circular_dependencies(templates)
        # Should detect both forward reference and circular dependency
        assert len(errors) >= 1
        # Check for either forward reference or circular dependency
        error_messages = " ".join(errors)
        assert "forward reference" in error_messages or "Circular dependency" in error_messages

    def test_complex_dependency_chain(self):
        """Test detection in complex dependency chains."""
        templates = [
            {"client_id": "abc-123"},  # Turn 1: no dependencies
            {"opportunity_id": "{{turn_1.client_id}}"},  # Turn 2: references turn 1
            {"quote_id": "{{turn_2.opportunity_id}}"},  # Turn 3: references turn 2
            {"client_id": "{{turn_1.client_id}}"},  # Turn 4: references turn 1
        ]
        errors = find_circular_dependencies(templates)
        assert errors == []


class TestRealWorldScenarios:
    """Test template resolution with realistic scenarios."""

    def test_client_onboarding_workflow(self):
        """Test resolving templates in a client onboarding workflow."""
        # Turn 1: Create client
        turn_1_result = {"client_id": "client-123"}
        
        # Turn 2: Create contact (references turn 1)
        turn_2_args = {
            "client_id": "{{turn_1.client_id}}",
            "first_name": "John",
            "last_name": "Doe",
        }
        resolved_2 = resolve_template(turn_2_args, {1: turn_1_result}, turn_number=2)
        assert resolved_2["client_id"] == "client-123"
        
        # Turn 3: Create opportunity (references turn 1)
        turn_3_args = {
            "client_id": "{{turn_1.client_id}}",
            "name": "Migration Project",
            "amount": 50000,
            "stage": "Prospecting",
        }
        resolved_3 = resolve_template(turn_3_args, {1: turn_1_result}, turn_number=3)
        assert resolved_3["client_id"] == "client-123"

    def test_deal_pipeline_workflow(self):
        """Test resolving templates in a deal pipeline workflow."""
        previous_turns = {
            1: {"client_id": "client-123"},
            2: {"opportunity_id": "opp-456"},
            4: {"quote_id": "quote-789"},
        }
        
        # Turn 5: Modify quote (references turn 4)
        turn_5_args = {
            "quote_id": "{{turn_4.quote_id}}",
            "updates": {"status": "Sent"},
        }
        resolved = resolve_template(turn_5_args, previous_turns, turn_number=5)
        assert resolved["quote_id"] == "quote-789"
        
        # Turn 6: Modify opportunity (references turn 2)
        turn_6_args = {
            "opportunity_id": "{{turn_2.opportunity_id}}",
            "updates": {"stage": "Negotiation"},
        }
        resolved = resolve_template(turn_6_args, previous_turns, turn_number=6)
        assert resolved["opportunity_id"] == "opp-456"

    def test_nested_updates_dict(self):
        """Test resolving templates in nested update dictionaries."""
        previous_turns = {
            1: {"opportunity_id": "opp-123"},
            2: {"stage": "Negotiation"},
        }
        
        args = {
            "opportunity_id": "{{turn_1.opportunity_id}}",
            "updates": {
                "stage": "{{turn_2.stage}}",
                "probability": 75,
            },
        }
        
        resolved = resolve_template(args, previous_turns, turn_number=3)
        assert resolved["opportunity_id"] == "opp-123"
        assert resolved["updates"]["stage"] == "Negotiation"
        assert resolved["updates"]["probability"] == 75

