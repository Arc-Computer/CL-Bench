"""Reference resolver for conversation template references.

Handles resolution of {{turn_N.field}} templates in conversation expected_args,
converting them to actual values from previous turn results.

Example:
    turn_1_result = {"client_id": "abc-123", "name": "Acme Corp"}
    expected_args = {"client_id": "{{turn_1.client_id}}", "amount": 50000}
    
    resolved = resolve_template(expected_args, {1: turn_1_result})
    # Returns: {"client_id": "abc-123", "amount": 50000}
"""

import re
from typing import Any, Dict, List, Optional, Set


# Pattern to match {{turn_N.field}} templates
TEMPLATE_PATTERN = re.compile(r'\{\{turn_(\d+)\.(\w+)\}\}')


class TemplateResolutionError(Exception):
    """Raised when template resolution fails."""
    pass


def extract_template_references(args: Dict[str, Any]) -> List[tuple]:
    """Extract all template references from a dictionary.
    
    Args:
        args: Dictionary potentially containing {{turn_N.field}} strings
        
    Returns:
        List of (turn_number, field_name) tuples
    """
    references = []
    
    def _extract_from_value(value: Any) -> None:
        if isinstance(value, str):
            matches = TEMPLATE_PATTERN.findall(value)
            for turn_num_str, field_name in matches:
                references.append((int(turn_num_str), field_name))
        elif isinstance(value, dict):
            for v in value.values():
                _extract_from_value(v)
        elif isinstance(value, list):
            for item in value:
                _extract_from_value(item)
    
    _extract_from_value(args)
    return references


def resolve_template(
    args: Dict[str, Any],
    previous_turns: Dict[int, Dict[str, Any]],
    turn_number: int,
    strict: bool = True
) -> Dict[str, Any]:
    """Resolve {{turn_N.field}} templates in arguments dictionary.
    
    Args:
        args: Dictionary containing template references
        previous_turns: Dictionary mapping turn numbers to their results
        turn_number: Current turn number (for validation)
        strict: If True, raise error on unresolved templates; if False, leave as-is
        
    Returns:
        Dictionary with templates resolved to actual values
        
    Raises:
        TemplateResolutionError: If template references non-existent turn or field
    """
    def _resolve_value(value: Any) -> Any:
        if isinstance(value, str):
            def _replace_template(match: re.Match) -> str:
                turn_num = int(match.group(1))
                field_name = match.group(2)
                
                # Validate turn number exists and is before current turn
                if turn_num >= turn_number:
                    if strict:
                        raise TemplateResolutionError(
                            f"Template {{turn_{turn_num}.{field_name}}} references "
                            f"turn {turn_num} which is >= current turn {turn_number}"
                        )
                    return match.group(0)  # Leave unresolved
                
                if turn_num not in previous_turns:
                    if strict:
                        raise TemplateResolutionError(
                            f"Template {{turn_{turn_num}.{field_name}}} references "
                            f"turn {turn_num} which doesn't exist in previous_turns"
                        )
                    return match.group(0)  # Leave unresolved
                
                turn_result = previous_turns[turn_num]
                if field_name not in turn_result:
                    if strict:
                        raise TemplateResolutionError(
                            f"Template {{turn_{turn_num}.{field_name}}} references "
                            f"field '{field_name}' which doesn't exist in turn {turn_num} result. "
                            f"Available fields: {list(turn_result.keys())}"
                        )
                    return match.group(0)  # Leave unresolved
                
                return str(turn_result[field_name])
            
            return TEMPLATE_PATTERN.sub(_replace_template, value)
        elif isinstance(value, dict):
            return {k: _resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_resolve_value(item) for item in value]
        else:
            return value
    
    return _resolve_value(args)


def validate_template_references(
    args: Dict[str, Any],
    previous_turns: Dict[int, Dict[str, Any]],
    turn_number: int
) -> List[str]:
    """Validate that all template references are resolvable.
    
    Args:
        args: Dictionary containing template references
        previous_turns: Dictionary mapping turn numbers to their results
        turn_number: Current turn number
        
    Returns:
        List of error messages (empty if all valid)
    """
    errors = []
    references = extract_template_references(args)
    
    for turn_num, field_name in references:
        # Check turn number is before current turn
        if turn_num >= turn_number:
            errors.append(
                f"Template {{turn_{turn_num}.{field_name}}} references "
                f"turn {turn_num} which is >= current turn {turn_number}"
            )
            continue
        
        # Check turn exists
        if turn_num not in previous_turns:
            errors.append(
                f"Template {{turn_{turn_num}.{field_name}}} references "
                f"turn {turn_num} which doesn't exist"
            )
            continue
        
        # Check field exists
        turn_result = previous_turns[turn_num]
        if field_name not in turn_result:
            errors.append(
                f"Template {{turn_{turn_num}.{field_name}}} references "
                f"field '{field_name}' which doesn't exist in turn {turn_num}. "
                f"Available fields: {list(turn_result.keys())}"
            )
    
    return errors


def find_circular_dependencies(
    templates: List[Dict[str, Any]],
    turn_results: Optional[Dict[int, Dict[str, Any]]] = None
) -> List[str]:
    """Find circular dependencies in template references.
    
    A circular dependency occurs when turn N references turn M, and turn M references turn N.
    
    Args:
        templates: List of argument dictionaries for each turn
        turn_results: Optional previous turn results (for validation)
        
    Returns:
        List of error messages describing circular dependencies (empty if none)
    """
    if turn_results is None:
        turn_results = {}
    
    errors = []
    dependencies: Dict[int, Set[int]] = {}  # turn_num -> set of referenced turn numbers
    
    for turn_idx, template in enumerate(templates):
        turn_num = turn_idx + 1
        references = extract_template_references(template)
        referenced_turns = {turn_num for turn_num, _ in references}
        dependencies[turn_num] = referenced_turns
        
        # Check for forward references (turn references future turn)
        for ref_turn in referenced_turns:
            if ref_turn >= turn_num:
                errors.append(
                    f"Turn {turn_num} references turn {ref_turn} (forward reference)"
                )
    
    # Check for circular dependencies
    for turn_num, referenced_turns in dependencies.items():
        for ref_turn in referenced_turns:
            if ref_turn in dependencies and turn_num in dependencies[ref_turn]:
                errors.append(
                    f"Circular dependency: Turn {turn_num} references turn {ref_turn}, "
                    f"and turn {ref_turn} references turn {turn_num}"
                )
    
    return errors


def get_required_fields_from_templates(
    templates: List[Dict[str, Any]]
) -> Dict[int, Set[str]]:
    """Extract fields that each turn requires from previous turns.
    
    Args:
        templates: List of argument dictionaries for each turn
        
    Returns:
        Dictionary mapping turn number to set of required field names
    """
    required_fields: Dict[int, Set[str]] = {}
    
    for turn_idx, template in enumerate(templates):
        turn_num = turn_idx + 1
        references = extract_template_references(template)
        required_fields[turn_num] = {(turn_num, field_name) for turn_num, field_name in references}
    
    return required_fields

