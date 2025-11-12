#!/usr/bin/env python3
"""Test script to verify token tracking works correctly."""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integration.atlas_integration import _store_student_and_judge_token_usage
from atlas.runtime.orchestration.execution_context import ExecutionContext


def create_mock_conversation_result() -> dict:
    """Create a mock conversation_result with token usage."""
    return {
        "metadata": {
            "agent": {
                "token_usage": {
                    "prompt_tokens": 1500,
                    "completion_tokens": 200,
                    "total_tokens": 1700,
                }
            }
        },
        "per_turn_results": [
            {
                "token_usage": {
                    "judge": {
                        "prompt_tokens": 800,
                        "completion_tokens": 100,
                        "total_tokens": 900,
                    },
                    "judge_response": {
                        "prompt_tokens": 600,
                        "completion_tokens": 50,
                        "total_tokens": 650,
                    }
                }
            },
            {
                "token_usage": {
                    "judge": {
                        "prompt_tokens": 700,
                        "completion_tokens": 80,
                        "total_tokens": 780,
                    }
                }
            }
        ]
    }


def test_token_tracking():
    """Test that token tracking function works correctly."""
    print("=" * 70)
    print("TESTING TOKEN TRACKING")
    print("=" * 70)
    
    # Reset ExecutionContext
    context = ExecutionContext.get()
    context.reset()
    
    # Create mock conversation result
    conv_result = create_mock_conversation_result()
    
    print("\n1. Before calling _store_student_and_judge_token_usage():")
    print(f"   ExecutionContext metadata: {context.metadata.get('token_usage', 'NOT SET')}")
    
    # Call the function
    _store_student_and_judge_token_usage(conv_result)
    
    print("\n2. After calling _store_student_and_judge_token_usage():")
    token_usage = context.metadata.get("token_usage", {})
    
    if not token_usage:
        print("   ❌ ERROR: token_usage not found in ExecutionContext metadata")
        return False
    
    print(f"   token_usage keys: {list(token_usage.keys())}")
    
    # Check student tokens
    student = token_usage.get("student", {})
    if student:
        print(f"\n   ✅ Student tokens:")
        print(f"      prompt_tokens: {student.get('prompt_tokens')} (expected: 1500)")
        print(f"      completion_tokens: {student.get('completion_tokens')} (expected: 200)")
        print(f"      total_tokens: {student.get('total_tokens')} (expected: 1700)")
        
        if student.get("prompt_tokens") != 1500:
            print("      ❌ Student prompt_tokens mismatch!")
            return False
        if student.get("completion_tokens") != 200:
            print("      ❌ Student completion_tokens mismatch!")
            return False
    else:
        print("   ❌ ERROR: student tokens not found")
        return False
    
    # Check judge tokens
    judge = token_usage.get("judge", {})
    if judge:
        expected_judge_prompt = 800 + 700  # From both turns
        expected_judge_completion = 100 + 80  # From both turns
        expected_judge_response_prompt = 600  # From first turn
        expected_judge_response_completion = 50  # From first turn
        
        total_judge_prompt = expected_judge_prompt + expected_judge_response_prompt
        total_judge_completion = expected_judge_completion + expected_judge_response_completion
        
        print(f"\n   ✅ Judge tokens:")
        print(f"      prompt_tokens: {judge.get('prompt_tokens')} (expected: {total_judge_prompt})")
        print(f"      completion_tokens: {judge.get('completion_tokens')} (expected: {total_judge_completion})")
        print(f"      total_tokens: {judge.get('total_tokens')} (expected: {total_judge_prompt + total_judge_completion})")
        
        if judge.get("prompt_tokens") != total_judge_prompt:
            print(f"      ❌ Judge prompt_tokens mismatch! Got {judge.get('prompt_tokens')}, expected {total_judge_prompt}")
            return False
        if judge.get("completion_tokens") != total_judge_completion:
            print(f"      ❌ Judge completion_tokens mismatch! Got {judge.get('completion_tokens')}, expected {total_judge_completion}")
            return False
    else:
        print("   ❌ ERROR: judge tokens not found")
        return False
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
    return True


if __name__ == "__main__":
    try:
        success = test_token_tracking()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

