"""CLI script for generating multi-turn conversations using Curator."""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crm_sandbox import MockCrmApi
from src.entity_sampler import EntitySampler, SamplerConfig
from src.curator_conversation_generator import CuratorConversationDatasetGenerator
from src.conversation_schema import Conversation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_conversations(
    target_count: int = 1500,
    simple_count: int = 900,
    medium_count: int = 450,
    complex_count: int = 150,
    output_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    smoke_test: bool = False,
) -> None:
    """Generate multi-turn conversations using Curator.
    
    Args:
        target_count: Total number of conversations (ignored if smoke_test=True)
        simple_count: Number of simple conversations (ignored if smoke_test=True)
        medium_count: Number of medium conversations (ignored if smoke_test=True)
        complex_count: Number of complex conversations (ignored if smoke_test=True)
        output_dir: Output directory for conversations.jsonl
        seed: Random seed for reproducibility
        smoke_test: If True, generate only 10 conversations (3 simple, 5 medium, 2 complex)
    """
    if output_dir is None:
        output_dir = Path("artifacts/conversations")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Initializing CRM API...")
    api = MockCrmApi()
    
    print(f"Configuring entity sampler...")
    config = SamplerConfig(seed=seed)
    sampler = EntitySampler(api, config)
    
    print(f"Initializing Curator conversation generator...")
    generator = CuratorConversationDatasetGenerator(api=api, sampler=sampler, seed=seed)
    
    if smoke_test:
        print(f"Running SMOKE TEST: Generating 10 conversations (3 simple, 5 medium, 2 complex)...")
        conversations = generator.generate_conversations(
            simple_count=3,
            medium_count=5,
            complex_count=2,
            batch_size=5,
        )
    else:
        print(f"Generating {target_count} conversations ({simple_count} simple, {medium_count} medium, {complex_count} complex)...")
        conversations = generator.generate_conversations(
            simple_count=simple_count,
            medium_count=medium_count,
            complex_count=complex_count,
            batch_size=10,
        )
    
    print(f"✓ Generated {len(conversations)} conversations")
    
    # Validate conversations
    print(f"Validating conversations...")
    valid_conversations = []
    validation_errors = []
    
    for conv in conversations:
        try:
            # Basic validation
            if not conv.turns:
                validation_errors.append(f"{conv.conversation_id}: No turns")
                continue
            
            # Check template resolution
            from src.reference_resolver import resolve_template, validate_template_references
            
            previous_turns_dict = {}
            for turn in conv.turns:
                if turn.turn_id > 1:
                    # Resolve templates for this turn
                    errors = validate_template_references(
                        turn.expected_args,
                        previous_turns_dict,
                        turn.turn_id,
                    )
                    if errors:
                        validation_errors.append(
                            f"{conv.conversation_id} turn {turn.turn_id}: {errors[0]}"
                        )
                
                # Simulate execution result (simplified)
                # In real execution, this would come from actual tool execution
                execution_result = {}
                for key in ["client_id", "contact_id", "opportunity_id", "quote_id", "contract_id"]:
                    if key in turn.expected_args:
                        execution_result[key] = turn.expected_args[key]
                    elif "updates" in turn.expected_args and isinstance(turn.expected_args["updates"], dict):
                        if key in turn.expected_args["updates"]:
                            execution_result[key] = turn.expected_args["updates"][key]
                
                previous_turns_dict[turn.turn_id] = execution_result
            
            valid_conversations.append(conv)
        except Exception as e:
            validation_errors.append(f"{conv.conversation_id}: {e}")
    
    if validation_errors:
        print(f"⚠️  Found {len(validation_errors)} validation errors:")
        for error in validation_errors[:10]:
            print(f"  - {error}")
        if len(validation_errors) > 10:
            print(f"  ... and {len(validation_errors) - 10} more")
    
    print(f"✓ {len(valid_conversations)} valid conversations")
    
    # Write conversations to JSONL
    print(f"Writing conversations to {output_dir}...")
    conversations_path = output_dir / "conversations.jsonl"
    
    with conversations_path.open("w", encoding="utf-8") as f:
        for conv in valid_conversations:
            # Convert Conversation to dict for JSON serialization
            conv_dict = {
                "conversation_id": conv.conversation_id,
                "workflow_category": conv.workflow_category,
                "complexity_level": conv.complexity_level,
                "turns": [
                    {
                        "turn_id": turn.turn_id,
                        "user_utterance": turn.user_utterance,
                        "expected_tool": turn.expected_tool,
                        "expected_args": turn.expected_args,
                        "references_previous_turns": turn.references_previous_turns,
                        "expect_success": turn.expect_success,
                    }
                    for turn in conv.turns
                ],
                "initial_entities": conv.initial_entities,
                "verification_mode": conv.verification_mode.value,
            }
            f.write(json.dumps(conv_dict) + "\n")
    
    print(f"  - Conversations: {conversations_path}")
    
    # Print statistics
    simple_count = sum(1 for c in valid_conversations if c.complexity_level == "simple")
    medium_count = sum(1 for c in valid_conversations if c.complexity_level == "medium")
    complex_count = sum(1 for c in valid_conversations if c.complexity_level == "complex")
    
    print(f"\n✓ Generation complete!")
    print(f"  Total: {len(valid_conversations)}")
    print(f"  Simple: {simple_count}")
    print(f"  Medium: {medium_count}")
    print(f"  Complex: {complex_count}")
    print(f"  Average turns: {sum(len(c.turns) for c in valid_conversations) / len(valid_conversations):.1f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate multi-turn CRM conversations")
    parser.add_argument("--count", type=int, default=1500, help="Total number of conversations")
    parser.add_argument("--simple", type=int, default=900, help="Number of simple conversations")
    parser.add_argument("--medium", type=int, default=450, help="Number of medium conversations")
    parser.add_argument("--complex", type=int, default=150, help="Number of complex conversations")
    parser.add_argument("--output-dir", type=str, help="Output directory path")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--smoke-test", action="store_true", help="Generate only 10 conversations for smoke test")
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir) if args.output_dir else None
    
    generate_conversations(
        target_count=args.count,
        simple_count=args.simple,
        medium_count=args.medium,
        complex_count=args.complex,
        output_dir=output_path,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )

