#!/usr/bin/env python3
"""Extract complete token usage from sessions.jsonl and database.

This script extracts all available token usage data from:
1. sessions.jsonl file (reward evaluation tokens)
2. Atlas database (learning registry, session metadata)
3. Attempts to reconstruct missing token data where possible
"""

import json
import sys
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

# Add repo root to path
_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))

try:
    import asyncpg
    from dotenv import load_dotenv
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


def extract_from_sessions_jsonl(sessions_path: Path) -> Dict[str, Any]:
    """Extract all token usage from sessions.jsonl."""
    reward_tokens = defaultdict(int)
    student_tokens = defaultdict(int)
    judge_tokens = defaultdict(int)
    learning_tokens = defaultdict(int)
    
    sessions_with_reward = 0
    sessions_with_student = 0
    sessions_with_judge = 0
    sessions_with_learning = 0
    total_sessions = 0
    
    with open(sessions_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                session = json.loads(line)
                total_sessions += 1
                
                # 1. Reward audit tokens
                atlas_metadata = session.get("atlas_metadata", {})
                reward_audit = atlas_metadata.get("session_reward_audit", [])
                if reward_audit:
                    sessions_with_reward += 1
                    for entry in reward_audit:
                        if isinstance(entry, dict):
                            raw_response = entry.get("raw_response", {})
                            usage = raw_response.get("usage", {})
                            if usage:
                                reward_tokens["prompt_tokens"] += usage.get("prompt_tokens", 0)
                                reward_tokens["completion_tokens"] += usage.get("completion_tokens", 0)
                                reward_tokens["total_tokens"] += usage.get("total_tokens", 0)
                
                # 2. Student tokens from conversation_result
                conv_result = session.get("conversation_result")
                if conv_result:
                    metadata = conv_result.get("metadata", {})
                    agent_metadata = metadata.get("agent", {})
                    agent_token_usage = agent_metadata.get("token_usage", {})
                    if agent_token_usage:
                        sessions_with_student += 1
                        student_tokens["prompt_tokens"] += agent_token_usage.get("prompt_tokens", 0)
                        student_tokens["completion_tokens"] += agent_token_usage.get("completion_tokens", 0)
                        student_tokens["total_tokens"] += agent_token_usage.get("total_tokens", 0)
                    
                    # 3. Judge tokens from per-turn records
                    per_turn = conv_result.get("per_turn_results", [])
                    for turn in per_turn:
                        turn_tokens = turn.get("token_usage", {})
                        if isinstance(turn_tokens, dict):
                            for judge_key in ("judge", "judge_response"):
                                judge_usage = turn_tokens.get(judge_key, {})
                                if isinstance(judge_usage, dict) and judge_usage:
                                    sessions_with_judge += 1
                                    judge_tokens["prompt_tokens"] += judge_usage.get("prompt_tokens", 0)
                                    judge_tokens["completion_tokens"] += judge_usage.get("completion_tokens", 0)
                                    judge_tokens["total_tokens"] += judge_usage.get("total_tokens", 0)
                
                # 4. Learning tokens from atlas_metadata.token_usage.learning
                token_usage_meta = atlas_metadata.get("token_usage", {})
                if isinstance(token_usage_meta, dict):
                    learning_usage = token_usage_meta.get("learning", {})
                    if learning_usage:
                        sessions_with_learning += 1
                        learning_tokens["prompt_tokens"] += learning_usage.get("prompt_tokens", 0)
                        learning_tokens["completion_tokens"] += learning_usage.get("completion_tokens", 0)
                        learning_tokens["total_tokens"] += learning_usage.get("total_tokens", 0)
                        
            except Exception as e:
                continue
    
    return {
        "total_sessions": total_sessions,
        "by_role": {
            "reward": {
                "sessions": sessions_with_reward,
                "tokens": dict(reward_tokens),
            },
            "student": {
                "sessions": sessions_with_student,
                "tokens": dict(student_tokens),
            },
            "judge": {
                "sessions": sessions_with_judge,
                "tokens": dict(judge_tokens),
            },
            "learning": {
                "sessions": sessions_with_learning,
                "tokens": dict(learning_tokens),
            },
        },
    }


async def extract_from_database(db_url: str, learning_key: str = None) -> Dict[str, Any]:
    """Extract learning data and any token usage from database."""
    if not DB_AVAILABLE:
        return {"error": "Database libraries not available"}
    
    conn = await asyncpg.connect(db_url)
    
    try:
        # Get learning key if not provided
        if not learning_key:
            learning_key = await conn.fetchval("""
                SELECT metadata->>'learning_key'
                FROM sessions
                WHERE metadata->>'source' = 'crm-benchmark'
                ORDER BY created_at DESC
                LIMIT 1
            """)
        
        if not learning_key:
            return {"error": "No learning key found"}
        
        # Get learning registry
        learning = await conn.fetchrow("""
            SELECT 
                student_learning,
                teacher_learning,
                metadata,
                updated_at
            FROM learning_registry
            WHERE learning_key = $1
        """, learning_key)
        
        result = {
            "learning_key": learning_key,
            "learning_registry": None,
        }
        
        if learning:
            metadata = learning['metadata']
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            result["learning_registry"] = {
                "student_learning_length": len(learning['student_learning']) if learning['student_learning'] else 0,
                "teacher_learning_length": len(learning['teacher_learning']) if learning['teacher_learning'] else 0,
                "playbook_entries": len(metadata.get('playbook_entries', [])) if isinstance(metadata, dict) else 0,
                "updated_at": str(learning['updated_at']),
            }
        
        return result
        
    finally:
        await conn.close()


def calculate_costs(token_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate costs based on model pricing."""
    pricing = {
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "claude-4.5-sonnet": {"input": 3.00, "output": 15.00},
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    }
    
    costs = {}
    
    # Reward evaluation (GPT-4.1 mini)
    reward = token_data["by_role"]["reward"]["tokens"]
    if reward:
        costs["reward"] = {
            "model": "gpt-4.1-mini",
            "input_cost": (reward.get("prompt_tokens", 0) / 1_000_000) * pricing["gpt-4.1-mini"]["input"],
            "output_cost": (reward.get("completion_tokens", 0) / 1_000_000) * pricing["gpt-4.1-mini"]["output"],
            "total_cost": 0,
        }
        costs["reward"]["total_cost"] = costs["reward"]["input_cost"] + costs["reward"]["output_cost"]
    
    # Student (GPT-4.1 mini)
    student = token_data["by_role"]["student"]["tokens"]
    if student:
        costs["student"] = {
            "model": "gpt-4.1-mini",
            "input_cost": (student.get("prompt_tokens", 0) / 1_000_000) * pricing["gpt-4.1-mini"]["input"],
            "output_cost": (student.get("completion_tokens", 0) / 1_000_000) * pricing["gpt-4.1-mini"]["output"],
            "total_cost": 0,
        }
        costs["student"]["total_cost"] = costs["student"]["input_cost"] + costs["student"]["output_cost"]
    
    # Judge (GPT-4.1)
    judge = token_data["by_role"]["judge"]["tokens"]
    if judge:
        costs["judge"] = {
            "model": "gpt-4.1",
            "input_cost": (judge.get("prompt_tokens", 0) / 1_000_000) * pricing["gpt-4.1"]["input"],
            "output_cost": (judge.get("completion_tokens", 0) / 1_000_000) * pricing["gpt-4.1"]["output"],
            "total_cost": 0,
        }
        costs["judge"]["total_cost"] = costs["judge"]["input_cost"] + costs["judge"]["output_cost"]
    
    # Learning (Gemini 2.5 Flash)
    learning = token_data["by_role"]["learning"]["tokens"]
    if learning:
        costs["learning"] = {
            "model": "gemini-2.5-flash",
            "input_cost": (learning.get("prompt_tokens", 0) / 1_000_000) * pricing["gemini-2.5-flash"]["input"],
            "output_cost": (learning.get("completion_tokens", 0) / 1_000_000) * pricing["gemini-2.5-flash"]["output"],
            "total_cost": 0,
        }
        costs["learning"]["total_cost"] = costs["learning"]["input_cost"] + costs["learning"]["output_cost"]
    
    return costs


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: extract_complete_token_usage.py <sessions.jsonl> [learning_key]")
        sys.exit(1)
    
    sessions_path = Path(sys.argv[1])
    learning_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not sessions_path.exists():
        print(f"Error: File not found: {sessions_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("COMPLETE TOKEN USAGE EXTRACTION")
    print("=" * 70)
    print()
    
    # Extract from sessions.jsonl
    print("Extracting from sessions.jsonl...")
    token_data = extract_from_sessions_jsonl(sessions_path)
    
    # Extract from database if available
    db_data = None
    if DB_AVAILABLE:
        load_dotenv()
        db_url = os.getenv("STORAGE__DATABASE_URL")
        if db_url:
            print("Extracting from database...")
            import asyncio
            db_data = asyncio.run(extract_from_database(db_url, learning_key))
    
    # Calculate costs
    costs = calculate_costs(token_data)
    
    # Print results
    print()
    print("=" * 70)
    print("TOKEN USAGE SUMMARY")
    print("=" * 70)
    print()
    print(f"Total Sessions: {token_data['total_sessions']}")
    print()
    
    for role, data in token_data["by_role"].items():
        tokens = data["tokens"]
        sessions = data["sessions"]
        if any(tokens.values()):
            print(f"{role.upper()}:")
            print(f"  Sessions with data: {sessions}/{token_data['total_sessions']}")
            print(f"  Prompt tokens: {tokens.get('prompt_tokens', 0):,}")
            print(f"  Completion tokens: {tokens.get('completion_tokens', 0):,}")
            print(f"  Total tokens: {tokens.get('total_tokens', 0):,}")
            if role in costs:
                cost = costs[role]
                print(f"  Cost: ${cost['total_cost']:.4f} ({cost['model']})")
            print()
        else:
            print(f"{role.upper()}: ❌ No token data found")
            print()
    
    # Print totals
    total_prompt = sum(d["tokens"].get("prompt_tokens", 0) for d in token_data["by_role"].values())
    total_completion = sum(d["tokens"].get("completion_tokens", 0) for d in token_data["by_role"].values())
    total_tokens = sum(d["tokens"].get("total_tokens", 0) for d in token_data["by_role"].values())
    total_cost = sum(c["total_cost"] for c in costs.values())
    
    print("=" * 70)
    print("TOTALS")
    print("=" * 70)
    print(f"  Prompt tokens: {total_prompt:,}")
    print(f"  Completion tokens: {total_completion:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total cost: ${total_cost:.4f}")
    print()
    
    # Print database learning data
    if db_data and "learning_registry" in db_data and db_data["learning_registry"]:
        print("=" * 70)
        print("LEARNING REGISTRY (from database)")
        print("=" * 70)
        lr = db_data["learning_registry"]
        print(f"  Student learning: {lr['student_learning_length']:,} characters")
        print(f"  Teacher learning: {lr['teacher_learning_length']:,} characters")
        print(f"  Playbook entries: {lr['playbook_entries']}")
        print(f"  Last updated: {lr['updated_at']}")
        print()
    
    # Save to JSON
    output_path = sessions_path.parent / "complete_token_usage.json"
    output_data = {
        "token_usage": token_data,
        "costs": costs,
        "database": db_data,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Complete token usage data written to: {output_path}")


if __name__ == "__main__":
    main()

