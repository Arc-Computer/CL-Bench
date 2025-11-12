#!/usr/bin/env python3
"""Clear Atlas database for fair evaluation baseline.

This script clears learning state and session data to ensure a clean
baseline for evaluation runs. Use before running evaluations to ensure
fair, reproducible results.
"""

import argparse
import os
import sys
from pathlib import Path

# Add repo root to path
_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
import asyncpg
from datetime import datetime


async def clear_database(
    database_url: str,
    learning_key_pattern: str | None = None,
    clear_all: bool = False,
    dry_run: bool = False,
) -> None:
    """Clear Atlas database tables for evaluation.
    
    Args:
        database_url: PostgreSQL connection URL
        learning_key_pattern: Pattern to match learning keys (e.g., 'crm-benchmark%')
        clear_all: If True, clear all data regardless of learning_key
        dry_run: If True, show what would be cleared without actually clearing
    """
    conn = await asyncpg.connect(database_url)
    
    try:
        print("=" * 70)
        print("DATABASE CLEARING FOR FAIR EVALUATION")
        print("=" * 70)
        print()
        
        if dry_run:
            print("🔍 DRY RUN MODE - No data will be deleted")
            print()
        
        # Check current state
        if learning_key_pattern:
            print(f"Checking sessions with learning_key matching: {learning_key_pattern}")
            session_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM sessions
                WHERE metadata->>'learning_key' LIKE $1
                """,
                learning_key_pattern,
            )
            learning_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM learning_registry
                WHERE learning_key LIKE $1
                """,
                learning_key_pattern,
            )
        elif clear_all:
            print("Checking all sessions and learning entries...")
            session_count = await conn.fetchval("SELECT COUNT(*) FROM sessions")
            learning_count = await conn.fetchval("SELECT COUNT(*) FROM learning_registry")
        else:
            print("⚠️  No pattern specified and clear_all=False. Nothing to clear.")
            return
        
        print(f"  Sessions to clear: {session_count}")
        print(f"  Learning entries to clear: {learning_count}")
        print()
        
        if session_count == 0 and learning_count == 0:
            print("✅ Database is already clean - nothing to clear")
            return
        
        if not dry_run:
            # Confirm before clearing
            if not clear_all and learning_key_pattern:
                print(f"⚠️  WARNING: This will delete {session_count} sessions and {learning_count} learning entries")
                print(f"   matching pattern: {learning_key_pattern}")
            elif clear_all:
                print(f"⚠️  WARNING: This will delete ALL {session_count} sessions and {learning_count} learning entries")
            
            print()
            response = input("Type 'yes' to confirm deletion: ")
            if response.lower() != 'yes':
                print("❌ Cancelled - no data deleted")
                return
        
        # Clear data
        print()
        print("Clearing database...")
        
        if clear_all:
            # Clear all data
            if not dry_run:
                await conn.execute("DELETE FROM trajectory_events")
                await conn.execute("DELETE FROM guidance_notes")
                await conn.execute("DELETE FROM step_attempts")
                await conn.execute("DELETE FROM step_results")
                await conn.execute("DELETE FROM plans")
                await conn.execute("DELETE FROM sessions")
                await conn.execute("DELETE FROM learning_registry")
                print("✅ Cleared all sessions and learning entries")
            else:
                print("  [DRY RUN] Would delete all sessions and learning entries")
        elif learning_key_pattern:
            # Clear data matching pattern
            if not dry_run:
                # Delete sessions (CASCADE will handle related tables)
                deleted_sessions = await conn.execute(
                    """
                    DELETE FROM sessions
                    WHERE metadata->>'learning_key' LIKE $1
                    """,
                    learning_key_pattern,
                )
                
                # Delete learning registry entries
                deleted_learning = await conn.execute(
                    """
                    DELETE FROM learning_registry
                    WHERE learning_key LIKE $1
                    """,
                    learning_key_pattern,
                )
                
                print(f"✅ Cleared sessions and learning entries matching: {learning_key_pattern}")
            else:
                print(f"  [DRY RUN] Would delete sessions and learning entries matching: {learning_key_pattern}")
        
        # Verify clearing
        if not dry_run:
            if clear_all:
                remaining_sessions = await conn.fetchval("SELECT COUNT(*) FROM sessions")
                remaining_learning = await conn.fetchval("SELECT COUNT(*) FROM learning_registry")
            elif learning_key_pattern:
                remaining_sessions = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM sessions
                    WHERE metadata->>'learning_key' LIKE $1
                    """,
                    learning_key_pattern,
                )
                remaining_learning = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM learning_registry
                    WHERE learning_key LIKE $1
                    """,
                    learning_key_pattern,
                )
            
            print(f"  Remaining sessions: {remaining_sessions}")
            print(f"  Remaining learning entries: {remaining_learning}")
        
        print()
        print("=" * 70)
        print("✅ Database cleared successfully")
        print("=" * 70)
        
    finally:
        await conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clear Atlas database for fair evaluation baseline"
    )
    parser.add_argument(
        "--database-url",
        help="PostgreSQL connection URL (default: from STORAGE__DATABASE_URL env var)",
    )
    parser.add_argument(
        "--learning-key-pattern",
        default="crm-benchmark%",
        help="Pattern to match learning keys (default: crm-benchmark%%)",
    )
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Clear all data regardless of learning_key (use with caution!)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleared without actually clearing",
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get database URL
    database_url = args.database_url or os.getenv("STORAGE__DATABASE_URL")
    if not database_url:
        print("❌ Error: --database-url or STORAGE__DATABASE_URL must be set")
        return 1
    
    # Run clearing
    import asyncio
    asyncio.run(
        clear_database(
            database_url=database_url,
            learning_key_pattern=args.learning_key_pattern if not args.clear_all else None,
            clear_all=args.clear_all,
            dry_run=args.dry_run,
        )
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

