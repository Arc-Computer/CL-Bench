#!/usr/bin/env python3
"""Monitor Atlas learning and performance trends.

Run this script periodically to track learning accumulation and performance improvement.
"""

import asyncio
import asyncpg
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))

# Load environment
env_file = _REPO_ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


async def monitor_atlas_learning():
    """Monitor current Atlas run learning and performance."""
    db_url = os.getenv("STORAGE__DATABASE_URL")
    if not db_url:
        print("⚠️  STORAGE__DATABASE_URL not set")
        return

    try:
        conn = await asyncpg.connect(db_url)

        print("=" * 80)
        print("ATLAS LEARNING & PERFORMANCE MONITOR")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()

        # Get current run learning key
        learning_key = await conn.fetchval("""
            SELECT metadata->>'learning_key'
            FROM sessions
            WHERE metadata->>'source' = 'crm-benchmark'
            ORDER BY created_at DESC
            LIMIT 1
        """)

        if not learning_key:
            print("⚠️  No CRM benchmark sessions found")
            await conn.close()
            return

        # Count sessions
        session_count = await conn.fetchval("""
            SELECT COUNT(*)
            FROM sessions
            WHERE metadata->>'source' = 'crm-benchmark'
            AND metadata->>'learning_key' = $1
        """, learning_key)

        # Get learning registry
        learning = await conn.fetchrow("""
            SELECT student_learning, teacher_learning, updated_at
            FROM learning_registry
            WHERE learning_key = $1
        """, learning_key)

        # Get reward scores
        rewards = await conn.fetch("""
            SELECT reward, created_at
            FROM sessions
            WHERE metadata->>'source' = 'crm-benchmark'
            AND metadata->>'learning_key' = $1
            AND reward IS NOT NULL
            ORDER BY created_at ASC
        """, learning_key)

        # Get learning usage
        learning_usage = await conn.fetch("""
            SELECT metadata->'learning_usage'->'session' as usage
            FROM sessions
            WHERE metadata->>'source' = 'crm-benchmark'
            AND metadata->>'learning_key' = $1
            AND metadata->'learning_usage'->'session' IS NOT NULL
        """, learning_key)

        print(f"📊 Current Run Status")
        print(f"   Sessions: {session_count} / 400")
        print(f"   Learning Key: {learning_key[:40]}...")
        print()

        if learning:
            student_len = len(learning["student_learning"]) if learning["student_learning"] else 0
            teacher_len = len(learning["teacher_learning"]) if learning["teacher_learning"] else 0
            total_len = student_len + teacher_len

            print(f"📚 Learning Accumulation")
            print(f"   Student Learning: {student_len:,} chars")
            print(f"   Teacher Learning: {teacher_len:,} chars")
            print(f"   Total: {total_len:,} chars")
            print(f"   Last Updated: {learning['updated_at']}")
            print()

        if rewards:
            scores = []
            for r in rewards:
                try:
                    reward = json.loads(r["reward"]) if isinstance(r["reward"], str) else r["reward"]
                    score = reward.get("score")
                    if score is not None:
                        scores.append(float(score))
                except:
                    pass

            if scores:
                print(f"🎯 Performance Metrics")
                print(f"   Sessions Scored: {len(scores)} / {session_count}")
                print(f"   Average Score: {sum(scores) / len(scores):.3f}")
                print(f"   Score Range: {min(scores):.3f} - {max(scores):.3f}")

                if len(scores) >= 2:
                    first_half = scores[: len(scores) // 2]
                    second_half = scores[len(scores) // 2 :]
                    first_avg = sum(first_half) / len(first_half)
                    second_avg = sum(second_half) / len(second_half)
                    trend = second_avg - first_avg
                    if first_avg > 0:
                        print(f"   Trend: {trend:+.3f} ({trend/first_avg*100:+.1f}%)")
                    else:
                        print(f"   Trend: {trend:+.3f}")
                print()

        if learning_usage:
            total_cue_hits = 0
            total_adoptions = 0
            sessions_with_adoption = 0

            for usage in learning_usage:
                try:
                    u = json.loads(usage["usage"]) if isinstance(usage["usage"], str) else usage["usage"]
                    cue_hits = u.get("cue_hits", 0)
                    adoptions = u.get("action_adoptions", 0)
                    total_cue_hits += cue_hits
                    total_adoptions += adoptions
                    if adoptions > 0:
                        sessions_with_adoption += 1
                except:
                    pass

            print(f"🔄 Learning Adoption")
            print(f"   Cue Hits: {total_cue_hits}")
            print(f"   Action Adoptions: {total_adoptions}")
            print(f"   Sessions with Adoption: {sessions_with_adoption} / {session_count}")
            print()

        # Show recent learning content
        if learning and learning["student_learning"]:
            print(f"📖 Recent Student Learning:")
            print(f"   {learning['student_learning'][:200]}...")
            print()

        if learning and learning["teacher_learning"]:
            print(f"👨‍🏫 Recent Teacher Learning:")
            print(f"   {learning['teacher_learning'][:200]}...")
            print()

        await conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(monitor_atlas_learning())

