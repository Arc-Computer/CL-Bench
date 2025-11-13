#!/usr/bin/env python3
"""Verify learning synthesis and persistence in Atlas database."""

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
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


async def verify_learning_persistence():
    """Verify learning synthesis and persistence."""
    db_url = os.getenv("STORAGE__DATABASE_URL")
    if not db_url:
        print("⚠️  STORAGE__DATABASE_URL not set")
        return

    try:
        conn = await asyncpg.connect(db_url)

        print("=" * 80)
        print("LEARNING SYNTHESIS & PERSISTENCE VERIFICATION")
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

        print(f"🔑 Learning Key: {learning_key[:50]}...")
        print()

        # 1. Check Learning Registry (Learning Synthesis)
        print("=" * 80)
        print("1. LEARNING REGISTRY (Learning Synthesis)")
        print("=" * 80)
        learning = await conn.fetchrow("""
            SELECT student_learning, teacher_learning, metadata, updated_at
            FROM learning_registry
            WHERE learning_key = $1
        """, learning_key)

        if learning:
            student_len = len(learning["student_learning"]) if learning["student_learning"] else 0
            teacher_len = len(learning["teacher_learning"]) if learning["teacher_learning"] else 0
            metadata = learning["metadata"] if learning["metadata"] else {}
            playbook_entries = metadata.get("playbook_entries", []) if isinstance(metadata, dict) else []

            print(f"✅ Learning Registry EXISTS")
            print(f"   Student Learning: {student_len:,} chars")
            print(f"   Teacher Learning: {teacher_len:,} chars")
            print(f"   Total: {student_len + teacher_len:,} chars")
            print(f"   Playbook Entries: {len(playbook_entries)}")
            print(f"   Last Updated: {learning['updated_at']}")
            
            if student_len > 0:
                print(f"\n   Student Learning Preview:")
                print(f"   {learning['student_learning'][:300]}...")
            
            if playbook_entries:
                print(f"\n   Playbook Entries:")
                for i, entry in enumerate(playbook_entries[:3], 1):
                    cue = entry.get("cue_pattern", "N/A")
                    action = entry.get("runtime_handle", "N/A")
                    print(f"   [{i}] Cue: {cue[:50]}... → Action: {action}")
        else:
            print("❌ Learning Registry NOT FOUND")
            print("   (Learning synthesis may not have started yet)")
        print()

        # 2. Check Session Metadata - Learning State Persistence
        print("=" * 80)
        print("2. SESSION METADATA - Learning State Persistence")
        print("=" * 80)
        sessions = await conn.fetch("""
            SELECT 
                id,
                created_at,
                jsonb_typeof(metadata->'learning_state') as learning_state_type,
                metadata->'learning_state'->>'student_learning' as learning_state_student,
                metadata->'learning_state'->>'teacher_learning' as learning_state_teacher,
                metadata->'learning_state'->'metadata'->'playbook_entries' as playbook_entries,
                LENGTH(metadata->'learning_state'->>'student_learning') as student_len
            FROM sessions 
            WHERE metadata->>'source' = 'crm-benchmark'
            AND metadata->>'learning_key' = $1
            ORDER BY created_at DESC
            LIMIT 10
        """, learning_key)

        if sessions:
            print(f"📊 Recent Sessions ({len(sessions)} shown):")
            print()
            
            sessions_with_learning_state = 0
            sessions_with_content = 0
            
            for i, s in enumerate(sessions, 1):
                learning_state_type = s['learning_state_type']
                student_len = s['student_len'] or 0
                has_content = student_len > 0
                
                status = "✅" if learning_state_type == 'object' else "❌"
                content_status = "📝" if has_content else "📭"
                
                print(f"   [{i}] Session {s['id']} ({s['created_at'].strftime('%H:%M:%S')})")
                print(f"       {status} learning_state_type: {learning_state_type}")
                print(f"       {content_status} student_learning: {student_len} chars")
                
                if learning_state_type == 'object':
                    sessions_with_learning_state += 1
                if has_content:
                    sessions_with_content += 1
                print()
            
            total_sessions = await conn.fetchval("""
                SELECT COUNT(*)
                FROM sessions
                WHERE metadata->>'source' = 'crm-benchmark'
                AND metadata->>'learning_key' = $1
            """, learning_key)
            
            print(f"📈 Summary:")
            print(f"   Total Sessions: {total_sessions}")
            print(f"   Sessions with learning_state: {sessions_with_learning_state} / {len(sessions)} ({sessions_with_learning_state/len(sessions)*100:.1f}%)")
            print(f"   Sessions with learning content: {sessions_with_content} / {len(sessions)} ({sessions_with_content/len(sessions)*100:.1f}%)")
        else:
            print("❌ No sessions found")
        print()

        # 3. Check Applied Student Learning Persistence
        print("=" * 80)
        print("3. APPLIED STUDENT LEARNING Persistence")
        print("=" * 80)
        applied_sessions = await conn.fetch("""
            SELECT 
                id,
                created_at,
                jsonb_typeof(metadata->'applied_student_learning') as applied_type,
                metadata->'applied_student_learning'->>'digest' as applied_digest,
                metadata->'applied_student_learning'->>'char_count' as applied_chars,
                metadata->'applied_student_learning'->>'entry_count' as applied_entries
            FROM sessions 
            WHERE metadata->>'source' = 'crm-benchmark'
            AND metadata->>'learning_key' = $1
            ORDER BY created_at DESC
            LIMIT 10
        """, learning_key)

        if applied_sessions:
            sessions_with_applied = sum(1 for s in applied_sessions if s['applied_type'] == 'object')
            
            print(f"📊 Recent Sessions ({len(applied_sessions)} shown):")
            print()
            
            for i, s in enumerate(applied_sessions, 1):
                applied_type = s['applied_type']
                digest = s['applied_digest']
                chars = s['applied_chars']
                entries = s['applied_entries']
                
                status = "✅" if applied_type == 'object' else "❌"
                
                print(f"   [{i}] Session {s['id']} ({s['created_at'].strftime('%H:%M:%S')})")
                print(f"       {status} applied_student_learning: {applied_type}")
                if applied_type == 'object':
                    print(f"       📝 digest: {digest[:20] if digest else 'N/A'}...")
                    print(f"       📝 chars: {chars}, entries: {entries}")
                print()
            
            total_sessions = await conn.fetchval("""
                SELECT COUNT(*)
                FROM sessions
                WHERE metadata->>'source' = 'crm-benchmark'
                AND metadata->>'learning_key' = $1
            """, learning_key)
            
            print(f"📈 Summary:")
            print(f"   Total Sessions: {total_sessions}")
            print(f"   Sessions with applied_student_learning: {sessions_with_applied} / {len(applied_sessions)} ({sessions_with_applied/len(applied_sessions)*100:.1f}%)")
        else:
            print("❌ No sessions found")
        print()

        # 4. Verify Learning Re-injection Flow
        print("=" * 80)
        print("4. LEARNING RE-INJECTION VERIFICATION")
        print("=" * 80)
        
        # Check if later sessions have more learning than earlier sessions
        early_sessions = await conn.fetch("""
            SELECT 
                id,
                created_at,
                LENGTH(metadata->'learning_state'->>'student_learning') as student_len
            FROM sessions 
            WHERE metadata->>'source' = 'crm-benchmark'
            AND metadata->>'learning_key' = $1
            ORDER BY created_at ASC
            LIMIT 5
        """, learning_key)
        
        late_sessions = await conn.fetch("""
            SELECT 
                id,
                created_at,
                LENGTH(metadata->'learning_state'->>'student_learning') as student_len
            FROM sessions 
            WHERE metadata->>'source' = 'crm-benchmark'
            AND metadata->>'learning_key' = $1
            ORDER BY created_at DESC
            LIMIT 5
        """, learning_key)
        
        if early_sessions and late_sessions:
            early_avg = sum(s['student_len'] or 0 for s in early_sessions) / len(early_sessions)
            late_avg = sum(s['student_len'] or 0 for s in late_sessions) / len(late_sessions)
            
            print(f"📊 Learning Accumulation Check:")
            print(f"   Early sessions (first 5): avg {early_avg:.0f} chars")
            print(f"   Late sessions (last 5): avg {late_avg:.0f} chars")
            
            if late_avg > early_avg:
                print(f"   ✅ Learning is accumulating! (+{late_avg - early_avg:.0f} chars)")
            elif late_avg == early_avg == 0:
                print(f"   ⚠️  No learning content yet (synthesis may not have started)")
            else:
                print(f"   ⚠️  Learning not accumulating as expected")
        print()

        # 5. Overall Status
        print("=" * 80)
        print("5. OVERALL STATUS")
        print("=" * 80)
        
        total_sessions = await conn.fetchval("""
            SELECT COUNT(*)
            FROM sessions
            WHERE metadata->>'source' = 'crm-benchmark'
            AND metadata->>'learning_key' = $1
        """, learning_key)
        
        sessions_with_learning_state = await conn.fetchval("""
            SELECT COUNT(*)
            FROM sessions
            WHERE metadata->>'source' = 'crm-benchmark'
            AND metadata->>'learning_key' = $1
            AND jsonb_typeof(metadata->'learning_state') = 'object'
        """, learning_key)
        
        sessions_with_applied = await conn.fetchval("""
            SELECT COUNT(*)
            FROM sessions
            WHERE metadata->>'source' = 'crm-benchmark'
            AND metadata->>'learning_key' = $1
            AND jsonb_typeof(metadata->'applied_student_learning') = 'object'
        """, learning_key)
        
        print(f"📊 Overall Statistics:")
        print(f"   Total Sessions: {total_sessions}")
        print(f"   Sessions with learning_state: {sessions_with_learning_state} ({sessions_with_learning_state/total_sessions*100:.1f}%)")
        print(f"   Sessions with applied_student_learning: {sessions_with_applied} ({sessions_with_applied/total_sessions*100:.1f}%)")
        print()
        
        # Status assessment
        if learning and (learning["student_learning"] or learning["teacher_learning"]):
            print("✅ Learning Synthesis: WORKING")
        else:
            print("⚠️  Learning Synthesis: NOT STARTED (may need more sessions)")
        
        if sessions_with_learning_state == total_sessions:
            print("✅ Learning State Persistence: WORKING (100%)")
        elif sessions_with_learning_state > 0:
            print(f"⚠️  Learning State Persistence: PARTIAL ({sessions_with_learning_state/total_sessions*100:.1f}%)")
        else:
            print("❌ Learning State Persistence: BROKEN")
        
        if sessions_with_applied > 0:
            print(f"✅ Applied Learning Persistence: WORKING ({sessions_with_applied} sessions)")
        else:
            print("⚠️  Applied Learning Persistence: NOT YET (resolve_playbook may not be called yet)")

        await conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(verify_learning_persistence())

