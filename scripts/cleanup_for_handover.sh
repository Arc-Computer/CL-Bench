#!/bin/bash
# Cleanup repository structure for customer handover
# Run from repository root: bash scripts/cleanup_for_handover.sh
#
# WARNING: This script moves directories to archive/ subdirectories.
#          If you have hardcoded paths in scripts/notebooks, they may break.
#          Use --dry-run to preview changes without making them.

set -e

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE (no changes will be made) ==="
fi

echo "=== Repository Cleanup for Handover ==="
echo ""

if [ "$DRY_RUN" = false ]; then
    echo "⚠️  WARNING: This will move directories to archive/ subdirectories."
    echo "   Hardcoded paths in your scripts may break."
    echo ""
    echo "   Run with --dry-run first to preview changes."
    echo "   Press Ctrl+C to cancel, or Enter to continue..."
    read
    echo ""
fi

# 1. Remove debug files
echo "1. Removing temporary debug files..."
rm -f atlas_payload_debug.txt
echo "   ✓ Removed atlas_payload_debug.txt"

# 2. Move backup files
echo ""
echo "2. Organizing backup files..."
mkdir -p artifacts/conversations_multi_turn/20251107T134304Z/backups
if ls artifacts/conversations_multi_turn/20251107T134304Z/full/*.backup 1> /dev/null 2>&1; then
    mv artifacts/conversations_multi_turn/20251107T134304Z/full/*.backup \
       artifacts/conversations_multi_turn/20251107T134304Z/backups/
    echo "   ✓ Moved .backup files to backups/ directory"
else
    echo "   ℹ No backup files to move"
fi

# 3. Archive old QA runs (with symlinks for compatibility)
echo ""
echo "3. Archiving old QA runs..."
if [ "$DRY_RUN" = true ]; then
    echo "   [DRY RUN] Would archive $(ls -d artifacts/qa/2025110[67]* 2>/dev/null | wc -l) old QA directories"
else
    mkdir -p artifacts/qa/archive
    QA_MOVED=0
    for dir in artifacts/qa/2025110[67]*; do
        if [ -d "$dir" ] && [ "$(basename $dir)" != "20251107T163953Z" ]; then
            dirname=$(basename "$dir")
            mv "$dir" artifacts/qa/archive/ 2>/dev/null || true
            # Create symlink for backward compatibility
            ln -sf "archive/$dirname" "artifacts/qa/$dirname" 2>/dev/null || true
            QA_MOVED=$((QA_MOVED + 1))
        fi
    done
    [ -d "artifacts/qa/final_validation_20251107T112353Z" ] && \
        mv artifacts/qa/final_validation_20251107T112353Z artifacts/qa/archive/ && \
        ln -sf "archive/final_validation_20251107T112353Z" artifacts/qa/final_validation_20251107T112353Z
    echo "   ✓ Archived $QA_MOVED old QA runs (symlinks created for compatibility)"
fi

# 4. Rename current QA for clarity
echo ""
echo "4. Renaming current QA directory..."
if [ -d "artifacts/qa/final_validation_full" ]; then
    mv artifacts/qa/final_validation_full artifacts/qa/dataset_validation_20251107
    echo "   ✓ Renamed to dataset_validation_20251107"
else
    echo "   ℹ Already renamed"
fi

# 5. Create LATEST pointers
echo ""
echo "5. Creating LATEST pointers..."
echo "20251107T134304Z" > artifacts/conversations_multi_turn/LATEST
echo "   ✓ Created conversations_multi_turn/LATEST"

echo "dataset_validation_20251107/20251107T163953Z" > artifacts/qa/LATEST
echo "   ✓ Created qa/LATEST"

echo "mock/baseline_20251107_patched.jsonl" > artifacts/baselines/LATEST
echo "   ✓ Created baselines/LATEST"

# 6. Organize baselines
echo ""
echo "6. Organizing baseline files..."
mkdir -p artifacts/baselines/mock
if [ -f "artifacts/baselines/mock_patched_20251107T111445Z.jsonl" ]; then
    cp artifacts/baselines/mock_patched_20251107T111445Z.jsonl \
       artifacts/baselines/mock/baseline_20251107_patched.jsonl
    echo "   ✓ Copied to mock/baseline_20251107_patched.jsonl"
else
    echo "   ℹ Baseline already organized"
fi

# 7. Archive old conversation runs
echo ""
echo "7. Archiving old conversation dataset runs..."
mkdir -p artifacts/conversations_multi_turn/archive
for dir in artifacts/conversations_multi_turn/20251105* artifacts/conversations_multi_turn/20251106*; do
    if [ -d "$dir" ]; then
        mv "$dir" artifacts/conversations_multi_turn/archive/
    fi
done
echo "   ✓ Archived old dataset runs"

# 8. Summary
echo ""
echo "=== Cleanup Complete ==="
echo ""
echo "Current artifacts structure:"
echo "  📁 Dataset:   artifacts/conversations_multi_turn/LATEST → $(cat artifacts/conversations_multi_turn/LATEST)"
echo "  📁 Baseline:  artifacts/baselines/LATEST → $(cat artifacts/baselines/LATEST)"
echo "  📁 QA:        artifacts/qa/LATEST → $(cat artifacts/qa/LATEST)"
echo ""
echo "Archived items:"
echo "  🗄️  Old datasets → artifacts/conversations_multi_turn/archive/"
echo "  🗄️  Old QA runs  → artifacts/qa/archive/"
echo "  🗄️  Backups      → artifacts/conversations_multi_turn/20251107T134304Z/backups/"
echo ""
echo "✓ Repository ready for handover"
