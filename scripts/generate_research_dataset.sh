#!/bin/bash
set -e

# Research-Grade CRM Dataset Generation
# Generates 2,000 conversations with LLM natural language throughout
# - 1,500 multi-turn (5 chains with LLM-curated utterances)
# - 500 single-turn (weighted by Agent_tasks.csv)

set -a
source .env
set +a

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
BASE_DIR="artifacts/conversations_research/${TIMESTAMP}_2000_llm"
mkdir -p "${BASE_DIR}"

echo "=========================================================================="
echo "RESEARCH-GRADE DATASET GENERATION"
echo "=========================================================================="
echo ""
echo "Output: ${BASE_DIR}"
echo "Mode: CURATOR_SIMPLE_DATASET=0 (LLM natural language throughout)"
echo ""
echo "Plan:"
echo "  - 1,500 multi-turn conversations (5 chains)"
echo "  - 500 single-turn conversations (task-weighted)"
echo "  - 60/40 success/failure ratio"
echo ""

# ============================================================================
# PART 1: Multi-Turn Conversations (1,500 total)
# ============================================================================

echo "=========================================================================="
echo "PART 1: GENERATING 1,500 MULTI-TURN CONVERSATIONS"
echo "=========================================================================="
echo ""

# Distribution across 5 chains (scaled to 1,500):
# CHAIN-001: 300 (20%)
# CHAIN-002: 600 (40%)
# CHAIN-004: 225 (15%)
# CHAIN-005: 225 (15%)
# CHAIN-008: 150 (10%)

echo "[1/5] Generating CHAIN-001 (300 conversations)..."
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode chain \
  --chain-id CHAIN-001A \
  --chain-id CHAIN-001B \
  --count 300 \
  --seed 100 \
  --output-dir "${BASE_DIR}/multi_turn/CHAIN-001"

echo ""
echo "[2/5] Generating CHAIN-002 (600 conversations)..."
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode chain \
  --chain-id CHAIN-002A \
  --chain-id CHAIN-002B \
  --count 600 \
  --seed 200 \
  --output-dir "${BASE_DIR}/multi_turn/CHAIN-002"

echo ""
echo "[3/5] Generating CHAIN-004 (225 conversations)..."
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode chain \
  --chain-id CHAIN-004A \
  --chain-id CHAIN-004B \
  --count 225 \
  --seed 300 \
  --output-dir "${BASE_DIR}/multi_turn/CHAIN-004"

echo ""
echo "[4/5] Generating CHAIN-005 (225 conversations)..."
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode chain \
  --chain-id CHAIN-005A \
  --chain-id CHAIN-005B \
  --count 225 \
  --seed 400 \
  --output-dir "${BASE_DIR}/multi_turn/CHAIN-005"

echo ""
echo "[5/5] Generating CHAIN-008 (150 conversations)..."
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode chain \
  --chain-id CHAIN-008A \
  --chain-id CHAIN-008B \
  --count 150 \
  --seed 500 \
  --output-dir "${BASE_DIR}/multi_turn/CHAIN-008"

echo ""
echo "✓ Multi-turn generation complete (1,500 conversations)"
echo ""

# ============================================================================
# PART 2: Single-Turn Conversations (500 total)
# ============================================================================

echo "=========================================================================="
echo "PART 2: GENERATING 500 SINGLE-TURN CONVERSATIONS"
echo "=========================================================================="
echo ""

# Use workflow mode with single-workflow templates
# This gives us natural LLM-generated utterances for single operations
echo "Generating 500 single-turn conversations (task-weighted)..."
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode workflow \
  --count 500 \
  --success-ratio 0.6 \
  --seed 600 \
  --output-dir "${BASE_DIR}/single_turn"

echo ""
echo "✓ Single-turn generation complete (500 conversations)"
echo ""

# ============================================================================
# PART 3: Aggregate Dataset
# ============================================================================

echo "=========================================================================="
echo "PART 3: AGGREGATING DATASET"
echo "=========================================================================="
echo ""

echo "Aggregating all conversations..."
cat "${BASE_DIR}"/multi_turn/*/chains.jsonl "${BASE_DIR}"/single_turn/conversations.jsonl > "${BASE_DIR}/all_conversations.jsonl"

TOTAL_COUNT=$(wc -l < "${BASE_DIR}/all_conversations.jsonl" | tr -d ' ')
echo "✓ Aggregated ${TOTAL_COUNT} conversations"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "=========================================================================="
echo "GENERATION COMPLETE"
echo "=========================================================================="
echo ""
echo "Dataset: ${BASE_DIR}/all_conversations.jsonl"
echo "Total conversations: ${TOTAL_COUNT}"
echo ""
echo "Breakdown:"
echo "  Multi-turn: 1,500 conversations"
echo "  Single-turn: 500 conversations"
echo ""
echo "Next steps:"
echo "  1. Run LLM judge: PYTHONPATH=. python analysis/dataset_judge.py --dataset ${BASE_DIR}/all_conversations.jsonl --limit 100"
echo "  2. Run enrichment: bash scripts/execute_post_generation_pipeline.sh ${BASE_DIR}/all_conversations.jsonl"
echo "  3. Extract baseline sample"
echo "  4. Run baseline evaluations"
echo ""
