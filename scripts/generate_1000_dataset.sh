#!/bin/bash
# Generate 1,000-conversation dataset with proper task-weight distribution
# Based on configs/task_chain_mapping.json

set -e

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT="artifacts/conversations_multi_turn/${TIMESTAMP}_final"
MODEL="gpt-4.1-mini"

echo "========================================================================"
echo "GENERATING 1,000-CONVERSATION DATASET"
echo "========================================================================"
echo "Timestamp: ${TIMESTAMP}"
echo "Output: ${OUTPUT_ROOT}"
echo "Model: ${MODEL}"
echo "Success/Failure Ratio: 60% / 40%"
echo "========================================================================"
echo ""

# Load environment
set -a
source .env >/dev/null 2>&1
set +a

# Disable per-chain failure ratio validation (we validate at aggregation step)
export DISABLE_CHAIN_FAILURE_RATIO=1

# Create output directory
mkdir -p "${OUTPUT_ROOT}"

# Track totals
TOTAL_SUCCESS=0
TOTAL_FAILURE=0

echo "Starting Curator generation for all 10 chain variants..."
echo ""

# CHAIN-002: Opportunity + Quote workflows (436 total: 262 success + 174 failure)
echo "──────────────────────────────────────────────────────────────────────"
echo "[1/10] CHAIN-002A (Success): 262 conversations"
echo "       Workflows: client_management → opportunity_management → quote_generation"
echo "──────────────────────────────────────────────────────────────────────"
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode chain \
  --count 262 \
  --seed 42 \
  --model-name "${MODEL}" \
  --chain-id CHAIN-002A \
  --output-dir "${OUTPUT_ROOT}/CHAIN-002A" \
  2>&1 | tee "${OUTPUT_ROOT}/CHAIN-002A_generation.log"
TOTAL_SUCCESS=$((TOTAL_SUCCESS + 262))
echo "✓ CHAIN-002A complete"
echo ""

echo "──────────────────────────────────────────────────────────────────────"
echo "[2/10] CHAIN-002B (Failure): 174 conversations"
echo "       Workflows: client_management → opportunity_management[FAIL] → quote_generation"
echo "──────────────────────────────────────────────────────────────────────"
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode chain \
  --count 174 \
  --seed 43 \
  --model-name "${MODEL}" \
  --chain-id CHAIN-002B \
  --output-dir "${OUTPUT_ROOT}/CHAIN-002B" \
  2>&1 | tee "${OUTPUT_ROOT}/CHAIN-002B_generation.log"
TOTAL_FAILURE=$((TOTAL_FAILURE + 174))
echo "✓ CHAIN-002B complete"
echo ""

# CHAIN-003: Contact + Document workflows (227 total: 136 success + 91 failure)
echo "──────────────────────────────────────────────────────────────────────"
echo "[3/10] CHAIN-003A (Success): 136 conversations"
echo "       Workflows: contact_management → document_workflow → client_management"
echo "──────────────────────────────────────────────────────────────────────"
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode chain \
  --count 136 \
  --seed 44 \
  --model-name "${MODEL}" \
  --chain-id CHAIN-003A \
  --output-dir "${OUTPUT_ROOT}/CHAIN-003A" \
  2>&1 | tee "${OUTPUT_ROOT}/CHAIN-003A_generation.log"
TOTAL_SUCCESS=$((TOTAL_SUCCESS + 136))
echo "✓ CHAIN-003A complete"
echo ""

echo "──────────────────────────────────────────────────────────────────────"
echo "[4/10] CHAIN-003B (Failure): 91 conversations"
echo "       Workflows: contact_management → document_workflow[FAIL] → client_management"
echo "──────────────────────────────────────────────────────────────────────"
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode chain \
  --count 91 \
  --seed 45 \
  --model-name "${MODEL}" \
  --chain-id CHAIN-003B \
  --output-dir "${OUTPUT_ROOT}/CHAIN-003B" \
  2>&1 | tee "${OUTPUT_ROOT}/CHAIN-003B_generation.log"
TOTAL_FAILURE=$((TOTAL_FAILURE + 91))
echo "✓ CHAIN-003B complete"
echo ""

# CHAIN-006: Client Management (121 total: 73 success + 48 failure)
echo "──────────────────────────────────────────────────────────────────────"
echo "[5/10] CHAIN-006A (Success): 73 conversations"
echo "       Workflows: client_management"
echo "──────────────────────────────────────────────────────────────────────"
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode chain \
  --count 73 \
  --seed 46 \
  --model-name "${MODEL}" \
  --chain-id CHAIN-006A \
  --output-dir "${OUTPUT_ROOT}/CHAIN-006A" \
  2>&1 | tee "${OUTPUT_ROOT}/CHAIN-006A_generation.log"
TOTAL_SUCCESS=$((TOTAL_SUCCESS + 73))
echo "✓ CHAIN-006A complete"
echo ""

echo "──────────────────────────────────────────────────────────────────────"
echo "[6/10] CHAIN-006B (Failure): 48 conversations"
echo "       Workflows: client_management[FAIL]"
echo "──────────────────────────────────────────────────────────────────────"
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode chain \
  --count 48 \
  --seed 47 \
  --model-name "${MODEL}" \
  --chain-id CHAIN-006B \
  --output-dir "${OUTPUT_ROOT}/CHAIN-006B" \
  2>&1 | tee "${OUTPUT_ROOT}/CHAIN-006B_generation.log"
TOTAL_FAILURE=$((TOTAL_FAILURE + 48))
echo "✓ CHAIN-006B complete"
echo ""

# CHAIN-007: Contact + Document (146 total: 88 success + 58 failure)
echo "──────────────────────────────────────────────────────────────────────"
echo "[7/10] CHAIN-007A (Success): 88 conversations"
echo "       Workflows: contact_management → document_workflow"
echo "──────────────────────────────────────────────────────────────────────"
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode chain \
  --count 88 \
  --seed 48 \
  --model-name "${MODEL}" \
  --chain-id CHAIN-007A \
  --output-dir "${OUTPUT_ROOT}/CHAIN-007A" \
  2>&1 | tee "${OUTPUT_ROOT}/CHAIN-007A_generation.log"
TOTAL_SUCCESS=$((TOTAL_SUCCESS + 88))
echo "✓ CHAIN-007A complete"
echo ""

echo "──────────────────────────────────────────────────────────────────────"
echo "[8/10] CHAIN-007B (Failure): 58 conversations"
echo "       Workflows: contact_management → document_workflow[FAIL]"
echo "──────────────────────────────────────────────────────────────────────"
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode chain \
  --count 58 \
  --seed 49 \
  --model-name "${MODEL}" \
  --chain-id CHAIN-007B \
  --output-dir "${OUTPUT_ROOT}/CHAIN-007B" \
  2>&1 | tee "${OUTPUT_ROOT}/CHAIN-007B_generation.log"
TOTAL_FAILURE=$((TOTAL_FAILURE + 58))
echo "✓ CHAIN-007B complete"
echo ""

# CHAIN-009: Opportunity Summary + Contract (70 total: 42 success + 28 failure)
echo "──────────────────────────────────────────────────────────────────────"
echo "[9/10] CHAIN-009A (Success): 42 conversations"
echo "       Workflows: opportunity_summary → contract_review"
echo "──────────────────────────────────────────────────────────────────────"
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode chain \
  --count 42 \
  --seed 50 \
  --model-name "${MODEL}" \
  --chain-id CHAIN-009A \
  --output-dir "${OUTPUT_ROOT}/CHAIN-009A" \
  2>&1 | tee "${OUTPUT_ROOT}/CHAIN-009A_generation.log"
TOTAL_SUCCESS=$((TOTAL_SUCCESS + 42))
echo "✓ CHAIN-009A complete"
echo ""

echo "──────────────────────────────────────────────────────────────────────"
echo "[10/10] CHAIN-009B (Failure): 28 conversations"
echo "        Workflows: opportunity_summary → contract_review[FAIL]"
echo "──────────────────────────────────────────────────────────────────────"
PYTHONPATH=. CURATOR_SIMPLE_DATASET=0 python scripts/generate_conversations.py \
  --mode chain \
  --count 28 \
  --seed 51 \
  --model-name "${MODEL}" \
  --chain-id CHAIN-009B \
  --output-dir "${OUTPUT_ROOT}/CHAIN-009B" \
  2>&1 | tee "${OUTPUT_ROOT}/CHAIN-009B_generation.log"
TOTAL_FAILURE=$((TOTAL_FAILURE + 28))
echo "✓ CHAIN-009B complete"
echo ""

# Calculate final statistics
TOTAL_CONVERSATIONS=$((TOTAL_SUCCESS + TOTAL_FAILURE))
SUCCESS_RATIO=$(echo "scale=4; ${TOTAL_SUCCESS} / ${TOTAL_CONVERSATIONS} * 100" | bc)
FAILURE_RATIO=$(echo "scale=4; ${TOTAL_FAILURE} / ${TOTAL_CONVERSATIONS} * 100" | bc)

echo "========================================================================"
echo "GENERATION COMPLETE"
echo "========================================================================"
echo "Total Conversations: ${TOTAL_CONVERSATIONS}"
echo "  Success: ${TOTAL_SUCCESS} (${SUCCESS_RATIO}%)"
echo "  Failure: ${TOTAL_FAILURE} (${FAILURE_RATIO}%)"
echo ""
echo "Output Directory: ${OUTPUT_ROOT}"
echo "========================================================================"

# List all generated files
echo ""
echo "Generated files:"
find "${OUTPUT_ROOT}" -name "chains.jsonl" | sort

echo ""
echo "Next steps:"
echo "  1. Aggregate all chains.jsonl files"
echo "  2. Filter to 5-10 turn conversations"
echo "  3. Run mock baseline (pre-repair)"
echo "  4. Repair seed metadata"
echo "  5. Run mock baseline (post-repair)"
echo "  6. Enrich expected responses"
echo "  7. Validate with judge"
echo "  8. Execute customer baselines"
