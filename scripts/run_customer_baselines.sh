#!/bin/bash
# Customer Baseline Execution Script
# Runs Claude Sonnet 4.5 and GPT-4.1 on the final 277-conversation dataset

set -e  # Exit on error

DATASET="artifacts/conversations_multi_turn/20251107T134304Z/full/chains_eval_enriched_5to10_repaired.jsonl"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)

echo "=========================================="
echo "Customer Baseline Execution"
echo "Dataset: ${DATASET}"
echo "Timestamp: ${TIMESTAMP}"
echo "=========================================="
echo ""

# Load environment
set -a
source .env >/dev/null 2>&1
set +a

# Create output directory
mkdir -p artifacts/baselines/customer_${TIMESTAMP}

echo "Starting Claude Sonnet 4.5 baseline..."
PYTHONPATH=. python -m src.evaluation.run_baseline \
  --conversations "${DATASET}" \
  --agent claude \
  --model claude-sonnet-4-5-20250929 \
  --output "artifacts/baselines/customer_${TIMESTAMP}/claude_sonnet_4_5.jsonl" \
  2>&1 | tee "artifacts/baselines/customer_${TIMESTAMP}/claude_sonnet_4_5.log"

echo ""
echo "Claude Sonnet 4.5 complete!"
echo ""

echo "Starting GPT-4.1 baseline..."
PYTHONPATH=. python -m src.evaluation.run_baseline \
  --conversations "${DATASET}" \
  --agent gpt4.1 \
  --model gpt-4.1 \
  --output "artifacts/baselines/customer_${TIMESTAMP}/gpt4_1.jsonl" \
  2>&1 | tee "artifacts/baselines/customer_${TIMESTAMP}/gpt4_1.log"

echo ""
echo "GPT-4.1 complete!"
echo ""

# Generate summary
echo "=========================================="
echo "Baseline Execution Complete"
echo "Output directory: artifacts/baselines/customer_${TIMESTAMP}/"
echo ""
echo "Files generated:"
ls -lh "artifacts/baselines/customer_${TIMESTAMP}/"
echo ""

# Extract success rates
CLAUDE_SUCCESS=$(grep "successes:" "artifacts/baselines/customer_${TIMESTAMP}/claude_sonnet_4_5.log" | tail -1 || echo "N/A")
GPT_SUCCESS=$(grep "successes:" "artifacts/baselines/customer_${TIMESTAMP}/gpt4_1.log" | tail -1 || echo "N/A")

echo "Results summary:"
echo "  Claude Sonnet 4.5: ${CLAUDE_SUCCESS}"
echo "  GPT-4.1: ${GPT_SUCCESS}"
echo "=========================================="
