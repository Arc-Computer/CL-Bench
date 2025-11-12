#!/bin/bash
# Restart GPT-4.1 and GPT-4.1 mini baselines with fixed configuration

set -e

cd "$(dirname "$0")/.."

# Load environment variables
set -a
source .env
set +a

# Ensure output directory exists
mkdir -p artifacts/evaluation

echo "=========================================="
echo "Restarting GPT Baselines with Fixed Config"
echo "=========================================="
echo ""

# Start GPT-4.1 baseline
echo "Starting GPT-4.1 baseline..."
nohup python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent gpt4.1 \
  --model gpt-4.1 \
  --backend postgres \
  --output artifacts/evaluation/baseline_gpt4_1.jsonl \
  --temperature 0.0 \
  --max-output-tokens 800 \
  > artifacts/evaluation/baseline_gpt4_1.log 2>&1 &

GPT4_1_PID=$!
echo "  GPT-4.1 PID: $GPT4_1_PID"

# Start GPT-4.1 mini baseline
echo "Starting GPT-4.1 mini baseline..."
nohup python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent gpt4.1 \
  --model gpt-4.1-mini \
  --backend postgres \
  --output artifacts/evaluation/baseline_gpt4_1_mini.jsonl \
  --temperature 0.0 \
  --max-output-tokens 800 \
  > artifacts/evaluation/baseline_gpt4_1_mini.log 2>&1 &

GPT4_1_MINI_PID=$!
echo "  GPT-4.1 mini PID: $GPT4_1_MINI_PID"

echo ""
echo "✅ Both GPT baselines restarted with fixed configuration"
echo ""
echo "Monitor progress:"
echo "  tail -f artifacts/evaluation/baseline_gpt4_1.log"
echo "  tail -f artifacts/evaluation/baseline_gpt4_1_mini.log"
echo ""
echo "Check status:"
echo "  python3 scripts/check_baseline_status.py"
