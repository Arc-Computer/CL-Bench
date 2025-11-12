#!/bin/bash
# Run all three baseline evaluations in parallel
# Safe because each uses isolated transactions and different API providers

set -e
set -a
source .env
set +a

OUTPUT_DIR="artifacts/evaluation"
DATASET="artifacts/deterministic/final_conversations_final_clean.jsonl"

echo "=========================================="
echo "Starting Parallel Baseline Evaluations"
echo "=========================================="
echo ""
echo "Dataset: $DATASET"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Starting 3 baseline agents in parallel..."
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start Claude baseline
echo "[1/3] Starting Claude 4.5 Sonnet baseline..."
python3 -m src.evaluation.run_baseline \
  --conversations "$DATASET" \
  --agent claude \
  --model claude-sonnet-4-5-20250929 \
  --backend postgres \
  --output "$OUTPUT_DIR/baseline_claude_sonnet_4_5.jsonl" \
  --temperature 0.0 \
  --max-output-tokens 800 > "$OUTPUT_DIR/baseline_claude.log" 2>&1 &
CLAUDE_PID=$!
echo "  Claude PID: $CLAUDE_PID"
echo "  Log: $OUTPUT_DIR/baseline_claude.log"

# Start GPT-4.1 baseline
echo "[2/3] Starting GPT-4.1 baseline..."
python3 -m src.evaluation.run_baseline \
  --conversations "$DATASET" \
  --agent gpt4.1 \
  --model gpt-4.1 \
  --backend postgres \
  --output "$OUTPUT_DIR/baseline_gpt4_1.jsonl" \
  --temperature 0.0 \
  --max-output-tokens 800 > "$OUTPUT_DIR/baseline_gpt4_1.log" 2>&1 &
GPT4_PID=$!
echo "  GPT-4.1 PID: $GPT4_PID"
echo "  Log: $OUTPUT_DIR/baseline_gpt4_1.log"

# Start GPT-4.1 mini baseline
echo "[3/3] Starting GPT-4.1 mini baseline..."
python3 -m src.evaluation.run_baseline \
  --conversations "$DATASET" \
  --agent gpt4.1 \
  --model gpt-4.1-mini \
  --backend postgres \
  --output "$OUTPUT_DIR/baseline_gpt4_1_mini.jsonl" \
  --temperature 0.0 \
  --max-output-tokens 800 > "$OUTPUT_DIR/baseline_gpt4_1_mini.log" 2>&1 &
GPT4MINI_PID=$!
echo "  GPT-4.1 Mini PID: $GPT4MINI_PID"
echo "  Log: $OUTPUT_DIR/baseline_gpt4_1_mini.log"

echo ""
echo "=========================================="
echo "All baselines started. Monitoring progress..."
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  tail -f $OUTPUT_DIR/baseline_claude.log"
echo "  tail -f $OUTPUT_DIR/baseline_gpt4_1.log"
echo "  tail -f $OUTPUT_DIR/baseline_gpt4_1_mini.log"
echo ""
echo "Or use the progress monitor:"
echo "  python3 scripts/monitor_evaluation_progress.py $OUTPUT_DIR"
echo ""

# Wait for all processes
wait $CLAUDE_PID
CLAUDE_EXIT=$?
wait $GPT4_PID
GPT4_EXIT=$?
wait $GPT4MINI_PID
GPT4MINI_EXIT=$?

echo ""
echo "=========================================="
echo "All Baselines Completed"
echo "=========================================="
echo ""
echo "Exit codes:"
echo "  Claude 4.5 Sonnet: $CLAUDE_EXIT"
echo "  GPT-4.1:           $GPT4_EXIT"
echo "  GPT-4.1 Mini:       $GPT4MINI_EXIT"
echo ""

if [ $CLAUDE_EXIT -eq 0 ] && [ $GPT4_EXIT -eq 0 ] && [ $GPT4MINI_EXIT -eq 0 ]; then
    echo "✅ All baselines completed successfully!"
    echo ""
    echo "Next step: Run Atlas evaluation"
    echo "  python3 scripts/run_atlas_evaluation.py \\"
    echo "    --conversations $DATASET \\"
    echo "    --config configs/atlas/crm_harness.yaml \\"
    echo "    --output-dir $OUTPUT_DIR/atlas_full"
    exit 0
else
    echo "❌ Some baselines failed. Check logs:"
    [ $CLAUDE_EXIT -ne 0 ] && echo "  - Claude: $OUTPUT_DIR/baseline_claude.log"
    [ $GPT4_EXIT -ne 0 ] && echo "  - GPT-4.1: $OUTPUT_DIR/baseline_gpt4_1.log"
    [ $GPT4MINI_EXIT -ne 0 ] && echo "  - GPT-4.1 Mini: $OUTPUT_DIR/baseline_gpt4_1_mini.log"
    exit 1
fi

