#!/bin/bash
# Monitor 1,000-conversation dataset generation
# Checks progress every 30 seconds and reports when complete

GENERATION_DIR="artifacts/conversations_multi_turn/20251108T041706Z_final"
EXPECTED_CHAINS=10

echo "Monitoring dataset generation in ${GENERATION_DIR}"
echo "Expected chains: ${EXPECTED_CHAINS}"
echo ""

while true; do
    # Count completed chains (those with chains.jsonl files)
    COMPLETED=$(find "${GENERATION_DIR}" -name "chains.jsonl" 2>/dev/null | wc -l | tr -d ' ')

    # Check if main generation process is still running
    if ! ps aux | grep -q "[g]enerate_1000_dataset.sh"; then
        echo "Generation script completed or stopped"
        break
    fi

    # Show progress
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[${TIMESTAMP}] Progress: ${COMPLETED}/${EXPECTED_CHAINS} chains completed"

    # Check for any chains.jsonl and report counts
    if [ ${COMPLETED} -gt 0 ]; then
        echo "  Generated conversations per chain:"
        find "${GENERATION_DIR}" -name "chains.jsonl" -exec sh -c 'echo "    $(basename $(dirname {})): $(wc -l < {}) conversations"' \;
    fi

    # Exit if all chains complete
    if [ ${COMPLETED} -eq ${EXPECTED_CHAINS} ]; then
        echo ""
        echo "=========================================="
        echo "ALL CHAINS COMPLETED!"
        echo "=========================================="

        # Calculate totals
        TOTAL_CONVERSATIONS=$(find "${GENERATION_DIR}" -name "chains.jsonl" -exec wc -l {} \; | awk '{sum+=$1} END {print sum}')
        echo "Total conversations generated: ${TOTAL_CONVERSATIONS}"
        echo ""
        echo "Next steps:"
        echo "  1. Verify output structure"
        echo "  2. Aggregate and filter to 5-10 turns"
        echo "  3. Execute validation pipeline"
        break
    fi

    sleep 30
done
