#!/bin/bash
# Master pipeline orchestration script
# Executes Steps 5-11 of the dataset generation pipeline
#
# Prerequisites: All 10 chains must be generated
# Quality gates enforced at each step

set -e  # Exit on error

# Configuration
GENERATION_DIR="${1:-artifacts/conversations_multi_turn/20251108T015444Z_final}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
PIPELINE_LOG="artifacts/pipeline_execution_${TIMESTAMP}.log"

# Colors for output
RED='\033[0:31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$PIPELINE_LOG"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$PIPELINE_LOG"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$PIPELINE_LOG"
}

check_quality_gate() {
    local gate_name="$1"
    local condition="$2"

    if [ "$condition" -ne 0 ]; then
        error "Quality gate failed: ${gate_name}"
    fi
    log "✓ Quality gate passed: ${gate_name}"
}

# Load environment
set -a
source .env >/dev/null 2>&1
set +a

log "==========================================================================="
log "POST-GENERATION PIPELINE EXECUTION"
log "==========================================================================="
log "Generation directory: ${GENERATION_DIR}"
log "Pipeline log: ${PIPELINE_LOG}"
log ""

# Verify generation directory exists
if [ ! -d "${GENERATION_DIR}" ]; then
    error "Generation directory not found: ${GENERATION_DIR}"
fi

# ============================================================================
# STEP 5: AGGREGATE AND FILTER
# ============================================================================

log "==========================================================================="
log "STEP 5: Aggregate all chains and filter to 5-10 turns"
log "==========================================================================="

AGGREGATED_FILE="${GENERATION_DIR}/chains_aggregated.jsonl"
FILTERED_FILE="${GENERATION_DIR}/chains_eval_5to10.jsonl"

python scripts/aggregate_and_filter.py \
    --input-dir "${GENERATION_DIR}" \
    --output-aggregated "${AGGREGATED_FILE}" \
    --output-filtered "${FILTERED_FILE}" \
    --min-turns 5 \
    --max-turns 10 \
    2>&1 | tee -a "$PIPELINE_LOG"

# Quality gate: Verify filtered file exists and has conversations
if [ ! -f "${FILTERED_FILE}" ]; then
    error "Filtered file not created"
fi

CONV_COUNT=$(wc -l < "${FILTERED_FILE}")
if [ "${CONV_COUNT}" -lt 100 ]; then
    error "Too few conversations after filtering: ${CONV_COUNT}"
fi

log "✓ Step 5 complete: ${CONV_COUNT} conversations (5-10 turns)"
log ""

# ============================================================================
# STEP 6: MOCK BASELINE (PRE-REPAIR)
# ============================================================================

log "==========================================================================="
log "STEP 6: Mock baseline (pre-repair) - Identify metadata gaps"
log "==========================================================================="

BASELINE_PREREPAIR="artifacts/baselines/mock_prerepair_${TIMESTAMP}.jsonl"

PYTHONPATH=. python -m src.evaluation.run_baseline \
    --conversations "${FILTERED_FILE}" \
    --agent mock \
    --output "${BASELINE_PREREPAIR}" \
    --no-judge \
    2>&1 | tee -a "$PIPELINE_LOG"

# Count failures
PREREPAIR_FAILURES=$(grep -c "failed" "${BASELINE_PREREPAIR}" || echo "0")
log "Pre-repair failures: ${PREREPAIR_FAILURES}"

if [ "${PREREPAIR_FAILURES}" -eq 0 ]; then
    warn "No metadata failures detected - repair may not be necessary"
fi

log "✓ Step 6 complete: Pre-repair baseline captured"
log ""

# ============================================================================
# STEP 7: REPAIR SEED METADATA
# ============================================================================

log "==========================================================================="
log "STEP 7: Repair seed metadata"
log "==========================================================================="

REPAIRED_FILE="${GENERATION_DIR}/chains_eval_5to10_repaired.jsonl"

python scripts/repair_seed_metadata.py \
    --input "${FILTERED_FILE}" \
    --output "${REPAIRED_FILE}" \
    --baseline-results "${BASELINE_PREREPAIR}" \
    2>&1 | tee -a "$PIPELINE_LOG"

# Quality gate: Verify repaired file exists
if [ ! -f "${REPAIRED_FILE}" ]; then
    error "Repaired file not created"
fi

REPAIRED_COUNT=$(wc -l < "${REPAIRED_FILE}")
if [ "${REPAIRED_COUNT}" -ne "${CONV_COUNT}" ]; then
    error "Conversation count mismatch after repair: ${REPAIRED_COUNT} vs ${CONV_COUNT}"
fi

log "✓ Step 7 complete: Metadata repaired"
log ""

# ============================================================================
# STEP 8: MOCK BASELINE (POST-REPAIR) - ZERO FAILURES REQUIRED
# ============================================================================

log "==========================================================================="
log "STEP 8: Mock baseline (post-repair) - CRITICAL QUALITY GATE"
log "==========================================================================="

BASELINE_POSTREPAIR="artifacts/baselines/mock_postrepair_${TIMESTAMP}.jsonl"

PYTHONPATH=. python -m src.evaluation.run_baseline \
    --conversations "${REPAIRED_FILE}" \
    --agent mock \
    --output "${BASELINE_POSTREPAIR}" \
    --no-judge \
    2>&1 | tee -a "$PIPELINE_LOG"

# CRITICAL QUALITY GATE: Zero execution failures
POSTREPAIR_FAILURES=$(grep -c "\"overall_success\": false" "${BASELINE_POSTREPAIR}" || echo "0")
log "Post-repair failures: ${POSTREPAIR_FAILURES}"

if [ "${POSTREPAIR_FAILURES}" -ne 0 ]; then
    error "CRITICAL: Post-repair baseline has ${POSTREPAIR_FAILURES} failures - repair incomplete!"
fi

log "✓ Step 8 complete: ZERO execution failures confirmed"
log ""

# ============================================================================
# STEP 9: ENRICH EXPECTED RESPONSES
# ============================================================================

log "==========================================================================="
log "STEP 9: Enrich expected responses from baseline results"
log "==========================================================================="

ENRICHED_FILE="${GENERATION_DIR}/chains_eval_5to10_enriched.jsonl"

python scripts/enrich_from_baseline.py \
    --conversations "${REPAIRED_FILE}" \
    --baseline "${BASELINE_POSTREPAIR}" \
    --output "${ENRICHED_FILE}" \
    2>&1 | tee -a "$PIPELINE_LOG"

# Quality gate: Verify no empty expected_response fields
log "Verifying enrichment completeness..."
EMPTY_RESPONSES=$(python3 << 'EOF'
import json
import sys

empty_count = 0
total_turns = 0

with open(sys.argv[1], 'r') as f:
    for line in f:
        conv = json.loads(line)
        for turn in conv['turns']:
            total_turns += 1
            if not turn.get('expected_response', {}).get('text'):
                empty_count += 1

print(empty_count)
EOF
"${ENRICHED_FILE}"
)

if [ "${EMPTY_RESPONSES}" -ne 0 ]; then
    error "Enrichment incomplete: ${EMPTY_RESPONSES} empty expected_response fields found"
fi

log "✓ Step 9 complete: All expected_response fields populated"
log ""

# ============================================================================
# STEP 10: JUDGE VALIDATION (100-conversation sample)
# ============================================================================

log "==========================================================================="
log "STEP 10: Judge validation on 100-conversation sample"
log "==========================================================================="

JUDGE_OUTPUT="artifacts/qa/judge_validation_${TIMESTAMP}"
mkdir -p "${JUDGE_OUTPUT}"

python analysis/dataset_judge.py \
    --dataset "${ENRICHED_FILE}" \
    --sample 100 \
    --model gpt-4.1-mini \
    --output "${JUDGE_OUTPUT}/" \
    2>&1 | tee -a "$PIPELINE_LOG"

# Quality gate: Pass rate 55-65%
if [ -f "${JUDGE_OUTPUT}/summary.json" ]; then
    PASS_RATE=$(python3 -c "import json; print(json.load(open('${JUDGE_OUTPUT}/summary.json'))['pass_rate'])")
    log "Judge pass rate: $(python3 -c "print(f'{${PASS_RATE}:.2%}')")"

    # Check if pass rate is within target range (0.55 to 0.65)
    PASS_CHECK=$(python3 -c "print(1 if 0.55 <= ${PASS_RATE} <= 0.65 else 0)")
    if [ "${PASS_CHECK}" -eq 0 ]; then
        warn "Judge pass rate out of target range (55-65%): ${PASS_RATE}"
    else
        log "✓ Pass rate within target range"
    fi
else
    warn "Judge summary file not found - skipping pass rate validation"
fi

log "✓ Step 10 complete: Judge validation finished"
log ""

# ============================================================================
# STEP 11: CUSTOMER BASELINES
# ============================================================================

log "==========================================================================="
log "STEP 11: Execute customer baselines (Claude Sonnet 4.5 + GPT-4.1)"
log "==========================================================================="

# Update the customer baseline script to use the enriched dataset
BASELINE_TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
BASELINE_OUTPUT="artifacts/baselines/customer_${BASELINE_TIMESTAMP}"
mkdir -p "${BASELINE_OUTPUT}"

log "Running Claude Sonnet 4.5 baseline..."
PYTHONPATH=. python -m src.evaluation.run_baseline \
    --conversations "${ENRICHED_FILE}" \
    --agent claude \
    --model claude-sonnet-4-5-20250929 \
    --output "${BASELINE_OUTPUT}/claude_sonnet_4_5.jsonl" \
    2>&1 | tee "${BASELINE_OUTPUT}/claude_sonnet_4_5.log"

CLAUDE_SUCCESS=$(grep "successes:" "${BASELINE_OUTPUT}/claude_sonnet_4_5.log" | tail -1 || echo "N/A")
log "Claude Sonnet 4.5: ${CLAUDE_SUCCESS}"

log "Running GPT-4.1 baseline..."
PYTHONPATH=. python -m src.evaluation.run_baseline \
    --conversations "${ENRICHED_FILE}" \
    --agent gpt4.1 \
    --model gpt-4.1 \
    --output "${BASELINE_OUTPUT}/gpt4_1.jsonl" \
    2>&1 | tee "${BASELINE_OUTPUT}/gpt4_1.log"

GPT_SUCCESS=$(grep "successes:" "${BASELINE_OUTPUT}/gpt4_1.log" | tail -1 || echo "N/A")
log "GPT-4.1: ${GPT_SUCCESS}"

log "✓ Step 11 complete: Customer baselines executed"
log ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================

log "==========================================================================="
log "PIPELINE EXECUTION COMPLETE"
log "==========================================================================="
log ""
log "Generated dataset: ${ENRICHED_FILE}"
log "Total conversations: ${CONV_COUNT}"
log "Pre-repair failures: ${PREREPAIR_FAILURES}"
log "Post-repair failures: ${POSTREPAIR_FAILURES} (MUST BE ZERO)"
log "Empty expected responses: ${EMPTY_RESPONSES} (MUST BE ZERO)"
log ""
log "Judge validation: ${JUDGE_OUTPUT}/"
log "Customer baselines: ${BASELINE_OUTPUT}/"
log ""
log "Complete pipeline log: ${PIPELINE_LOG}"
log "==========================================================================="
