# 1,000-Conversation Dataset Pipeline Execution Plan

**Date**: 2025-11-07
**Generation Timestamp**: 20251108T015444Z
**Target**: 1,000 conversations (601 success / 399 failure)

## Current Status

**PHASE 1: GENERATION (IN PROGRESS)**
- Script: `scripts/generate_1000_dataset.sh`
- Output: `artifacts/conversations_multi_turn/20251108T015444Z_final/`
- Progress: Chain 1/10 (CHAIN-002A - 262 conversations)
- Model: gpt-4.1-mini

## Chains to Generate

| Chain ID | Type | Count | Workflows | Status |
|----------|------|-------|-----------|--------|
| CHAIN-002A | Success | 262 | client→opportunity→quote | In Progress |
| CHAIN-002B | Failure | 174 | client→opportunity[FAIL]→quote | Pending |
| CHAIN-003A | Success | 136 | contact→document→client | Pending |
| CHAIN-003B | Failure | 91 | contact→document[FAIL]→client | Pending |
| CHAIN-006A | Success | 73 | client_management | Pending |
| CHAIN-006B | Failure | 48 | client_management[FAIL] | Pending |
| CHAIN-007A | Success | 88 | contact→document | Pending |
| CHAIN-007B | Failure | 58 | contact→document[FAIL] | Pending |
| CHAIN-009A | Success | 42 | opportunity_summary→contract | Pending |
| CHAIN-009B | Failure | 28 | opportunity_summary→contract[FAIL] | Pending |

**Total**: 1,000 conversations

---

## PHASE 2: POST-GENERATION PIPELINE

### Step 5: Aggregation & Filtering

**Wait Condition**: All 10 chains complete (10 `chains.jsonl` files exist)

**Validation Before Aggregation**:
```bash
# Verify each chain output
for chain_dir in artifacts/conversations_multi_turn/20251108T015444Z_final/CHAIN-*/; do
    chain_id=$(basename "$chain_dir")
    conv_count=$(wc -l < "${chain_dir}/chains.jsonl")
    echo "${chain_id}: ${conv_count} conversations"

    # Sample first conversation to verify structure
    head -1 "${chain_dir}/chains.jsonl" | python -m json.tool > /dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: Invalid JSON in ${chain_id}"
        exit 1
    fi
done
```

**Aggregation Script** (to be written after validation):
```bash
#!/bin/bash
# scripts/aggregate_chains.sh

INPUT_DIR="artifacts/conversations_multi_turn/20251108T015444Z_final"
OUTPUT_FILE="${INPUT_DIR}/chains_aggregated.jsonl"

# Concatenate all chains
find "${INPUT_DIR}" -name "chains.jsonl" -exec cat {} \; > "${OUTPUT_FILE}"

# Verify total count
TOTAL=$(wc -l < "${OUTPUT_FILE}")
echo "Aggregated ${TOTAL} conversations (expected: 1000)"

if [ ${TOTAL} -ne 1000 ]; then
    echo "WARNING: Count mismatch!"
fi
```

**Filter to 5-10 Turns**:
```python
# scripts/filter_turn_length.py
import json
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

filtered = []
with open(input_file, 'r') as f:
    for line in f:
        conv = json.loads(line)
        turn_count = len(conv['turns'])
        if 5 <= turn_count <= 10:
            filtered.append(conv)

with open(output_file, 'w') as f:
    for conv in filtered:
        f.write(json.dumps(conv) + '\n')

print(f"Filtered {len(filtered)} conversations (5-10 turns)")
```

**Quality Gates**:
- ✓ All 1,000 conversations present
- ✓ Valid JSON structure
- ✓ Turn counts within 5-10 range
- ✓ Correct success/failure ratio (60%/40% ±2%)

---

### Step 6: Mock Baseline (Pre-Repair)

**Purpose**: Identify seed metadata gaps before repair

**Command**:
```bash
set -a && source .env && set +a

PYTHONPATH=. python -m src.evaluation.run_baseline \
  --conversations artifacts/conversations_multi_turn/20251108T015444Z_final/chains_eval_5to10.jsonl \
  --agent mock \
  --output artifacts/baselines/mock_prerepair_$(date -u +%Y%m%dT%H%M%SZ).jsonl \
  --no-judge
```

**Expected Outcome**:
- Some conversations will fail with metadata errors (Contact missing first_name, etc.)
- Capture failure count for comparison with post-repair run

**Quality Gate**:
- Document all metadata failures by type
- Verify failures match known patterns (first_name, opportunity name, quote opportunity_id)

---

### Step 7: Repair Seed Metadata

**Script**: `scripts/repair_seed_metadata.py` (already exists)

**Command**:
```bash
python scripts/repair_seed_metadata.py \
  --input artifacts/conversations_multi_turn/20251108T015444Z_final/chains_eval_5to10.jsonl \
  --output artifacts/conversations_multi_turn/20251108T015444Z_final/chains_eval_5to10_repaired.jsonl \
  --baseline-results artifacts/baselines/mock_prerepair_*.jsonl
```

**Verification**:
```python
# Spot-check repaired metadata
import json

with open('chains_eval_5to10_repaired.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 5: break
        conv = json.loads(line)

        # Check for required metadata fields
        for entity_id, metadata in conv.get('initial_entities', {}).items():
            if 'Contact' in entity_id:
                assert 'first_name' in metadata, f"Missing first_name in {entity_id}"
            if 'Opportunity' in entity_id:
                assert 'name' in metadata, f"Missing name in {entity_id}"
```

---

### Step 8: Mock Baseline (Post-Repair)

**Purpose**: Verify zero execution failures after repair

**Command**:
```bash
PYTHONPATH=. python -m src.evaluation.run_baseline \
  --conversations artifacts/conversations_multi_turn/20251108T015444Z_final/chains_eval_5to10_repaired.jsonl \
  --agent mock \
  --output artifacts/baselines/mock_postrepair_$(date -u +%Y%m%dT%H%M%SZ).jsonl \
  --no-judge
```

**Quality Gate**:
- **ZERO execution failures** (all conversations must execute successfully)
- If any failures exist, repair script must be debugged
- Compare pre-repair vs post-repair failure counts

---

### Step 9: Response Enrichment

**Script**: `scripts/enrich_from_baseline.py` (already exists)

**Command**:
```bash
python scripts/enrich_from_baseline.py \
  --conversations artifacts/conversations_multi_turn/20251108T015444Z_final/chains_eval_5to10_repaired.jsonl \
  --baseline artifacts/baselines/mock_postrepair_*.jsonl \
  --output artifacts/conversations_multi_turn/20251108T015444Z_final/chains_eval_5to10_enriched.jsonl
```

**Verification**:
```python
# Verify enrichment completeness
import json

empty_responses = 0
total_turns = 0

with open('chains_eval_5to10_enriched.jsonl', 'r') as f:
    for line in f:
        conv = json.loads(line)
        for turn in conv['turns']:
            total_turns += 1
            if not turn.get('expected_response', {}).get('text'):
                empty_responses += 1

print(f"Empty responses: {empty_responses}/{total_turns} ({empty_responses/total_turns*100:.2f}%)")
assert empty_responses == 0, "Enrichment incomplete!"
```

---

### Step 10: Judge Validation (Sample)

**Purpose**: Validate dataset difficulty with LLM judge

**Command**:
```bash
python analysis/dataset_judge.py \
  --dataset artifacts/conversations_multi_turn/20251108T015444Z_final/chains_eval_5to10_enriched.jsonl \
  --sample 100 \
  --model gpt-4.1-mini \
  --output artifacts/qa/judge_validation_$(date -u +%Y%m%dT%H%M%SZ)/
```

**Quality Gates**:
- Pass rate: 55-65% (validates difficulty)
- Zero conversation errors
- Balanced failure distribution across chains

---

### Step 11: Customer Baselines

**Script**: `scripts/run_customer_baselines.sh`

**Models**:
1. Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
2. GPT-4.1 (`gpt-4.1`)

**Command**:
```bash
bash scripts/run_customer_baselines.sh
```

**Dataset**: `chains_eval_5to10_enriched.jsonl` (post-enrichment)

**Quality Gates**:
- Both baselines complete successfully
- Results include judge evaluation data
- Success rates captured for each model

---

## Validation Checklist

Before declaring dataset complete:

- [ ] All 10 chains generated (1,000 conversations)
- [ ] Aggregated and filtered to 5-10 turns
- [ ] Mock baseline pre-repair identifies metadata gaps
- [ ] Repair script fixes all metadata issues
- [ ] Mock baseline post-repair shows ZERO execution failures
- [ ] Enrichment completes with 0% empty expected_response fields
- [ ] Judge validation shows 55-65% pass rate on 100-sample
- [ ] Customer baselines complete for both Claude and GPT-4.1
- [ ] Task distribution matches Agent_tasks.csv weights (±5%)

---

## Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Generation | 45-90 min | All 10 chains (currently in progress) |
| Aggregation | 2 min | Concatenate + filter |
| Mock baseline (pre) | 10 min | Identify metadata gaps |
| Repair | 1 min | Fix metadata |
| Mock baseline (post) | 10 min | Verify fixes |
| Enrichment | 2 min | Populate expected responses |
| Judge validation | 15 min | 100-conversation sample |
| Customer baselines | 60 min | Claude + GPT-4.1 |
| **TOTAL** | **~135-180 min** | From generation start to final baselines |

---

## Critical Success Factors

1. **Zero Execution Failures**: Post-repair mock baseline must succeed 100%
2. **Complete Enrichment**: No empty expected_response.text fields
3. **Judge Pass Rate**: 55-65% validates appropriate difficulty
4. **Task Coverage**: Distribution matches Agent_tasks.csv weights

## Risk Mitigation

**Risk**: Curator generates conversations outside 5-10 turn range
**Mitigation**: Filter step removes outliers

**Risk**: Metadata repair incomplete
**Mitigation**: Post-repair mock baseline gates progression

**Risk**: Judge pass rate out of range
**Mitigation**: Sample validation before full customer baseline execution

**Risk**: Customer baseline failures due to argument format mismatch
**Mitigation**: This is a KNOWN ISSUE - see investigation in previous session
