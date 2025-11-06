# Chained Workflows Documentation

This document describes the chained workflow generation system for multi-segment CRM conversations.

## Overview

Chained workflows enable generation of conversations that span multiple workflow segments, with proper entity state propagation and cumulative context tracking. This allows for realistic multi-turn conversations that mirror production CRM usage patterns.

## Architecture

### Workflow Chains

A `WorkflowChain` defines a sequence of workflow templates that form a multi-segment conversation:

```python
from src.conversation_templates import WorkflowChain, WORKFLOW_CHAINS

chain = WORKFLOW_CHAINS["onboarding_pipeline_contract_success"]
# Chain contains:
# - workflow_sequence: ["client_onboarding", "deal_pipeline", "quote_generation"]
# - success_pattern: [True, True, True]
# - entity_handoff_rules: {"client_id": "propagate", ...}
```

Every production workflow has a paired failure-bearing variant (suffix `_failure`). When you need both, use `expand_chain_ids(["onboarding_pipeline_contract"])` to expand the legacy alias into discrete success/failure entries.

### New Template Coverage (Phase 5)

- **WF-009 Opportunity Summary** – surfaces `summarize_opportunities` and `view_opportunity_details` after a client/opportunity search.
- **WF-010 Quote Cleanup** – applies `quote_search`, `cancel_quote`, and `delete_quote` to retire outdated quotes.
- **WF-011 Contract Review** – runs `contract_search`, uploads collateral to the contract record, and captures a compliance note.
- **WF-012 Opportunity Clone** – clones an active opportunity, advances the duplicate, and attaches supporting documents.

New chain aliases pair these templates with existing flows:
- `quote_remediation` – `quote_generation` → `quote_cleanup`
- `summary_contract` – `opportunity_summary` → `contract_review`
- `clone_expansion` – `opportunity_management` → `opportunity_clone`

### Curator Structured Output

The chained generator exposes Bespoke Curator outputs through explicit Pydantic models:

- `TurnMetadata` – stage/persona hints and handoff dependencies per turn.
- `ScenarioSelectionResponse` – deterministic scenario picks with justifications and handoff actions.
- `TurnUtteranceResponse` – user utterances, stage focus, and referenced entity hints.
- `ChainSegmentSummary` / `SegmentContext` – segment-level summaries used for downstream analytics.

```python
from src.generation.curator_chain_models import (
    ScenarioSelectionResponse,
    TurnMetadata,
    TurnUtteranceResponse,
)

class ScenarioSelector(curator.LLM):
    response_format = ScenarioSelectionResponse
    # prompt(...) renders workflow + turn metadata
    # parse(...) returns rows such as {"scenario_id": "SC-010", ...}

class ChainUtteranceGenerator(curator.LLM):
    response_format = TurnUtteranceResponse
    # prompt(...) consumes turn metadata + cumulative context
    # parse(...) yields user utterances plus optional persona/stage annotations
```

These schemas keep Curator responses fully structured and ensure row-level determinism when replayed.

### Segment & Turn Metadata

`instantiate_chained_conversation` surfaces rich metadata alongside the canonical `Conversation` object:

- `conversation.segment_boundaries` – turn indices where segments end.
- `conversation.cumulative_context["segment_summaries"]` – per-segment payloads (expected/actual outcome, entities created, handoff trace).
- `conversation.cumulative_context["turn_annotations"]` – turn-level persona, stage focus, referenced entity hints, and scenario IDs.

This context powers deterministic handoffs, Atlas analytics, and end-to-end harness validation without re-running Curator.

### Generation Process

1. **Scenario Selection**: `ScenarioSelector` selects scenarios for each turn
2. **Utterance Generation**: `ChainUtteranceGenerator` generates natural language utterances
3. **Entity Propagation**: Entity state carries across segments using `cumulative_context`
4. **Template Resolution**: Cross-turn references (`{{turn_N.field}}`) resolve correctly

Setting `CURATOR_SIMPLE_DATASET=1` activates the offline stub pipeline: scenario selection and utterance generation fall back to deterministic logic built from validated scenario data (no API calls, no fallback utterances).

## Usage

### Generating Chained Conversations

```python
from src.generation.chain_conversation_generator import instantiate_chained_conversation
from src.generation.chain_curator import ScenarioSelector, ChainUtteranceGenerator
from src.conversation_templates import WORKFLOW_CHAINS
from src.pipeline.scenario_repository import ScenarioRepository
import random
import os

repo = ScenarioRepository.from_default_paths()
offline = os.environ.get("CURATOR_SIMPLE_DATASET") == "1"
scenario_selector = None if offline else ScenarioSelector(model_name="gpt-4.1-mini")
utterance_generator = None if offline else ChainUtteranceGenerator(model_name="gpt-4.1-mini")
rng = random.Random(42)

chain = WORKFLOW_CHAINS["onboarding_pipeline_contract_success"]
conversation = instantiate_chained_conversation(
    chain,
    repo,
    scenario_selector,
    utterance_generator,
    rng,
)
```

`scripts/generate_conversations.py` enforces a 60/40 success-to-failure conversation mix. The generator raises if a requested chain set cannot hit the target ratio; expand aliases before calling `compute_chain_plan` to ensure both variants are present.

### Executing Chained Conversations

```python
from src.evaluation.conversation_harness import ConversationHarness

harness = ConversationHarness([conversation])
results = harness.run()

result = results[0]
print(f"Chain success: {result.chain_success}")
print(f"Segments: {len(result.per_segment_results)}")
for segment in result.per_segment_results:
    print(
        f"  Segment {segment['segment_number']}: "
        f"actual={segment['actual_outcome']}, expected={segment['expected_outcome']}"
    )
```

## Template Resolution

Cross-turn references use `{{turn_N.field}}` syntax:

```python
{
    "client_id": "{{turn_1.client_id}}",  # References turn 1's client_id
    "amount": 50000
}
```

The `ReferenceResolver` validates and resolves these templates, ensuring:
- Turn numbers are sequential
- Referenced fields exist in previous turn results
- No circular dependencies

## Entity Handoff Rules

Chain segments transfer entities using handoff rules:

- `"propagate"`: Entity ID from previous segment is used
- `"create_in_segment_N"`: Entity is created in specified segment
- Custom rules can be defined per chain

## Reproduction Commands

### Generate Single Conversations

```bash
python scripts/generate_conversations.py \
    --count 100 \
    --seed 42 \
    --output-dir artifacts/conversations_multiturn
```

### Generate Chained Conversations

```bash
CURATOR_SIMPLE_DATASET=1 PYTHONPATH=. python scripts/generate_conversations.py \
    --mode chain \
    --chain-id onboarding_pipeline_contract \
    --chain-id client_opp_quote \
    --count 50 \
    --seed 42 \
    --output-dir artifacts/conversations_multi_turn

# The CLI prints per-chain counts and aborts if the failure ratio moves outside the
# configured 60/40 +/- 2% window.
```

This command validates every generated chain on the fly; any unexpected segment failure/success aborts the run.

### Chain Smoke Test

```bash
CURATOR_SIMPLE_DATASET=1 PYTHONPATH=. python scripts/generate_conversations.py \
    --mode chain \
    --smoke-test \
    --output-dir artifacts/conversations_multi_turn/smoke
```

The smoke run prints one conversation per chain with per-segment expected vs actual outcomes and exits on mismatches.

After a full run, inspect a sample JSON row to confirm metadata fields:

```bash
head -n 1 artifacts/conversations_multi_turn/chains.jsonl | jq
```

The `cumulative_context.segment_summaries` and `turn_annotations` blocks should match the segment logs printed during generation.

### Validate Conversations

```bash
PYTHONPATH=. python scripts/validate_chains.py \
    --conversations artifacts/conversations_multiturn/conversations.jsonl \
    --smoke-test
```

### Phase 5 Scaled Generation (1,500 conversations)

1. Export API keys stored in `.env`:

    ```bash
    set -a; source .env; set +a
    ```

2. Run the offline smoke validation to ensure the harness is healthy:

    ```bash
    CURATOR_SIMPLE_DATASET=1 PYTHONPATH=. python scripts/generate_conversations.py \
        --mode chain \
        --smoke-test \
        --output-dir artifacts/conversations_multi_turn/$(date -u +"%Y%m%dT%H%M%SZ")/smoke
    ```

3. Launch the scaled generation. For the Phase 5 refresh we run Curator online (`CURATOR_SIMPLE_DATASET=0`) in three passes targeting roughly 400 simple / 600 medium / 500 complex chains while keeping the 60/40 success mix:

    ```bash
    export TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
    mkdir -p artifacts/conversations_multi_turn/${TIMESTAMP}/{simple,medium,complex,full}

    # 400 simple (client + document hygiene) conversations
    CURATOR_SIMPLE_DATASET=0 PYTHONPATH=. python scripts/generate_conversations.py \
        --mode chain \
        --count 400 \
        --seed 101 \
        --model-name gpt-4.1-mini \
        --chain-id client_management_chain \
        --chain-id contact_document_note \
        --output-dir artifacts/conversations_multi_turn/${TIMESTAMP}/simple \
        | tee artifacts/conversations_multi_turn/${TIMESTAMP}/simple/run.log

    # 600 medium (quote remediation, pipeline insights)
    CURATOR_SIMPLE_DATASET=0 PYTHONPATH=. python scripts/generate_conversations.py \
        --mode chain \
        --count 600 \
        --seed 102 \
        --model-name gpt-4.1-mini \
        --chain-id client_opp_quote \
        --chain-id search_quote_review \
        --chain-id quote_remediation \
        --chain-id summary_contract \
        --chain-id clone_expansion \
        --output-dir artifacts/conversations_multi_turn/${TIMESTAMP}/medium \
        | tee artifacts/conversations_multi_turn/${TIMESTAMP}/medium/run.log

    # 500 complex (multi-segment onboarding / deal pipeline) conversations
    CURATOR_SIMPLE_DATASET=0 PYTHONPATH=. python scripts/generate_conversations.py \
        --mode chain \
        --count 500 \
        --seed 103 \
        --model-name gpt-4.1-mini \
        --chain-id onboarding_pipeline_contract \
        --chain-id onboarding_opp_deal \
        --output-dir artifacts/conversations_multi_turn/${TIMESTAMP}/complex \
        | tee artifacts/conversations_multi_turn/${TIMESTAMP}/complex/run.log

    # Merge the three runs and build a consolidated log
    PYTHONPATH=. python scripts/merge_chain_runs.py \
        --input artifacts/conversations_multi_turn/${TIMESTAMP}/simple/chains.jsonl \
        --input artifacts/conversations_multi_turn/${TIMESTAMP}/medium/chains.jsonl \
        --input artifacts/conversations_multi_turn/${TIMESTAMP}/complex/chains.jsonl \
        --output artifacts/conversations_multi_turn/${TIMESTAMP}/full/chains.jsonl

    cat artifacts/conversations_multi_turn/${TIMESTAMP}/simple/run.log \
        artifacts/conversations_multi_turn/${TIMESTAMP}/medium/run.log \
        artifacts/conversations_multi_turn/${TIMESTAMP}/complex/run.log \
        > artifacts/conversations_multi_turn/${TIMESTAMP}/full/run.log
    ```

> **Fallback guidance:** When Gemini credentials are unavailable and `gpt-5-mini` exhausts reasoning-token budgets (max token truncation), switch to `gpt-4.1-mini` (as above) to maintain deterministic output without fallbacks. The earlier offline stub run (20251105T142324Z) remains checked in for reproducibility comparisons.

### Dataset Manifest & Analytics

Generate reproducible statistics and a Markdown baseline report after the run:

```bash
PYTHONPATH=. python analysis/chains_manifest.py \
    --dataset artifacts/conversations_multi_turn/<timestamp>/full/chains.jsonl \
    --output artifacts/conversations_multi_turn/<timestamp>/full/manifest.json \
    --seed 42 \
    --model-name gpt-4.1-mini

PYTHONPATH=. python analysis/generate_chains_report.py \
    --dataset artifacts/conversations_multi_turn/<timestamp>/full/chains.jsonl \
    --output artifacts/conversations_multi_turn/<timestamp>/full/report.md \
    --seed 42 \
    --model-name gpt-4.1-mini

PYTHONPATH=. python scripts/verify_no_fallbacks.py \
    --artifacts-dir artifacts/conversations_multi_turn/<timestamp>/full \
    --output artifacts/conversations_multi_turn/<timestamp>/full/verification_report.json

PYTHONPATH=. python analysis/lint_chains.py \
    --dataset artifacts/conversations_multi_turn/<timestamp>/full/chains.jsonl \
    --summary artifacts/conversations_multi_turn/<timestamp>/full/lint_report.json \
    --max-findings 50
```

- `analysis/chains_manifest.py` captures counts per chain, success/failure mix, average turn and segment lengths, and the workflow definitions actually sampled.
- `analysis/generate_chains_report.py` renders the Phase 5 analytics snapshot. Append `--baseline path/to/log.jsonl` for any Phase 6 baseline runs (Claude 4.5, GPT‑4.1, etc.) once those logs exist.
- `scripts/verify_no_fallbacks.py` enforces the no-fallback/placeholder policy across the exported JSONL files; its report and a short `quality_checks.md` summary should live beside the run log for future audits.
- `analysis/lint_chains.py` highlights duplicate utterances and entity-name conflicts so we can triage conversational drift before shipping baselines.

## Success/Failure Distribution

The single-workflow generator maintains a 60/40 success/failure distribution via the `success_ratio` parameter (default 0.6):
- **Scenario level**: 60% success, 40% failure scenarios
- **Turn level**: 60% success turns, 40% failure turns
- **Conversation level**: 60% success conversations, 40% failure conversations

For chained workflows, success and failure segments are controlled explicitly through each chain’s `success_pattern`. Harness validation guarantees the expected outcome per segment.

## Validation

All generated conversations must:
1. Pass `ConversationHarness` validation (segment-by-segment expected vs actual outcome checks).
2. Resolve template references correctly (including cross-segment offsets).
3. Propagate entity state across turns and segments without fallbacks.
4. Comply with `fake_crm_tables_schema.json` and Mock CRM constraints.
5. Execute deterministically in offline mode (`CURATOR_SIMPLE_DATASET=1`) with no placeholder utterances.

## No Fallbacks Policy

- Every turn must derive from validated scenarios
- Missing tools trigger generation, not fallbacks
- No placeholder values or artificial data
- All conversations must execute successfully
- Offline mode mirrors the same guarantees—deterministic utterances replace Curator calls, never generic placeholders

## Reference Run (2025-11-05 14:44 UTC)

- Conversations: `artifacts/conversations_multi_turn/chains.jsonl` (1,500 conversations split 900 simple / 450 medium / 150 complex; 600 expected failures; 40.0% failure ratio within tolerance)
- Manifest: `artifacts/conversations_multi_turn/20251105T144453Z/full/manifest.json`
- Report: `artifacts/conversations_multi_turn/20251105T144453Z/full/report.md`
- Run log: `artifacts/conversations_multi_turn/20251105T144453Z/full/run.log`
- Verification report: `artifacts/conversations_multi_turn/20251105T144453Z/full/verification_report.json`
- Lint summary: `artifacts/conversations_multi_turn/20251105T144453Z/full/lint_report.json`
- Quality summary: `artifacts/conversations_multi_turn/20251105T144453Z/full/quality_checks.md`
- Single-turn additions: `SC-01250` (`opportunity_details` expected failure) appended to `artifacts/scenarios_single_turn/scenarios_clean.jsonl` and replayed via `ScenarioRepository` + `ConversationHarness` to confirm the "Opportunity not found ..." failure path prior to chained generation.

Use this run as the seed corpus for Phase 6 baselines and scaling exercises; the manifest guards the 60/40 split and every failure chain variant contains a deterministic failure segment validated by the harness.
