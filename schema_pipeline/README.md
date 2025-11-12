# Schema-First Conversation Pipeline

This directory implements **Option B** – a research-grade, schema-driven dataset generator for CRM conversations.  
The pipeline is intentionally modular so every component can be inspected, audited, and extended without drifting
from the canonical schema (`data/fake_crm_tables_schema.json`).

## Architecture

| Stage | Module | Purpose |
| --- | --- | --- |
| Schema ingest | `generate_schema_artifacts.py` → `generated/` | Turns the JSON schema into Pydantic models + metadata so no fields are hard-coded. |
| Task sampling | `task_sampler.py` | Samples tasks from `data/Agent_tasks.csv` using production weights. |
| Workflow planning | `workflow.py` | Gemini-powered Curator block that emits step-by-step tool plans for each sampled task. |
| Argument synthesis | `arguments.py` | Generates schema-compliant arguments for every tool call and validates them against the generated models. |
| Utterance generation | `utterances.py` | Produces natural multi-turn conversations that exercise the workflow and tool arguments. |
| Quality judge | `judge.py` | Acts as our “expert reviewer”, grading conversations on schema fidelity + linguistic realism. |
| Orchestration | `pipeline.py` | Wires all Curator blocks together with viewer logging and exposes a reproducible `SchemaFirstPipeline`. |

All LLM calls use the Gemini backend (default models: `gemini-2.5-flash-lite` for generation, `gemini-2.5-flash` for the judge).  
API keys are read from the environment (`.env`), so make sure `GOOGLE_API_KEY` (or equivalent) is loaded.

## Usage

1. **Regenerate schema artifacts whenever the CRM schema changes**:
   ```bash
   python schema_pipeline/generate_schema_artifacts.py
   ```
   This writes `schema_pipeline/generated/schema_models.py` + `schema_metadata.json`.

2. **Run the pipeline** (example inside a notebook or script):
   ```python
   from schema_pipeline import PipelineConfig, SchemaFirstPipeline

   pipeline = SchemaFirstPipeline(PipelineConfig())
   batch = pipeline.generate_batch(batch_size=10)
   ```
   `batch` contains intermediate artifacts (`workflows`, `arguments`, `conversations`, `judgements`) ready for
   downstream filtering, storage, or evaluation harness playback.

3. **Viewer logging**: set `GeminiConfig.viewer_enabled=True` (default) to stream progress to the Curator viewer.
   Disable by toggling the flag in `PipelineConfig`.

## Reproducibility Notes

- **Schema guarantees**: Required fields/enums are enforced by the generated models inside `schema_pipeline/generated`.
- **Task weights**: `TaskSampler` directly consumes `data/Agent_tasks.csv`; update that file to change sampling.
- **LLM determinism**: adjust `GeminiConfig.generation_params` to control temperature/top-p per stage.
- **Artifacts**: final JSON/JSONL dumps should live in `artifacts/schema_pipeline/` (configure via `PipelineConfig.output_dir`).

Treat this folder as the only source of truth for new dataset generation. Anything moved into `artifacts/legacy/`
is frozen for archival purposes and should not be reused without explicit validation.
