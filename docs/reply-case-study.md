# Reply Case Study

*Internal Technical Evaluation Report — Not for External Distribution*

**Date:** November 11, 2025

**Prepared by:** Arc Intelligence Jarrod Barnes 

## Executive Summary

This joint evaluation between Reply and Arc Intelligence was conducted to test whether inference-time continual learning and on-policy distillation can meaningfully improve reliability and efficiency in state-modifying enterprise systems, such as the CRM workflow built by Reply and recreated by Arc.

The study replicated a real production environment currently achieving ~78–81% reliability, providing a realistic and sample testbed for continual learning under the same task complexity, schema, and data constraints faced in client deployments.

The objective was to validate Arc’s [inference-time continual learning](https://arxiv.org/abs/2511.01093) system (Atlas SDK + Atlas Core) as a credible mechanism for achieving high-reliability (~95%) performance without full model retraining or manual prompt engineering.

The evaluation focused on three core questions:

1. Reliability Improvement: Can inference-time learning and on-policy distillation yield statistically significant uplift over the baseline model in realistic, multi-turn CRM tasks?
2. Operational Feasibility: Can the Atlas system integrate into Reply’s existing AI pipelines without disrupting current evaluation, monitoring, or data-governance processes?
3. Scalability and Reuse: Can the benchmark framework be standardized and replicated across other Reply environments (e.g., financial services, telco, manufacturing)?

The results have demonstrated 

### 1.1 Objective

- **Purpose:** Demonstrate that Arc’s Atlas continual-learning stack can lift an already production-ready CRM agent from ~80 % reliability to ≈95 %+ without accessing Reply’s proprietary data or retraining foundation models.
- **Scope of evaluation:** Synthetic CRM benchmark faithfully mirroring Reply’s schema and workflows, executed end‑to‑end on a Postgres sandbox that enforces the same state mutations, validation rules, and tool surface area as the live system.
- **Key questions tested:** (1) How much reliability uplift can Atlas deliver versus state-of-the-art baselines (Claude 4.5 Sonnet, GPT‑4.1/mini) on multi-turn, state-modifying tasks? (2) Can Atlas’ inference-time learning loop capture and replay specific improvements (reward strings, teacher/student guidance) that explain *what* was learned, *how* it was learned, and *why* interventions fired? (3) Can the full pipeline—synthetic generation → CRM replay → baselines → Atlas → distillation—operate fast enough to support Reply’s mid-November Worldwide Retreat milestone?

---

### 1.2 Problem Context

- **Baseline challenges:** Reply’s deployed CRM agents already clear ~78–81 % success, but squeezing out the remaining 15–20 % requires expensive weekly trace reviews, handcrafted prompt tweaks, and manual QA of DB state deltas. European enterprise buyers will not tolerate <90 % reliability for state mutations, so human-in-the-loop patching has become the bottleneck.
- **Limitations of existing approaches:** Context-training alone (Databricks ALHF, JEPA-style quality-diversity evolvers) yields fragile gains, especially when agents must *write* to CRM tables. Customers rarely expose large test environments, so uncontrolled evaluation runs risk polluting shared staging DBs; this prevents meaningful continual-learning experiments.
- **Relevance to Reply’s workflows:** The benchmark replicates the exact scenario Federico’s team described: an agent that already manipulates opportunities, quotes, contracts, and notes in production, yet stalls before 90 %. Reply needs proof that automated reward synthesis + student/teacher supervision can replace the current manual “download traces → categorize → hand-edit prompts → redeploy” loop. The mid-November Worldwide Retreat demands a credible demo that continual learning is the “automobile,” not another faster horse.

---

### 1.3 Methodology Overview

Our synthetic dataset is generated entirely in-repo so it mirrors Reply’s CRM semantics while remaining fully reproducible:

1. **Task Sampling (`data/Agent_tasks.csv`)** – Each record describes a production workflow (e.g., create opportunity, update quote). The schema pipeline samples a balanced mix of these tasks for every batch run, preserving Reply’s real-world frequency distribution.
2. **Schema Grounding (`data/fake_crm_tables_schema.json`)** – Before any prompts are drafted, the pipeline loads the CRM schema (tables, enums, FK relationships). This schema supplies the context blocks fed to every LLM stage so generated arguments always respect the same field constraints as the production database.
3. **Workflow Planning** – `schema_pipeline.workflow` expands a sampled task into 5–8 concrete tool steps (search → modify → verify). Each step is annotated with validation hints that the downstream argument generator must satisfy.
4. **Argument Generation** – `schema_pipeline.arguments` produces concrete payloads (e.g., `modify_opportunity` arguments) and explicitly links parameters (client IDs, quote IDs) across steps. All IDs remain symbolic at this stage.
5. **Utterance + Replay Loop** – Conversations are authored with placeholders, then `ConversationHarness` replays every turn against the Postgres CRM backend. During replay we seed deterministic CRM records, resolve placeholders into live GUIDs, and rewrite assistant turns with ground-truth values, ensuring datasets contain only executable traces.

Evaluation stages:
- **Baseline:** Claude 4.5 Sonnet, GPT‑4.1, and GPT‑4.1 mini agents execute the 519 clean conversations via ConversationHarness (Postgres backend, judge enabled).
- **Atlas Runtime:** The same dataset feeds `run_atlas_baseline`, which wraps the CRM harness with Atlas’ student/teacher loop (student GPT‑4.1 mini, teacher GPT‑4.1, paired mode).
- **Atlas + GKD:** Once Atlas sessions are captured, Atlas Core distills teacher interventions back into an improved student; those distilled checkpoints re-enter the harness for the final pass.

Metrics captured per stage:
- Task success rate (conversation‑level and turn‑level).
- Average tokens / cost per episode (from LiteLLM/Atlas telemetry).
- Atlas reward summaries (session reward mean/min/max, cue hits, action adoptions).
- Runtime efficiency (wall-clock, tool latency) pulled from harness logs.

---

### 1.4 Key Findings

| Evaluation Stage | Task Success (%) | Avg Tokens | Cost per Episode | Reliability Gain |
| --- | --- | --- | --- | --- |
| Baseline |  |  |  |  |
| Atlas Runtime |  |  |  |  |
| Atlas + GKD |  |  |  |  |
- Summary of improvements:
- Observations:

---

### 1.5 Key Learnings

- 
- 
- 
- 

---

### 1.6 Conclusion

- 

---

## Technical Overview

### 2.1 Configuration

**Synthetic data pipeline**
- **Source tables:** `data/Agent_tasks.csv` encodes Reply’s CRM workflows (task description, intent, verification mode, action count). `data/fake_crm_tables_schema.json` mirrors the production schema (clients, opportunities, quotes, contracts, documents, notes).
- **Pipeline stages:** `schema_pipeline` orchestrates workflow planning → argument generation → utterance synthesis → harness replay, persisting intermediate artifacts under `artifacts/schema_pipeline/`.
- **Execution backend:** ConversationHarness connects to a live Postgres CRM instance defined by `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`. During generation/replay we seed deterministic CRM records so every conversation issues real SQL mutations.

**Evaluation toolchain**
- **Baseline harness:** `src/evaluation/run_baseline.py` loads the cleaned conversation JSONL, instantiates a LiteLLM-backed agent, and steps through each turn via ConversationHarness. `--backend postgres` ensures tool calls hit the same DB used during data generation.
- **LLM agents (LiteLLM adapters):**
  - *Claude 4.5 Sonnet* – `--agent claude --model claude-sonnet-4-5-20250929`, provider `anthropic`, temperature 0.0, max_output_tokens 800.
  - *GPT‑4.1* – `--agent gpt4.1 --model gpt-4.1`, provider `openai`, temperature 0.0, max_output_tokens 800.
  - *GPT‑4.1 mini* – identical settings with `--model gpt-4.1-mini`.
- **Atlas runtime:** `run_atlas_baseline` wraps the same conversation file with Atlas SDK using `configs/atlas/crm_harness.yaml`. Key config blocks:
  - `agent`: `type: crm_harness`, backend `postgres`, `use_llm_judge: true`.
  - `teacher`: OpenAI GPT‑4.1 (`api_key_env: OPENAI_API_KEY`, temperature 0.05, paired orchestration).
  - `student`: tool_choice `auto`; runtime overrides set the LiteLLM agent to GPT‑4.1 mini (provider `openai`, temperature 0.0, max_output_tokens 800).
  - `storage`: Atlas telemetry persists to `STORAGE__DATABASE_URL` (dedicated Postgres schema for rewards, playbooks, and learning artifacts).
- **Environment variables:** `.env` exports both CRM credentials (`DB_*`) and Atlas storage (`STORAGE__DATABASE_URL`) plus model keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`). Every CLI invocation is wrapped with `set -a; source .env; set +a` so agents, judges, and Atlas share the same credentials.

Together these components recreate Reply’s CRM stack end-to-end: synthetic data rooted in the production task/schema definitions, execution/replay on a live Postgres environment, and evaluation agents configured exactly as they would be in deployment.

---

### 2.2 Experimental Design

- Steps:
- Hardware used:
- Visualization:

---

### 2.3 Results & Analysis

- Performance highlights:
- Supporting charts or visuals:
- Links to supporting documentation:

---

## Business Impacts & Applications

The CRM benchmark addressed a core operational challenge in Reply’s delivery work: the high cost and limited scalability of evaluation cycles for production systems in state-modifying environments. This evaluation demonstrates how Arc's continual learning framework (Atlas SDK and Atlas Core) significantly improves the efficiency and accuracy of production-grade AI systems, without requiring costly model retraining or additional infrastructure. 

By integrating Arc's Atlas, Reply can deliver smarter, more reliable, and lower-cost AI systems. This increases project margins through operational efficiency and accuracy of agents. 

This combination of operational efficiency and accuracy at scale provides a defensible commercial advantage in Reply's core markets. It also supports the development of a repeatable, high-margin managed AI offering.

---

## **Next Steps & Collaboration Pathways**

Following this benchmark, Arc recommends continued collaboration with the Reply research and delivery teams to formalize how continual learning can strengthen Reply’s service portfolio and  operational workflow:

The table below outlines potential next steps and partnership pathways between Arc and Reply:

| **Pathway** | **Description** | **Benefit to Reply** |
| --- | --- | --- |
| **A. Joint Case Study** | Co-author an internal or external publication summarizing benchmark findings and the role of causality data in improving agent learning efficiency and accuracy. | Reinforces Reply’s leadership in adaptive AI and provides a credible, data-backed reference for engaging enterprise clients on continual learning. |
| **B. Internal Capability Handoff** | Deliver distilled evaluation scripts, reproducible benchmark framework, and trained model artifacts for internal testing and reuse. | Enables Reply’s R&D teams to replicate the methodology across additional use cases, reducing future setup time and engineering overhead. |
| **C. Managed Continual-Learning Integration** | Embed Arc’s Atlas SDK and trainer into Reply’s existing MLOps stack to automate evaluation, reward assignment, and ongoing model adaptation. | Converts high-cost evaluation cycles into a repeatable, scalable process—reducing delivery cost and improving model accuracy for client engagements. |
| **D. Channel Partnership for Adaptive AI Services** | Establish a structured partnership where Reply integrates Arc’s platform into its enterprise offerings as a white-labeled or co-branded service. | Creates a differentiated “adaptive AI” line of business for Reply, generating recurring revenue through reliability-focused managed learning contracts. |

---

## Appendix

- 
- 
- 

---

## Delivery Checklist

| Deliverable | Owner | Status | Due |
| --- | --- | --- | --- |
| Technical Results Summary | Jarrod | ☐ |  |
| Executive Summary + Business Context | Steph | ☐ |  |
| Full Report PDF | Jarrod + Steph | ☐ |  |
| Slack Summary to Reply | Steph | ☐ |  |
| Case Study Template Finalized for Reuse | Steph | ☐ |  |

The objective was to validate continual learning as a credible mechanism for achieving **high-reliability (≈95%) performance** without full model retraining or manual prompt engineering.

The evaluation focused on three core questions:

1. **Reliability Improvement:** Can inference-time learning and on-policy distillation yield statistically significant uplift over the baseline model in realistic, multi-turn CRM tasks?
2. **Operational Feasibility:** Can the Atlas system integrate into Reply’s existing AI pipelines without disrupting current evaluation, monitoring, or data-governance processes?
3. **Scalability and Reuse:** Can the benchmark framework be standardized and replicated across other Reply environments (e.g., financial services, telco, manufacturing)?
