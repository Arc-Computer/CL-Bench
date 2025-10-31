# CRM Continual-Learning Benchmark Plan  
_Updated: October 31, 2025 (evening)_

## 1. Background & Objectives
- **Customer mandate:** Deliver a synthetic yet production-faithful CRM benchmark by mid-November to demonstrate we can push state-modifying workflows from ~80 % to ≥95 % reliability using continual learning. Customer cannot share raw data; all artifacts must be reproducible on our infrastructure.
- **Target models:** Claude 4.5 (Teacher) paired with GPT‑4.1 (Student), reflecting the architecture described in _Continual Learning, Not Training: Online Adaptation for Agents_. Salesforce CRM LLM Leaderboard (Oct 8, 2025) benchmarks provide baseline expectations, with Claude 4 Sonnet as the reference top-line model.  
- **Key deliverables:**  
  1. Reproducible CRM sandbox with realistic state transitions (schema, enums, failure modes).  
  2. Baseline evaluations (Claude, GPT) against the new environment.  
  3. Atlas-driven continual-learning loop (teacher/student orchestration, reward ensemble, persistent learning memory).  
  4. Telemetry capturing _what_ was learned (reward deltas + pamphlets), _how_ (adapter/teacher events), and _why_ (verifier rationale, drift notes).  

## 2. Current Status (October 31)
- **Environment:** `CrmEnv` (Gymnasium wrapper) + harness + docs shipped; GPT‑4.1 rollouts succeed on the Postgres-backed sandbox with schema + seeds aligned to `fake_crm_tables_schema.json`. Snapshot/reset CLIs (`scripts/db_snapshot.py`, `scripts/db_reset.py`) keep database state deterministic.  
- **Telemetry:** JSONL per-episode logs now include reward breakdowns, verifier scores/rationales, validator metadata, and placeholder learning signals (student/teacher, drift notes) ahead of Atlas integration.  
- **Open GitHub issues:** #4, #12, #13, #14, #21.  
- **Gap:** Atlas dual-agent adapter (Issue #14) and the uplift evaluation/hand-off (Issue #21) remain outstanding; telemetry needs real adapter events once Atlas hooks land.

## 3. Atlas Architecture & Integration Requirements
Based on the Atlas SDK (`atlas-sdk`) and the _Continual Learning Online Adaptation_ paper:
1. **Dual-agent control loop**  
   - _Teacher (Claude 4.5)_: Monitors Student trajectories, issues targeted guidance, and escalates supervision lanes.  
   - _Student (GPT‑4.1)_: Executes CRM actions (tool calls) using prompts augmented with the latest pamphlets.  
   - _Planner_ (optional): Seeds Student plans, referencing Teacher/Student pamphlets.  

2. **Persistent Learning Memory (PLM)**  
   - Stores distilled “pamphlets” (Teacher principles + Student action schema), keyed by task metadata.  
   - Retrieval occurs prior to each episode; pamphlets injected into prompts and planner.

3. **Reward ensemble**  
   - Fast judges evaluate trajectories (factuality, efficiency, adherence).  
   - Arbiter judge resolves disagreements; attaches structured rationale.  
   - Outputs feed both telemetry and pamphlet synthesis.

4. **Adapter events & guidance history**  
   - Atlas orchestrator records supervision lane, escalations, and guidance injections per step.  
   - These events must surface in our telemetry for the customer’s “how/why” audit.

5. **Runtime integration points**  
   - **Before reset:** Retrieve pamphlets via Atlas SDK.  
   - **During step:** Send state to Atlas (tools used, observations, reward).  
   - **After episode:** Persist new pamphlets + update PLM.  
   - **Configuration:** Toggle Atlas on/off for baseline vs. continual-learning runs.

## 4. Workstream Overview & Sequencing
Timeline assumes parallel effort but respects dependencies. “Week” references are relative to Kickoff Week (KW) = Oct 30.

### Phase A – Backend Realism (KW)  
1. **Issue #8 – Dockerized Postgres CRM backend** ✅ (delivered Oct 30)  
   - SQL schema, docker-compose, seed data aligning with `fake_crm_tables_schema.json`.  
2. **Issue #9 – Rewire CRM tools to DB** ✅ (delivered Oct 31)  
   - Postgres repository + transaction helpers wired into `CrmEnv`, harness, validators; mock backend retained for fast unit tests.  
3. **Issue #10 – Snapshot/reset utilities** ✅ (delivered Oct 31)  
   - CLI tools (`db_snapshot.py`, `db_reset.py`) restore canonical Postgres state before/after rollouts.

_Milestone A:_ `CrmEnv` successfully executes against Postgres; snapshots ensure deterministic tests.

### Phase B – Failure Coverage & Verification (KW+1)  
4. **Issue #13 – Map failure taxonomy into negative cases**  
   - Translate customer CSV + future traces into golden cases with validators; ensure coverage feeds the continual-learning evaluation suite.  
5. **Issue #15 – Add native verifier interface** ✅ (delivered Oct 31)  
   - Structured + LLM judge verifiers registered; telemetry captures rationale/score for continual-learning analysis.

_Milestone B:_ Environment reproduces high-frequency failure modes with verifier scoring.

### Phase C – Telemetry & Atlas Instrumentation (KW+2)  
6. **Issue #11 – Extend telemetry with continual-learning signals** ✅ (delivered Oct 31)  
   - Rollouts emit reward breakdowns, verifier metadata, and learning signal placeholders; Atlas adapter events to populate in Issue #14.  
7. **Issue #14 – Integrate Atlas continual-learning adapter**  
   - Embed Teacher/Student orchestration, pamphlet persistence, reward ensemble.  
   - Ensure toggles for baseline vs. CL runs, leveraging Atlas SDK connectors.  
   - Populate adapter event telemetry and validate guidance history aligns with PLM updates.

_Milestone C:_ Rollouts emit Atlas telemetry; pamphlets persisted via PLM.

### Phase D – Baselines & Reporting (KW+3)  
8. **Issue #12 – Re-run baselines on Postgres + verifier**  
   - GPT‑4.1, Claude 4 Sonnet, mock agent. Capture token usage, verifier scores, and align reporting with Salesforce CRM LLM Leaderboard reference metrics.  
9. **Issue #4 – Generate baseline analytics & report**  
   - Summaries, plots, and narratives for customer + investor briefings.  

_Milestone D:_ Deliver reproducible baselines and a narrative that benchmarks Atlas uplift.

### Phase E – Atlas Uplift Delivery (KW+4)  
10. **Issue #21 – Deliver Atlas uplift evaluation and hand-off package**  
    - Seed PLM, run guided continual-learning episodes, and execute Atlas-enabled evaluations covering all golden cases (positive + negative).  
    - Produce comparative analytics (baseline vs. Atlas) and bundle reproducible artifacts (snapshots, telemetry, pamphlets, docs) for the customer.  

_Milestone E:_ Demonstrate ≥95 % reliability uplift with a turnkey hand-off bundle reproducible on Arc infrastructure.

### Cross-cutting Considerations
- **Security & secrets:** `.env.example` must include Postgres + Atlas keys.  
- **Cost controls:** Document token spend per run; consider dry-run mode.  
- **Testing:** Add Postgres-backed CI job (nightly) once Docker services stabilise.  
- **Docs:** Update README/docs after each milestone (setup, telemetry schema, Atlas usage).  
- **Customer bundle:** Final delivery must include database snapshot, telemetry schema, pamphlets, and regeneration docs validated on a clean environment.
- **Atlas SDK integration:** Document how to run `atlas env init` / `atlas run` with the CRM environment wrapper implementing `AtlasEnvironmentProtocol` + telemetry emitters so discovery and continual-learning loops work out of the box.

## 5. Expected Atlas Experiment Flow
1. **Baseline pass:** Run Student-only (GPT‑4.1) against Postgres environment to establish ~80 % starting point.  
2. **Atlas-enabled pass:**  
   - Initialize Teacher (Claude 4.5) + Student (GPT‑4.1) with empty PLM.  
   - Run 20–30 seed episodes to populate pamphlets (mirrors ExCyTIn seeding).  
   - Re-run full suite with pamphlets retrieved; measure uplift in success, efficiency, and verifier score.  
3. **Cross-task validation:** Hold out specific CRM tasks (e.g., modify-opportunity composites) to evaluate transfer.  
4. **Telemetry review:** Confirm JSONL captures learner rationale, teacher interventions, and drift notes for audit.

## 6. Risk & Mitigation
- **Backend shift delays telemetry:** Retain mock fallback to unblock Atlas scaffolding if Postgres work slips.  
- **Verifier accuracy:** Start with deterministic structured checks; incrementally layer LLM judge ensemble.  
- **Atlas complexity:** Begin with stub adapter that records events; iterate to full teacher/student orchestration.  
- **Token costs:** Use smaller seeds for development; reserve full runs (Claude/GPT) for nightly pipeline.

## 7. Next Steps (immediate)
1. Align with Atlas team to scope adapter orchestration & PLM hooks (Issue #14); build stub that emits adapter events into telemetry and validates against `atlas env init` discovery expectations.  
2. Expand negative-case coverage from customer taxonomy (`Agent tasks - updated.csv`) to close Issue #13 and feed verifier regression tests + Atlas evaluation.  
3. Prepare Postgres-backed baseline reruns (Issue #12) using the new snapshot/reset tooling; document token/cost assumptions for customer share-outs and Issue #21 comparisons.  
4. Draft the Atlas uplift evaluation/checklist (Issue #21) so the hand-off bundle scope is clear ahead of integration completion.  
5. Continue weekly sync with Federico’s team to surface telemetry progress and gather pending failure traces.

---
_Document owner:_ **Jarrod Barnes**  
_Contributors:_ Aman Jaglan, Atlas SDK team, CRM Benchmark engineering
