# CRM Continual-Learning Benchmark Plan
_Updated: November 2, 2025_

## 1. Background & Objectives
- **Customer mandate:** Deliver a synthetic yet production-faithful CRM benchmark by mid-November to demonstrate we can push state-modifying workflows from ~80% to ≥95% reliability using continual learning. Customer cannot share raw data; all artifacts must be reproducible on our infrastructure.
- **Target models:** Claude Sonnet 4.5 (Teacher) paired with GPT-4.1 (Student), reflecting the architecture described in _Continual Learning, Not Training: Online Adaptation for Agents_. Salesforce CRM LLM Leaderboard (Oct 8, 2025) benchmarks provide baseline expectations, with Claude Sonnet 4.5 as the reference top-line model.
- **Success definition:** **95% reliability = 95% of golden cases achieve verifier score ≥0.9** (stricter than ExCyTIn-Bench's 0.4 threshold for production readiness).
- **Key deliverables:**
  1. Reproducible CRM sandbox with realistic state transitions (schema, enums, failure modes).
  2. Baseline evaluations (Claude, GPT) against the new environment.
  3. Atlas-driven continual-learning loop (teacher/student orchestration, reward ensemble, persistent learning memory).
  4. Telemetry capturing _what_ was learned (reward deltas + pamphlets), _how_ (adapter/teacher events), and _why_ (verifier rationale, drift notes).
  5. Learning metrics demonstrating gradient-free adaptation: Cue Hit Rate, Adoption Rate, Reward Delta, Token Delta, Transfer Success.

## 2. Current Status (November 2, 2025)
- **Environment:** `CrmEnv` (Gymnasium wrapper) + harness + docs shipped; GPT-4.1 rollouts succeed on the Postgres-backed sandbox with schema + seeds aligned to `fake_crm_tables_schema.json`. Snapshot/reset CLIs (`scripts/db_snapshot.py`, `scripts/db_reset.py`) keep database state deterministic.
- **Telemetry:** JSONL per-episode logs now include reward breakdowns, verifier scores/rationales, validator metadata, and placeholder learning signals (student/teacher, drift notes) ahead of Atlas integration.
- **Open GitHub issues:** #4, #12, #13, #14, #21, #22.
- **Gap:** Golden-case corpus at 103 scenarios (Issue #22) must reach ≥1,500 for robust evaluation; Atlas dual-agent adapter (Issue #14) and the uplift evaluation/hand-off (Issue #21) remain outstanding; telemetry needs real adapter events and learning metrics once Atlas hooks land.

## 3. Atlas Architecture & Integration Requirements
Based on the Atlas SDK (`atlas-sdk`, v0.1.10+) and the _Continual Learning Online Adaptation_ paper:

### 3.1 Dual-Agent Control Loop
   - **Teacher (Claude Sonnet 4.5)**: Monitors Student trajectories, issues targeted guidance, and escalates supervision lanes.
   - **Student (GPT-4.1)**: Executes CRM actions (tool calls) using prompts augmented with the latest learning playbooks (pamphlets).
   - **Planner** (optional): Seeds Student plans, referencing Teacher/Student playbooks.

### 3.2 Persistent Learning Memory (PLM)
   - **Storage backend**: Postgres (already integrated in Atlas SDK).
   - **Content**: Stores distilled "learning playbooks" (Teacher principles + Student action schemas), keyed by task metadata.
   - **Retrieval strategy**: Semantic similarity search prior to each episode; playbooks injected into Student/Teacher prompts.
   - **Implementation**: Atlas SDK's `atlas.learning.playbook` module handles persistence, retrieval, and cache invalidation.

### 3.3 Reward Integration Module (RIM)
   - **Fast judges**: Multiple lightweight evaluators score trajectories (factuality, efficiency, adherence, safety).
   - **Arbiter judge**: Resolves disagreements when variance/uncertainty exceeds thresholds; attaches structured rationale.
   - **Output**: Principle-grounded rationales feeding telemetry and playbook synthesis.
   - **Implementation**: Atlas SDK's `atlas.evaluation.evaluator` module with pluggable judge definitions.

### 3.4 Adapter Events & Guidance History
   - **Atlas orchestrator** (`atlas.runtime.orchestration.orchestrator`) records:
     - Supervision lane selection (autonomous, step-by-step, full teacher control)
     - Guidance injections per step
     - Playbook retrieval/adoption events
     - Escalation triggers
   - **Telemetry requirement**: These events must surface in JSONL logs for "how/why" audit trail.

### 3.5 Runtime Integration Points
   - **Before reset**: Atlas runtime retrieves relevant playbooks via PLM, injects into Student/Teacher context.
   - **During step**: CrmEnv.step() actions forwarded to Atlas orchestrator, which:
     - Executes action via Student agent
     - Monitors for drift/failure patterns
     - Applies Teacher guidance as needed
     - Records trajectory events
   - **After episode**: RIM evaluates full trajectory, synthesizes new playbooks, persists to PLM.
   - **Configuration**: YAML config toggles Atlas on/off (`atlas_enabled: true/false`).

### 3.6 Learning Metrics Framework
Atlas telemetry captures five core metrics demonstrating gradient-free continual learning:

| Metric | Definition | Measurement Source |
|--------|-----------|-------------------|
| **Cue Hit Rate** | % of episodes where task triggers playbook retrieval | `metadata.secrl_applied_guidance` presence in session traces |
| **Adoption Rate** | % of retrieved playbooks actually injected into prompts | Playbook cache logs + prompt digest comparison |
| **Reward Delta** | Δ(reward) baseline vs. guided episodes | RIM reward scores: `guided_avg - baseline_avg` |
| **Token Delta** | Δ(tokens) baseline vs. guided episodes | Session telemetry: `(baseline_tokens - guided_tokens) / baseline_tokens` |
| **Transfer Success** | Accuracy improvement on held-out tasks/incidents | Cross-task/cross-incident evaluation harness |

**Target thresholds (based on ExCyTIn-Bench results):**
- Cue Hit Rate: ≥60% (ExCyTIn achieved 69/98 = 70%)
- Adoption Rate: ≥90% (high-confidence retrieval should be adopted)
- Reward Delta: ≥+15pp (33.7% → 54.1% = +20.4pp on ExCyTIn)
- Token Delta: ≥-40% (ExCyTIn achieved -45%)
- Transfer Success: ≥+40% (ExCyTIn cross-incident: 28% → 41% = +46%)

## 4. Workstream Overview & Sequencing
Timeline references are relative to **Kickoff Date = November 2, 2025**. Mid-November deadline = **November 16, 2025** (14 days).

### Phase A – Backend Realism ✅ **COMPLETED**
1. **Issue #8 – Dockerized Postgres CRM backend** ✅ (delivered Oct 30)
2. **Issue #9 – Rewire CRM tools to DB** ✅ (delivered Oct 31)
3. **Issue #10 – Snapshot/reset utilities** ✅ (delivered Oct 31)
4. **Issue #15 – Add native verifier interface** ✅ (delivered Oct 31)

_Milestone A:_ `CrmEnv` successfully executes against Postgres; snapshots ensure deterministic tests; verifiers score agent behavior.

---

### Phase B – Failure Coverage & Dataset Expansion (Nov 2-8, Week 1)

#### 4. **Issue #13 – Failure taxonomy blueprints** (Nov 2-4, 3 days)
   - Convert customer failure categories into structured templates:
     - **Schema violations**: Invalid enum values, missing required fields, type mismatches
     - **State inconsistencies**: Creating quotes for non-existent opportunities, duplicate entity errors
     - **Workflow violations**: Stage transitions that skip required steps, permission errors
   - Output: Blueprint schema with `FailureCategoryTemplate` dataclass
   - Generator hooks: Templates → negative case instantiation with validator/verifier expectations

#### 5. **Issue #22 – Synthetic case generator (1,500 scenarios)** (Nov 4-8, 5 days)
   - **Phased approach**:
     - **Phase 1 (Nov 4-5)**: Build blueprint-driven generator, emit 500 scenarios, validate pipeline
     - **Phase 2 (Nov 6-8)**: Scale to 1,500 scenarios (success/failure mix: 60/40)
   - **Output format**: Manifests + DB seeds aligned to `fake_crm_tables_schema.json`
   - **Validation**: Run 50-scenario smoke test, verify verifier coverage
   - **Rationale for 1,500**:
     - Statistical power for learning metrics (Cue Hit Rate, Transfer Success)
     - Coverage across abstraction hierarchy (task-specific → universal principles)
     - ExCyTIn-Bench used 98 tasks; CRM domain complexity justifies 15x scale
     - Phased generation mitigates timeline risk (500 scenarios enables partial evaluation if needed)

_Milestone B:_ Environment reproduces high-frequency failure modes; 1,500-scenario corpus ready for baseline evaluation.

---

### Phase C – Atlas Integration & Learning Metrics (Nov 9-12, Week 2, Days 1-4)

#### 7. **Issue #14 – Integrate Atlas continual-learning adapter** (Nov 9-12, 4 days)
   - **Simplified scope** (Atlas SDK provides infrastructure):
     - **Day 1 (Nov 9)**: Run `atlas env init` on CrmEnv, validate autodiscovery generates `.atlas/generated_config.yaml`
     - **Day 2 (Nov 10)**: Configure RIM judges to use existing `StructuredVerifier` + `LlmJudgeVerifier`, test reward aggregation
     - **Day 3 (Nov 11)**: Run 10-episode smoke test (5 success, 5 failure cases), validate:
       - Playbook retrieval/injection
       - Teacher guidance generation
       - Learning metrics telemetry (Cue Hit Rate, Adoption Rate, Token Delta)
     - **Day 4 (Nov 12)**: Create baseline toggle (`atlas_enabled: false` config), document Atlas usage

   - **No custom implementation needed**:
     - ✅ PLM storage: Atlas SDK's Postgres backend (already integrated)
     - ✅ Reward ensemble: Atlas SDK's RIM module (configure judges via YAML)
     - ✅ Telemetry: Atlas SDK's `atlas.runtime.telemetry` (streams to Postgres)
     - ✅ Orchestrator: Atlas SDK's `atlas.runtime.orchestration.orchestrator`

   - **Integration pattern**:
     ```python
     # CrmEnv already Gymnasium-compatible; Atlas SDK autodiscovery handles wrapping
     # atlas env init  # Scans CrmEnv, writes config
     # atlas run --config .atlas/generated_config.yaml --task "Create client with validation"
     ```

   - **CRM-specific metadata** (populate Atlas SDK's `session_metadata` field):
     ```json
     {
       "session_metadata": {
         "session_id": 42,
         "status": "succeeded",
         "student_learning": "Learned to validate email format before API calls",
         "teacher_learning": "Student needs explicit schema validation reminders",
         "execution_mode": "adaptive",
         "token_usage": {"total": 6820, "student": 5200, "teacher": 1620},
         "crm_context": {
           "golden_case_id": "gc_103",
           "task_category": "client_creation",
           "playbooks_retrieved": [
             {"id": "crm_client_validation_v3", "similarity": 0.89},
             {"id": "email_format_check_v1", "similarity": 0.76}
           ],
           "playbooks_adopted": ["crm_client_validation_v3"]
         }
       }
     }
     ```
   - **Note**: Learning metrics (Cue Hit Rate, Adoption Rate, Reward/Token Delta) calculated post-hoc via Atlas SDK's `atlas.evaluation.learning_report` module (see Appendix B)

_Milestone C:_ Rollouts emit Atlas telemetry with learning metrics; playbooks persisted via PLM; baseline toggle validated.

---

### Phase D – Baselines & Comparative Analytics (Nov 12-14, Week 2, Days 4-6)

#### 8. **Issue #12 – Re-run baselines on Postgres + verifier** (Nov 12-13, 2 days parallel with Issue #14 Day 4)
   - **Baseline configurations**:
     - GPT-4.1 (Student-only, no Atlas)
     - Claude Sonnet 4.5 (Teacher-only, no Atlas)
     - Mock random agent (sanity check)
   - **Evaluation subset**: 300-500 scenarios (stratified sample from 1,500 corpus)
   - **Metrics captured**: Token usage, verifier scores, success rate, cost per episode
   - **Alignment**: Map CRM tasks to Salesforce CRM LLM Leaderboard categories (lead creation, opportunity modification, etc.)

#### 9. **Issue #4 – Generate baseline analytics & comparative report** (Nov 13-14, 2 days)
   - **Deliverables**:
     - Performance comparison tables (baseline vs. Atlas-enabled)
     - Learning progression plots (Cue Hit Rate, Token Delta over episodes)
     - Cost-efficiency analysis ($ per successful episode)
     - Narrative for customer/investor briefings
   - **Format**: Markdown report + Jupyter notebook with interactive plots

_Milestone D:_ Deliver reproducible baselines and comparative narrative demonstrating Atlas uplift trajectory.

---

### Phase E – Atlas Uplift Evaluation & Hand-off (Nov 14-16, Week 2, Days 6-8)

#### 10. **Issue #21 – Deliver Atlas uplift evaluation and hand-off package** (Nov 14-16, 3 days)
   - **Seeding phase (Nov 14, 0.5 days)**:
     - Run 20-30 seed episodes to populate PLM (mirrors ExCyTIn seeding)
     - Validate playbook synthesis quality, ensure guidance is actionable

   - **Uplift evaluation (Nov 14-15, 1.5 days)**:
     - Atlas-enabled runs on full 1,500-scenario corpus (or 500+ if timeline compressed)
     - Measure vs. success definition: **95% of cases achieve verifier score ≥0.9**
     - Track learning metrics across episodes (Cue Hit Rate, Reward Delta, Token Delta)

   - **Cross-task transfer validation (Nov 15-16, 1 day)**:
     - Hold out 10-15% of scenarios (not seen during seeding)
     - Run with frozen playbooks (no new Teacher guidance)
     - Measure transfer success: Target ≥+40% accuracy improvement

   - **Hand-off package (Nov 16)**:
     - **Code**: Tagged release with Atlas integration (v1.0.0)
     - **Data**: Database snapshot (Postgres dump), 1,500-scenario manifests, seed data
     - **Telemetry**: Complete JSONL session traces with learning metrics
     - **Playbooks**: Exported PLM artifacts (JSON)
     - **Documentation**:
       - Setup instructions (`atlas env init` → `atlas run`)
       - Reproduction guide (baseline vs. Atlas runs)
       - Telemetry schema reference
       - Learning metrics interpretation guide
     - **Analytics**: Comparative report (Issue #4 output)

_Milestone E:_ Demonstrate ≥95% reliability uplift with turnkey hand-off bundle reproducible on Arc infrastructure.

---

### Cross-cutting Considerations
- **Security & secrets**: `.env.example` includes:
  - `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (LLM providers)
  - `DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas` (Atlas SDK Postgres)
  - `STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas` (session persistence)
- **Cost controls**:
  - Token budget: ~$500 development, ~$2,000 final evaluation runs
  - Use `ATLAS_FAKE_LLM=1` for offline testing (no API costs)
  - Use GPT-4.1-mini for rapid iteration (Phase C smoke tests)
  - Reserve Claude Sonnet 4.5 + GPT-4.1 for Phase D/E
  - Track spend in real-time via `session_metadata.token_usage` (streamed to Postgres)
- **Testing**:
  - Add Postgres-backed integration tests (nightly CI once Phase B complete)
  - Atlas adapter smoke tests (Phase C, Day 3)
  - Baseline reproducibility tests (snapshot → reset → identical results)
- **Docs**:
  - Update README after Phase C (Atlas usage examples)
  - Create `docs/atlas_integration.md` with architecture diagrams
  - Add learning metrics interpretation guide
  - Document Atlas SDK telemetry schema extensions (Appendix B reference)
- **Customer bundle validation**:
  - Run full reproduction test on clean macOS/Linux environment
  - Verify snapshots restore correctly
  - Validate Atlas SDK telemetry captures all "what/how/why" requirements
  - Export Postgres database dump for reproducibility

## 5. Expected Atlas Experiment Flow
1. **Baseline pass** (Phase D, Issue #12):
   - Run Student-only (GPT-4.1, `atlas_enabled: false`) against 300-500 scenario subset
   - Establish ~80% starting point (or lower if failure coverage is high)
   - Capture token usage, verifier scores, cost per episode

2. **Atlas-enabled pass** (Phase E, Issue #21):
   - **Seeding (20-30 episodes)**:
     - Initialize empty PLM
     - Teacher (Claude Sonnet 4.5) observes Student (GPT-4.1) trajectories
     - Generate initial playbooks (principles + action schemas)
   - **Uplift evaluation (remaining episodes)**:
     - Retrieve playbooks via semantic similarity
     - Inject into Student/Teacher prompts
     - Measure learning metrics:
       - Cue Hit Rate ≥60%
       - Adoption Rate ≥90%
       - Reward Delta ≥+15pp
       - Token Delta ≥-40%
   - **Target outcome**: 95% of cases achieve verifier score ≥0.9

3. **Cross-task transfer validation** (Phase E, hold-out set):
   - Freeze PLM (no new playbooks)
   - Switch to Student-only mode (`atlas_enabled: true`, but Teacher provides no new guidance)
   - Run on held-out scenarios (10-15% of corpus)
   - Measure Transfer Success: ≥+40% accuracy improvement vs. baseline

4. **Telemetry audit** (Phase E, final review):
   - Confirm JSONL captures:
     - Learning metrics (Cue Hit Rate, Adoption Rate, Reward/Token Delta)
     - Teacher interventions with rationale
     - Playbook retrieval/injection events
     - Drift notes (supervision lane escalations)
   - Validate against customer audit requirements

## 6. Risk & Mitigation
- **Dataset generation delays** (Issue #22):
  - **Risk**: 1,500 scenarios in 5 days is aggressive; quality may suffer
  - **Mitigation**: Phase 1 validates pipeline with 500 scenarios; can proceed to Phase D/E with partial corpus if needed
  - **Fallback**: 500 high-quality scenarios > 1,500 low-quality scenarios

- **Atlas integration complexity**:
  - **Risk**: CrmEnv incompatibility with Atlas SDK autodiscovery
  - **Mitigation**: CrmEnv already Gymnasium-compatible; Atlas SDK handles wrapping. Day 1 smoke test validates integration.
  - **Fallback**: Manual adapter implementation (2-day slip)

- **95% reliability threshold not achieved**:
  - **Risk**: Even with Atlas, verifier score ≥0.9 on 95% of cases may be too strict
  - **Mitigation**: Track threshold sensitivity (0.7, 0.8, 0.9); report range in analytics
  - **Customer communication**: If 95%@0.9 not reached, demonstrate improvement trajectory + provide roadmap

- **Token cost overruns**:
  - **Risk**: Claude Sonnet 4.5 Teacher + GPT-4.1 Student on 1,500 episodes may exceed budget
  - **Mitigation**:
    - Use GPT-4.1-mini for smoke tests (Phase C)
    - Run baselines on subset (300-500 scenarios)
    - Full uplift evaluation budgeted at ~$2,000 (monitored via telemetry)

- **Timeline compression** (14 days total):
  - **Risk**: Phases B-E must execute in parallel to meet Nov 16 deadline
  - **Mitigation**:
    - Issue #13 (blueprints) unblocks Issue #22 (generator) immediately
    - Issue #14 (Atlas integration) can proceed with 500 scenarios from Phase B.1
    - Issue #12 (baselines) runs parallel with Phase C completion
  - **Buffer**: Phase E can slip to Nov 18 if critical path blocked (negotiate 2-day extension)

## 7. Next Steps (November 2-3, Immediate)
1. **Issue #13** (Jarrod/Aman): Finalize failure taxonomy blueprints, document template schema (target: Nov 4 EOD)
2. **Issue #22 Phase 1** (Aman): Build blueprint-driven generator, emit first 500 scenarios (target: Nov 5 EOD)
3. **Issue #14 Day 1** (Jarrod): Run `atlas env init` on CrmEnv, validate autodiscovery (target: Nov 9 AM)
4. **Coordination**: Weekly sync with Federico's team (schedule Nov 6 or 7) to surface telemetry progress and gather pending failure traces

---

## Appendix A: Learning Abstraction Hierarchy

Atlas learning playbooks target principles at four abstraction levels, enabling transfer from task-specific to universal knowledge:

| Level | Scope | Example CRM Principles | Transfer Breadth |
|-------|-------|------------------------|------------------|
| **Level 1: Task-Specific** | Single task instance | "For task X, client_id must be 'C-12345'" | Zero transfer (cached answer) |
| **Level 2: Domain-Specific** | CRM domain | "Always validate email format before client creation" | Narrow (CRM client tasks) |
| **Level 3: Workflow-Specific** | Cross-domain workflows | "Verify entity existence before creating dependent records" | Broad (any entity system) |
| **Level 4: Universal** | General reasoning | "When API returns 404, check argument spelling before retrying" | Maximum (any API task) |

**Evaluation strategy**: Hold-out scenarios designed to test Level 3-4 transfer (e.g., apply client-validation principles to opportunity-modification tasks).

---

## Appendix B: Telemetry Schema (Atlas SDK Integration)

Atlas SDK provides comprehensive telemetry via `AtlasSessionTrace` and `AtlasStepTrace` (streamed to Postgres). We extend the existing schema using the `session_metadata` and `step.metadata` fields.

### What Atlas SDK Already Captures (Out-of-the-Box)

**Per-Session** (`AtlasSessionTrace`):
```json
{
  "task": "Create client with email validation",
  "final_answer": "...",
  "plan": {"steps": [...]},
  "steps": [...],
  "session_metadata": {
    "session_id": 42,
    "status": "succeeded",
    "student_learning": "Learned to validate email format before API calls",
    "teacher_learning": "Student needs explicit schema validation reminders",
    "execution_mode": "adaptive",
    "token_usage": {"total": 6820, "student": 5200, "teacher": 1620}
  }
}
```

**Per-Step** (`AtlasStepTrace`):
```json
{
  "step_id": 2,
  "description": "Create new client",
  "trace": "...",
  "output": "...",
  "tool": "create_new_client",
  "tool_params": {"name": "...", "email": "..."},
  "guidance": ["Validate email format using regex before API call"],
  "reward": {
    "score": 0.92,
    "rationale": "Email format validated; status enum correct",
    "judges": [
      {
        "identifier": "structured_validator",
        "score": 1.0,
        "rationale": "All required fields present, enums valid",
        "escalated": false
      },
      {
        "identifier": "llm_semantic_judge",
        "score": 0.85,
        "rationale": "Semantically correct client creation flow",
        "escalated": false
      }
    ]
  },
  "metadata": {},
  "artifacts": {}
}
```

### CRM-Specific Metadata Extensions

We populate `session_metadata` and `step.metadata` with CRM context:

**Session Metadata** (CRM-specific fields):
```json
{
  "session_metadata": {
    // ... Atlas SDK defaults (session_id, status, student_learning, etc.)
    "crm_context": {
      "golden_case_id": "gc_103",
      "task_category": "client_creation",
      "verification_mode": "email_validation",
      "baseline_verifier_score": 0.3,  // From baseline run (if available)
      "playbooks_retrieved": [
        {"id": "crm_client_validation_v3", "similarity": 0.89},
        {"id": "email_format_check_v1", "similarity": 0.76}
      ],
      "playbooks_adopted": ["crm_client_validation_v3"],
      "supervision_transitions": [
        {"step": 1, "from": "autonomous", "to": "step-by-step", "reason": "Novel task variant"}
      ]
    }
  }
}
```

**Step Metadata** (CRM-specific fields):
```json
{
  "metadata": {
    "crm_action": {
      "entity_type": "client",
      "operation": "create",
      "validator_checks": ["email_format", "status_enum", "required_fields"],
      "db_snapshot_pre": "snapshot_gc103_step1",
      "db_snapshot_post": "snapshot_gc103_step2"
    }
  }
}
```

### Learning Metrics (Post-Hoc Analysis via Atlas SDK)

Atlas SDK's `atlas.evaluation.learning_report` module (`PlaybookImpactEntry` dataclass) **already calculates all 5 learning metrics**:

```python
# scripts/eval_learning.py (from Atlas SDK)
# Filters sessions by project/task/tags, generates learning metrics

playbook_metrics = {
    "total_cue_hits": 69,              # Cue Hit Rate: 69/98 = 70%
    "adoption_events": 69,
    "successful_adoptions": 66,
    "adoption_rate": 0.957,            # Adoption Rate: 66/69 = 95.7%
    "average_reward_with": 0.541,
    "average_reward_without": 0.337,
    "reward_delta": 0.204,             # Reward Delta: +20.4pp
    "average_tokens_with": 78118,
    "average_tokens_without": 141660,
    "token_delta": -0.449,             # Token Delta: -44.9%
    "transfer_success": True,          # Transfer Success: Incident #55 uplift
    "unique_incidents": 2              # Cross-incident validation
}
```

**Usage** (Issue #4 analytics):
```bash
# Export sessions from Postgres
arc-atlas \
  --database-url postgresql://atlas:atlas@localhost:5433/atlas \
  --output crm_traces.jsonl \
  --limit 1500

# Generate learning report (Atlas SDK built-in)
python -m atlas.evaluation.learning_report \
  --input crm_traces.jsonl \
  --output learning_metrics.json \
  --format markdown

# Custom analysis for phased progression (Issue #4)
python scripts/analyze_learning_progression.py \
  --traces crm_traces.jsonl \
  --phases 3 \
  --output phases_report.md
```

### Postgres Storage (Atlas SDK)

All telemetry streams to Postgres via Atlas SDK's `atlas.runtime.storage.database` module:

**Tables** (auto-created by Atlas SDK):
- `atlas.sessions` – Session-level traces (task, final_answer, session_metadata)
- `atlas.steps` – Step-level traces (step_id, tool, guidance, reward, metadata)
- `atlas.learning_playbooks` – Distilled playbooks (Teacher/Student artifacts)
- `atlas.trajectory_events` – Fine-grained event stream (escalations, drift signals)

**Configuration** (`.env`):
```bash
DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas
STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas
```

### Real-Time Monitoring (Streaming)

During Atlas runs, telemetry streams in real-time:
- **Playbook retrieval**: Logged when PLM search triggers (per episode)
- **Teacher guidance**: Captured in `step.guidance` (per step)
- **Reward scores**: Calculated per step via RIM judges
- **Token counts**: Accumulated in `session_metadata.token_usage`

**Dashboard queries** (example for real-time monitoring):
```sql
-- Monitor playbook adoption rate (live)
SELECT
  COUNT(*) FILTER (WHERE session_metadata->'crm_context'->>'playbooks_adopted' IS NOT NULL) * 100.0 / COUNT(*) as adoption_rate_pct
FROM atlas.sessions
WHERE created_at > NOW() - INTERVAL '1 hour';

-- Track token efficiency progression (per phase)
SELECT
  AVG((session_metadata->'token_usage'->>'total')::int) as avg_tokens,
  COUNT(*) as episodes
FROM atlas.sessions
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY (row_number() OVER (ORDER BY created_at) - 1) / 25;  -- 25-episode phases
```

---

_Document owner:_ **Jarrod Barnes**
_Contributors:_ Aman Jaglan, Atlas SDK team, CRM Benchmark engineering
_Last reviewed:_ November 2, 2025
