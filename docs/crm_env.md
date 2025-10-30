# CRM Reinforcement-Learning Environment

The CRM benchmark now exposes a Gymnasium-compatible environment so continual-learning loops can interact with the sandbox instead of replaying static golden cases. The environment wraps the `MockCrmApi` tools, golden-case metadata, and validator suite into a step-based API that matches `gymnasium.Env` (and falls back to classic Gym at runtime if Gymnasium is not available).

> **Import path:** `from src.crm_env import CrmEnv, TaskManager, RewardConfig`

> **Dependencies:** Install `gymnasium` (preferred) or a compatible `gym` release before importing the environment:
> ```bash
> pip install gymnasium
> ```

## Postgres Sandbox Deployment

Phase A of `CONTINUAL_LEARNING_PLAN.md` introduces a Dockerized Postgres backend that mirrors the in-memory schema. Bring it up alongside the environment to exercise agents against a realistic datastore:

1. `cp .env.example .env` – provides `crm_app/ crm_password` defaults and optional pgAdmin credentials.
2. `docker compose up -d` – starts Postgres (`db`) plus pgAdmin (`pgadmin`). On first boot the containers run `sql/01_schema.sql` and `sql/02_seed_data.sql`, loading the clients, opportunities, quotes, documents, and notes referenced in the golden cases' happy paths.
3. `./scripts/db_seed.sh` – re-apply the seed data after resets; uses the same environment variables as Compose.
4. Connect via `psql postgresql://crm_app:crm_password@localhost:5432/crm_sandbox` or browse `http://localhost:8080` (default pgAdmin).

Issue #9 will adapt the tool layer (`MockCrmApi` replacements) to read/write this database while tests continue to rely on the in-memory mock for speed.

## Observation Schema

Each observation is a dictionary that conforms to `CrmEnv.observation_space`:

| Key | Type | Description |
| --- | --- | --- |
| `task.case_id` | `str` | Golden-case identifier (e.g., `"CNC-001"`). |
| `task.task` | `str` | Canonical task name (e.g., `"create_new_client"`). |
| `task.description` | `str` | Short human-readable description of the goal. |
| `task.expected_tool` | `str` | Tool the validator expects for the case. |
| `task.expected_arguments` | `str (JSON)` | Reference payload encoded as JSON. When `expose_reference=False`, this is an empty JSON object (`"{}"`). |
| `last_tool.tool` | `str` | Tool name invoked on the previous step. Empty after reset. |
| `last_tool.arguments` | `str (JSON)` | Arguments used on the previous step. |
| `last_tool.success` | `int` | `0` before any call, `1` on success, `2` on error. |
| `last_tool.error` | `str` | Error string when the previous invocation failed. |
| `crm_summary.*` | `int` | Counts for clients, contacts, opportunities, quotes, contracts, documents, and notes. |
| `steps_remaining` | `int` | Steps left before the episode truncates. |

All numeric summaries are emitted as `np.int32` arrays to match Gymnasium’s `spaces.Box`.

## Action Schema

`CrmEnv.action_space` is a `spaces.Dict` with:

- `tool` – `spaces.Discrete(len(ALLOWED_TOOLS))`; indices map to `src.crm_env.ALLOWED_TOOLS`.
- `arguments` – `spaces.Text(max_length=4096)`; JSON object describing keyword arguments.

To make scripted agents easier to author, `CrmEnv.step` also accepts structured dictionaries:

```python
env.step({"tool": "create_new_client", "arguments": {"name": "...", "email": "...", "status": "Active"}})
```

If a raw string is supplied, it is parsed as JSON. Invalid actions produce reward `0.0`, surface a descriptive error in both the observation and `info`, and leave the CRM state unchanged.

## Reward Model

- Default reward: `1.0` when the validator succeeds; `0.0` otherwise.
- Optional shaping is controlled via `RewardConfig` (e.g., `tool_match_bonus`, `partial_progress`). Enable it by passing `shaping_enabled=True` and a custom `RewardConfig`.

## Episode Management

- `max_steps`: cap on steps before the environment returns `truncated=True`. Defaults to `1` but can be increased for multi-step composites.
- `TaskManager`: supports deterministic sampling (via `seed`), filtering to explicit case IDs, and optionally including failure-oriented golden cases (`include_negative_cases=True`).
- `reset(..., options={"case_id": "...", "task": "..."})`: force a specific golden case or task family.
- `info["history"]`: cumulative record of tool calls, validator outcomes, and intermediate status for logging pipelines.

## Configuration Flags

- `expose_reference` (default `True`): when set to `False`, observations and `info` omit the ground-truth `expected_arguments`, forcing agents to infer parameters from natural language and context.
- `include_tool_hints` (default `False`): when `True`, `CrmEnv.reset` includes `info["tool_hints"]`, a map of tool names to signature-style descriptions that can be inserted into prompts.
- `CrmEnv.active_context`: property exposing the setup context (clients, opportunities, etc.) generated for the active episode—useful for prompt construction alongside the user utterance.

## Example Rollout

```python
from pprint import pprint
import random

from src.crm_env import CrmEnv

env = CrmEnv(max_steps=2)
observation, info = env.reset(seed=0)
expected_tool = info["expected_tool_index"]
expected_args = info["expected_arguments"]
cumulative_reward = 0.0

while True:
    if random.random() < 0.5:
        action = {"tool": expected_tool, "arguments": expected_args}
    else:
        action = {"tool": env.action_space["tool"].sample(), "arguments": "{}"}
    observation, reward, terminated, truncated, info = env.step(action)
    cumulative_reward += reward
    pprint({"reward": reward, "terminated": terminated, "truncated": truncated, "validator": info["validator_message"]})
    if terminated or truncated:
        break

print(f"Cumulative reward: {cumulative_reward:.2f}")
env.close()
```

## Atlas SDK Registration (Stub)

```python
try:
    from atlas.envs import register_env
except ImportError:
    register_env = None

if register_env:
    register_env("crm-env", CrmEnv)
```

## Validation Checklist

Before opening a PR that touches the environment:

1. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest`
2. `python examples/run_crm_env.py`
3. If you add new dependencies (e.g., `gymnasium`), update your installation instructions accordingly.

## Running With Live LLM Agents

Use `examples/run_crm_env_with_llm.py` to execute multiple episodes against a real model and log telemetry:

```bash
python examples/run_crm_env_with_llm.py --provider openai --model gpt-4.1 --episodes 5 --log-json artifacts/live_rollouts.jsonl
```

Key features:

- Automatically hides reference arguments by instantiating `CrmEnv(..., expose_reference=False)`.
- Reuses the existing harness prompt (`src.harness.build_prompt`) to keep prompting consistent with leaderboard evaluations.
- Logs per-step rewards, validator messages, and success flags to stdout and (optionally) a JSONL file for further analysis.
- Supports Anthropic (`--provider anthropic`) and a mock offline agent (`--provider mock`) for dry runs without API calls.
