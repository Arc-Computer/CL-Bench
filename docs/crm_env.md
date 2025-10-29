# CRM Reinforcement-Learning Environment

The CRM benchmark now exposes a Gymnasium-compatible environment so continual-learning loops can interact with the sandbox instead of replaying static golden cases. The environment wraps the `MockCrmApi` tools, golden-case metadata, and validator suite into a step-based API that matches `gymnasium.Env` (and falls back to classic Gym at runtime if Gymnasium is not available).

> **Import path:** `from src.crm_env import CrmEnv, TaskManager, RewardConfig`

## Observation Schema

Each observation is a dictionary that conforms to `CrmEnv.observation_space`:

| Key | Type | Description |
| --- | --- | --- |
| `task.case_id` | `str` | Golden-case identifier (e.g., `"CNC-001"`). |
| `task.task` | `str` | Canonical task name (e.g., `"create_new_client"`). |
| `task.description` | `str` | Short human-readable description of the goal. |
| `task.expected_tool` | `str` | Tool the validator expects for the case. |
| `task.expected_arguments` | `str (JSON)` | Reference payload encoded as JSON. |
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
