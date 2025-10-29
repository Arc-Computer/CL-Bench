"""Example rollout for the CRM Gymnasium environment."""

from __future__ import annotations

import random
from pathlib import Path
from pprint import pprint
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.crm_env import CrmEnv


def main() -> None:
    env = CrmEnv(max_steps=2)
    observation, info = env.reset(seed=0)
    expected_tool = info["expected_tool_index"]
    expected_args = info["expected_arguments"]

    cumulative_reward = 0.0
    step = 0
    done = False

    while not done:
        if random.random() < 0.5:
            action = {"tool": expected_tool, "arguments": expected_args}
        else:
            action = {"tool": env.action_space["tool"].sample(), "arguments": "{}"}

        observation, reward, terminated, truncated, step_info = env.step(action)
        cumulative_reward += reward
        step += 1
        pprint(
            {
                "step": step,
                "action_tool": observation["last_tool"]["tool"],
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "validator_message": step_info["validator_message"],
            }
        )
        done = terminated or truncated

    print(f"Cumulative reward: {cumulative_reward:.2f}")
    env.close()

    try:
        from atlas.envs import register_env  # type: ignore
    except ImportError:
        print("atlas-sdk not installed; skipping register_env demonstration.")
    else:
        register_env("crm-env", CrmEnv)
        print("Registered 'crm-env' with atlas.envs.register_env.")


if __name__ == "__main__":
    main()
