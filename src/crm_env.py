"""Gymnasium-compatible environment that exposes the CRM sandbox.

CrmEnv wraps :class:`MockCrmApi` and the golden-case catalog as a reinforcement
learning environment. Episodes correspond to sampled golden cases and end when
the associated validator passes (success cases) or the configured step budget is
exhausted. The default reward is binary (0/1) with optional shaping hooks.

Observation schema (see :attr:`observation_space` for formal definition):
    - ``task``: metadata for the sampled golden case (IDs, text, expectations).
    - ``last_tool``: outcome of the most recent tool invocation (or neutral
      defaults after reset).
    - ``crm_summary``: counts of key CRM entities for lightweight state context.
    - ``steps_remaining``: how many steps remain before truncation.

Action schema (see :attr:`action_space`):
    - ``tool`` (int): index into ``ALLOWED_TOOLS``.
    - ``arguments`` (str): JSON object encoding the keyword arguments for the
      selected tool. ``CrmEnv.step`` also accepts structured dictionaries in
      addition to this canonical form; the JSON channel ensures Gymnasium agents
      with pure-text policies can still interoperate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import string
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

try:  # Prefer Gymnasium if available.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:  # pragma: no cover - fallback for legacy installs.
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "CrmEnv requires either the `gymnasium` package (preferred) or a modern `gym` install. "
            "Install with `pip install gymnasium`."
        ) from exc

try:  # Gymnasium re-exports seeding utilities from gym.
    from gymnasium.utils import seeding
except ImportError:  # pragma: no cover - legacy gym fallback.
    from gym.utils import seeding  # type: ignore

from .crm_backend import DatabaseConfig, PostgresCrmBackend
from .crm_sandbox import MockCrmApi
from .golden_cases import GOLDEN_CASES, GoldenCase
from .validators import CrmStateSnapshot, ValidationResult, VerificationMode, get_task_verification_mode
from .verifier import ToolTrace, Verifier, VerifierRequest, VerifierResult, get_registered_verifier


ALLOWED_TOOLS: Tuple[str, ...] = (
    "create_new_client",
    "create_new_opportunity",
    "create_quote",
    "upload_document",
    "modify_opportunity",
)

DEFAULT_TOOL_HINTS: Mapping[str, str] = {
    "create_new_client": (
        "create_new_client(name: str, email: str, status: str, **optional_fields) – "
        "Status must be exactly one of 'Active', 'Inactive', or 'Prospect'."
    ),
    "create_new_opportunity": (
        "create_new_opportunity(name: str, client_id: str, amount: float, stage: str, **optional_fields) – "
        "Stage must be one of 'Prospecting', 'Qualification', 'Proposal', 'Negotiation', 'Closed-Won', 'Closed-Lost'."
    ),
    "create_quote": (
        "create_quote(opportunity_id: str, amount: float, status: str, **optional_fields) – "
        "Quote status options: 'Draft', 'Sent', 'Approved', 'Rejected', 'Canceled'."
    ),
    "upload_document": (
        "upload_document(entity_type: str, entity_id: str, file_name: str, **optional_fields) – "
        "entity_type must be 'Client', 'Opportunity', 'Quote', or 'Contract'; file names must keep their extension."
    ),
    "modify_opportunity": (
        "modify_opportunity(opportunity_id: str, updates: Dict[str, Any]) – "
        "Updates may include 'stage', 'probability', 'amount', 'close_date', 'notes', or 'owner'; "
        "stage/probability must follow the CRM enum rules."
    ),
}

_TEXT_CHARSET = string.printable


class ActionDecodingError(ValueError):
    """Raised when an agent action cannot be interpreted."""


@dataclass(frozen=True)
class TaskSample:
    """Container describing the sampled golden case and derived metadata."""

    case: GoldenCase
    context: Dict[str, Any]
    expected_args: Dict[str, Any]


@dataclass
class RewardConfig:
    """Runtime-configurable reward shaping weights."""

    success: float = 1.0
    failure: float = 0.0
    tool_match_bonus: float = 0.0
    partial_progress: float = 0.0

    def clip(self, value: float) -> float:
        """Clamp rewards between configured failure/success bounds."""
        upper = max(self.success, self.failure)
        lower = min(self.success, self.failure)
        return float(np.clip(value, lower, upper))


class TaskManager:
    """Sample golden cases with optional filtering and deterministic seeding."""

    def __init__(
        self,
        cases: Sequence[GoldenCase] | None = None,
        *,
        include_negative_cases: bool = False,
        allowed_case_ids: Optional[Iterable[str]] = None,
    ) -> None:
        raw_cases = list(cases or GOLDEN_CASES)
        if not include_negative_cases:
            raw_cases = [case for case in raw_cases if case.expect_success]
        if allowed_case_ids is not None:
            allowed = set(allowed_case_ids)
            raw_cases = [case for case in raw_cases if case.case_id in allowed]

        if not raw_cases:
            raise ValueError("TaskManager requires at least one golden case to sample.")

        self._cases: List[GoldenCase] = raw_cases

    @property
    def cases(self) -> Sequence[GoldenCase]:
        """Return the currently active golden cases."""
        return tuple(self._cases)

    def sample(
        self,
        rng: np.random.Generator,
        *,
        case_id: Optional[str] = None,
        task: Optional[str] = None,
    ) -> GoldenCase:
        """Return a golden case determined by the provided parameters."""
        if case_id:
            for case in self._cases:
                if case.case_id == case_id:
                    return case
            raise ValueError(f"Unknown case_id '{case_id}'.")

        candidate_cases = self._cases
        if task:
            filtered = [case for case in candidate_cases if case.task == task]
            if not filtered:
                raise ValueError(f"No golden cases available for task '{task}'.")
            candidate_cases = filtered

        index = int(rng.integers(0, len(candidate_cases)))
        return candidate_cases[index]


class CrmEnv(gym.Env):
    """Gym/Gymnasium environment that surfaces the CRM sandbox."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        *,
        task_manager: Optional[TaskManager] = None,
        max_steps: int = 1,
        reward_config: Optional[RewardConfig] = None,
        include_negative_cases: bool = False,
        shaping_enabled: bool = False,
        seed: Optional[int] = None,
        expose_reference: bool = True,
        include_tool_hints: bool = False,
        tool_hint_map: Optional[Mapping[str, str]] = None,
        backend: str = "mock",
        db_config: Optional[DatabaseConfig] = None,
        reset_database_each_episode: Optional[bool] = None,
        enable_verifier: bool = False,
        verifier_name: Optional[str] = None,
        verifier_config: Optional[Mapping[str, Any]] = None,
        verifier_reward_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self._max_steps = int(max_steps)
        self._reward_config = reward_config or RewardConfig()
        self._shaping_enabled = shaping_enabled
        self._expose_reference = bool(expose_reference)

        self._np_random, _ = seeding.np_random(seed)
        self._task_manager = task_manager or TaskManager(include_negative_cases=include_negative_cases)

        resolved_hints = dict(tool_hint_map or DEFAULT_TOOL_HINTS)
        self._tool_hints: Optional[Mapping[str, str]] = resolved_hints if include_tool_hints else None

        self.action_space = spaces.Dict(
            {
                "tool": spaces.Discrete(len(ALLOWED_TOOLS)),
                "arguments": spaces.Text(min_length=0, max_length=4096, charset=_TEXT_CHARSET),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "task": spaces.Dict(
                    {
                        "case_id": spaces.Text(min_length=0, max_length=16, charset=_TEXT_CHARSET),
                        "task": spaces.Text(min_length=0, max_length=64, charset=_TEXT_CHARSET),
                        "description": spaces.Text(min_length=0, max_length=640, charset=_TEXT_CHARSET),
                        "expected_tool": spaces.Text(min_length=0, max_length=64, charset=_TEXT_CHARSET),
                        "expected_arguments": spaces.Text(min_length=0, max_length=2048, charset=_TEXT_CHARSET),
                    }
                ),
                "last_tool": spaces.Dict(
                    {
                        "tool": spaces.Text(min_length=0, max_length=64, charset=_TEXT_CHARSET),
                        "arguments": spaces.Text(min_length=0, max_length=2048, charset=_TEXT_CHARSET),
                        "success": spaces.Discrete(3),  # 0=not called, 1=success, 2=error
                        "error": spaces.Text(min_length=0, max_length=512, charset=_TEXT_CHARSET),
                        "tool_index": spaces.Box(
                            low=np.array([-1], dtype=np.int32),
                            high=np.array([len(ALLOWED_TOOLS) - 1], dtype=np.int32),
                            shape=(1,),
                            dtype=np.int32,
                        ),
                    }
                ),
                "crm_summary": spaces.Dict(
                    {
                        "clients": spaces.Box(low=0, high=512, shape=(1,), dtype=np.int32),
                        "contacts": spaces.Box(low=0, high=512, shape=(1,), dtype=np.int32),
                        "opportunities": spaces.Box(low=0, high=512, shape=(1,), dtype=np.int32),
                        "quotes": spaces.Box(low=0, high=512, shape=(1,), dtype=np.int32),
                        "contracts": spaces.Box(low=0, high=512, shape=(1,), dtype=np.int32),
                        "documents": spaces.Box(low=0, high=512, shape=(1,), dtype=np.int32),
                        "notes": spaces.Box(low=0, high=512, shape=(1,), dtype=np.int32),
                    }
                ),
                "steps_remaining": spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.int32),
            }
        )

        backend_mode = backend.lower().strip()
        if backend_mode not in {"mock", "postgres"}:
            raise ValueError("backend must be either 'mock' or 'postgres'.")
        self._backend_mode = backend_mode
        self._db_config = db_config
        if backend_mode == "postgres" and self._db_config is None:
            self._db_config = DatabaseConfig.from_env()
        self._db_backend: Optional[PostgresCrmBackend] = None
        if reset_database_each_episode is None:
            reset_database_each_episode = backend_mode == "postgres"
        self._reset_database_each_episode = bool(reset_database_each_episode)

        self._backend: Optional[Any] = None
        self._task_sample: Optional[TaskSample] = None
        self._pre_snapshot: Optional[CrmStateSnapshot] = None
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._last_result: Dict[str, Any] = self._neutral_last_result()
        self._history: List[Dict[str, Any]] = []
        self._verifier_enabled = bool(enable_verifier)
        self._default_verifier_name = verifier_name.strip() if verifier_name else None
        self._default_verifier_config: Dict[str, Any] = dict(verifier_config or {})
        self._verifier_reward_weight = float(np.clip(verifier_reward_weight, 0.0, 1.0))
        self._active_verifier: Optional[Verifier] = None
        self._active_verifier_name: Optional[str] = None
        self._active_verifier_config: Dict[str, Any] = {}
        self._last_verifier_result: Optional[VerifierResult] = None
        self._tool_traces: List[ToolTrace] = []
        self._last_reward_breakdown: Dict[str, Any] = self._empty_reward_breakdown()
        self._last_learning_signals: Dict[str, Any] = self._empty_learning_signals()
        self._last_validator_metadata: Dict[str, Any] = self._empty_validator_metadata()

    @property
    def max_steps(self) -> int:
        """Return the configured step cap."""
        return self._max_steps

    @property
    def active_case(self) -> Optional[GoldenCase]:
        """Expose the current golden case."""
        return self._task_sample.case if self._task_sample else None

    @property
    def active_context(self) -> Optional[Mapping[str, Any]]:
        """Return the setup context associated with the active case."""
        if not self._task_sample:
            return None
        return dict(self._task_sample.context)

    @property
    def history(self) -> Sequence[Mapping[str, Any]]:
        """Return a copy of the per-step telemetry history."""
        return tuple(self._history)

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Reseed both Gymnasium RNG and the task manager."""
        self._np_random, actual_seed = seeding.np_random(seed)
        return [int(actual_seed)]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Initialize a new episode and return the starting observation."""
        if seed is not None:
            self.seed(seed)

        if self._np_random is None:
            self.seed(None)

        self._teardown_backend()
        backend = self._setup_backend()
        self._backend = backend
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._history = []
        self._last_result = self._neutral_last_result()
        self._tool_traces = []
        self._last_verifier_result = None
        self._active_verifier = None
        self._active_verifier_name = None
        self._active_verifier_config = {}
        self._last_reward_breakdown = self._empty_reward_breakdown()
        self._last_learning_signals = self._empty_learning_signals()
        self._last_validator_metadata = self._empty_validator_metadata()

        options = dict(options or {})
        case = self._task_manager.sample(
            self._np_random,
            case_id=options.get("case_id"),
            task=options.get("task"),
        )

        context = case.setup(backend)
        expected_args = case.expected_args(context)
        self._task_sample = TaskSample(case=case, context=context, expected_args=expected_args)
        self._pre_snapshot = CrmStateSnapshot.from_backend(backend)
        self._activate_verifier(case)

        observation = self._build_observation()
        info = self._build_info(validator_result=None, verification_mode=get_task_verification_mode(case.task))
        info["expected_arguments"] = expected_args.copy() if self._expose_reference else None
        info["expected_tool_index"] = ALLOWED_TOOLS.index(case.expected_tool)
        info["expected_tool"] = case.expected_tool
        info["case_id"] = case.case_id
        info["task"] = case.task
        info["utterance"] = case.utterance
        info["description"] = case.description
        info["expected_success"] = case.expect_success
        if self._tool_hints is not None:
            info["tool_hints"] = dict(self._tool_hints)

        return observation, info

    def step(
        self,
        action: Union[Mapping[str, Any], str],
    ) -> Tuple[Mapping[str, Any], float, bool, bool, Mapping[str, Any]]:
        """Execute a tool call through the sandbox and return environment feedback."""
        if self._terminated or self._truncated:
            raise RuntimeError("Episode has already ended. Call reset() before step().")

        backend = self._backend
        if not backend or not self._task_sample or not self._pre_snapshot:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._step_count += 1
        case = self._task_sample.case
        verification_mode = get_task_verification_mode(case.task)
        pre_snapshot = self._pre_snapshot
        post_snapshot = self._pre_snapshot
        post_snapshot_for_verifier: Optional[CrmStateSnapshot] = pre_snapshot

        action_decoding_error: Optional[str] = None
        tool_name: Optional[str] = None
        arguments: Dict[str, Any] = {}
        tool_index: Optional[int] = None

        try:
            tool_name, arguments, tool_index = self._decode_action(action)
        except ActionDecodingError as exc:
            action_decoding_error = str(exc)

        execution_result: Optional[ValidationResult] = None
        validator_result: Optional[ValidationResult] = None
        reward = self._reward_config.failure
        terminated = False
        truncated = False
        tool_correct = False
        execution_success = False
        savepoint_name: Optional[str] = None

        if action_decoding_error is None and tool_name:
            tool_correct = tool_name == case.expected_tool
            if not tool_correct:
                execution_result = ValidationResult.fail(
                    f"Expected tool '{case.expected_tool}' but received '{tool_name}'."
                )
                post_snapshot = self._pre_snapshot
                post_snapshot_for_verifier = post_snapshot
            else:
                if self._using_database:
                    savepoint_name = backend.create_savepoint()
                execution_result = self._execute_tool(backend, tool_name, arguments)
                post_snapshot = CrmStateSnapshot.from_backend(backend)
                execution_success = execution_result.success
                post_snapshot_for_verifier = post_snapshot

            if case.expect_success:
                if execution_result.success and tool_correct:
                    validator_kwargs = case.validator_kwargs(self._task_sample.context, arguments)
                    if verification_mode is VerificationMode.DATABASE:
                        validator_result = case.validator(self._pre_snapshot, post_snapshot, arguments, **validator_kwargs)
                    else:
                        validator_result = ValidationResult.ok("Runtime evaluation succeeded.")
                    terminated = bool(validator_result.success)
                else:
                    validator_result = ValidationResult.fail(
                        execution_result.message,
                        execution_result.details if execution_result and not execution_result.success else None,
                    )
            else:
                if execution_result.success:
                    validator_result = ValidationResult.fail("Negative case expected failure but tool succeeded.")
                else:
                    substring_ok = (
                        case.expected_error_substring is None
                        or (execution_result.message and case.expected_error_substring in execution_result.message)
                    )
                    state_unchanged = self._pre_snapshot == post_snapshot
                    success = substring_ok and state_unchanged
                    msg = execution_result.message
                    details = dict(execution_result.details or {})
                    details["substring_match"] = substring_ok
                    details["state_unchanged"] = state_unchanged
                    validator_result = ValidationResult(success, msg, details)
                    terminated = success
                post_snapshot_for_verifier = post_snapshot

            if execution_result.success and tool_correct:
                if self._using_database and savepoint_name:
                    backend.release_savepoint(savepoint_name)
                self._pre_snapshot = post_snapshot
            else:
                if self._using_database and savepoint_name:
                    backend.rollback_to_savepoint(savepoint_name)
                self._restore_snapshot(self._pre_snapshot)
        else:
            validator_result = ValidationResult.fail(action_decoding_error or "Invalid tool selection.")
            execution_result = ValidationResult.fail(action_decoding_error or "Invalid tool selection.")

        self._last_result = {
            "tool": tool_name or "",
            "tool_index": tool_index if tool_index is not None else -1,
            "arguments": json.dumps(arguments, sort_keys=True),
            "success": 1 if execution_result and execution_result.success else (2 if execution_result else 0),
            "error": "" if execution_result and execution_result.success else (
                execution_result.message if execution_result else (action_decoding_error or "")
            ),
        }

        history_entry = {
            "step": self._step_count,
            "tool": tool_name,
            "tool_correct": tool_correct,
            "arguments": arguments,
            "execution_success": execution_result.success if execution_result else False,
            "validator_success": validator_result.success,
            "validator_message": validator_result.message,
        }
        self._history.append(history_entry)

        base_reward = self._reward_config.success if validator_result.success else self._reward_config.failure
        shaping_details = {
            "enabled": bool(self._shaping_enabled),
            "applied": False,
            "tool_match_bonus": 0.0,
            "partial_progress": 0.0,
            "delta": 0.0,
        }
        reward_after_shaping = base_reward
        if not validator_result.success and self._shaping_enabled:
            shaping_details["applied"] = True
            if tool_correct:
                updated_reward = max(reward_after_shaping, self._reward_config.tool_match_bonus)
                if updated_reward > reward_after_shaping:
                    shaping_details["tool_match_bonus"] = updated_reward - reward_after_shaping
                    reward_after_shaping = updated_reward
            if execution_success:
                updated_reward = max(reward_after_shaping, self._reward_config.partial_progress)
                if updated_reward > reward_after_shaping:
                    shaping_details["partial_progress"] = updated_reward - reward_after_shaping
                    reward_after_shaping = updated_reward
            shaping_details["delta"] = reward_after_shaping - base_reward
        reward = reward_after_shaping

        self._append_tool_trace(
            step=self._step_count,
            tool_name=tool_name,
            arguments=arguments,
            execution_result=execution_result,
            validator_result=validator_result,
        )
        verifier_result = self._invoke_active_verifier(
            case=case,
            pre_state=pre_snapshot,
            post_state=post_snapshot_for_verifier,
            validator_result=validator_result,
        )
        self._last_verifier_result = verifier_result

        verifier_contribution = 0.0
        verifier_score = verifier_result.score if verifier_result else 0.0
        if verifier_result and self._verifier_reward_weight > 0.0:
            success_reward = self._reward_config.success
            blended_reward = (1.0 - self._verifier_reward_weight) * reward + self._verifier_reward_weight * (
                verifier_score * success_reward
            )
            verifier_contribution = blended_reward - reward
            reward = blended_reward

        final_reward = reward
        clipped_reward = float(self._reward_config.clip(final_reward))

        self._last_reward_breakdown = {
            "base": float(base_reward),
            "shaping": {
                "enabled": shaping_details["enabled"],
                "applied": shaping_details["applied"],
                "tool_match_bonus": float(shaping_details["tool_match_bonus"]),
                "partial_progress": float(shaping_details["partial_progress"]),
                "delta": float(shaping_details["delta"]),
            },
            "verifier": {
                "enabled": bool(verifier_result and self._verifier_reward_weight > 0.0),
                "weight": self._verifier_reward_weight,
                "score": float(verifier_score),
                "contribution": float(verifier_contribution),
            },
            "final": clipped_reward,
        }

        self._last_validator_metadata = self._build_validator_metadata(
            validator_result=validator_result,
            action_decoding_error=action_decoding_error,
            tool_correct=tool_correct,
            execution_success=execution_success,
        )
        self._last_learning_signals = self._build_learning_signals(
            validator_result=validator_result,
            verifier_result=verifier_result,
            reward_breakdown=self._last_reward_breakdown,
        )

        history_entry["reward"] = clipped_reward
        history_entry["reward_breakdown"] = self._last_reward_breakdown
        history_entry["learning_signals"] = self._last_learning_signals

        truncated = not terminated and self._step_count >= self._max_steps
        self._terminated = terminated
        self._truncated = truncated

        observation = self._build_observation()
        info = self._build_info(validator_result=validator_result, verification_mode=verification_mode)
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["history"] = list(self._history)
        info["tool_correct"] = tool_correct
        info["reward"] = clipped_reward

        return observation, clipped_reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Render the latest step outcome (textual debug info only)."""
        if mode != "human":
            raise NotImplementedError("Only human render mode is supported.")
        if not self._task_sample:
            print("CrmEnv: no active episode.")
            return
        print(
            f"[CrmEnv] case={self._task_sample.case.case_id} "
            f"step={self._step_count} terminated={self._terminated} truncated={self._truncated} "
            f"last_tool={self._last_result['tool']} success={self._last_result['success'] == 1}"
        )

    def close(self) -> None:  # noqa: D401 - interface method
        """Release environment resources (no-op for in-memory sandbox)."""
        self._teardown_backend()
        self._backend = None
        if self._backend_mode == "postgres" and self._db_backend:
            self._db_backend.close()
            self._db_backend = None
        self._task_sample = None
        self._pre_snapshot = None
        self._history = []

    def _build_observation(self) -> Dict[str, Any]:
        if not self._task_sample:
            raise RuntimeError("Cannot build observation without active task.")
        case = self._task_sample.case
        summary = self._summarize_crm()
        expected_args = self._task_sample.expected_args if self._expose_reference else {}

        observation = {
            "task": {
                "case_id": case.case_id,
                "task": case.task,
                "description": case.description,
                "expected_tool": case.expected_tool,
                "expected_arguments": json.dumps(expected_args, sort_keys=True),
            },
            "last_tool": dict(self._last_result),
            "crm_summary": {
                "clients": np.array([summary["clients"]], dtype=np.int32),
                "contacts": np.array([summary["contacts"]], dtype=np.int32),
                "opportunities": np.array([summary["opportunities"]], dtype=np.int32),
                "quotes": np.array([summary["quotes"]], dtype=np.int32),
                "contracts": np.array([summary["contracts"]], dtype=np.int32),
                "documents": np.array([summary["documents"]], dtype=np.int32),
                "notes": np.array([summary["notes"]], dtype=np.int32),
            },
            "steps_remaining": np.array([max(self._max_steps - self._step_count, 0)], dtype=np.int32),
        }
        observation["last_tool"]["tool_index"] = np.array(
            [int(self._last_result.get("tool_index", -1))],
            dtype=np.int32,
        )
        return observation

    def _build_info(
        self,
        *,
        validator_result: Optional[ValidationResult],
        verification_mode: VerificationMode,
    ) -> Dict[str, Any]:
        if not self._task_sample:
            return {}
        info = {
            "case_id": self._task_sample.case.case_id,
            "task": self._task_sample.case.task,
            "verification_mode": verification_mode.value,
            "validator_message": validator_result.message if validator_result else "",
            "validator_success": validator_result.success if validator_result else False,
            "validator_details": dict(validator_result.details) if validator_result and validator_result.details else None,
            "expected_success": self._task_sample.case.expect_success,
        }
        if self._tool_hints is not None:
            info["tool_hints"] = dict(self._tool_hints)
        info["verifier_enabled"] = self._verifier_enabled
        info["verifier_weight"] = self._verifier_reward_weight
        info["verifier_name"] = self._active_verifier_name
        info["verifier_config"] = dict(self._active_verifier_config) if self._active_verifier_config else None
        if self._last_verifier_result:
            info["verifier_score"] = self._last_verifier_result.score
            info["verifier_rationale"] = self._last_verifier_result.rationale
            info["verifier_metadata"] = dict(self._last_verifier_result.metadata or {})
        else:
            info["verifier_score"] = None
            info["verifier_rationale"] = None
            info["verifier_metadata"] = None
        info["reward_breakdown"] = self._last_reward_breakdown
        info["learning_signals"] = self._last_learning_signals
        info["validator_metadata"] = self._last_validator_metadata
        info["drift_notes"] = self._last_learning_signals.get("drift_notes", "")
        return info

    def _activate_verifier(self, case: GoldenCase) -> None:
        """Instantiate the verifier for the active episode, if configured."""
        self._active_verifier = None
        self._active_verifier_name = None
        self._active_verifier_config = {}
        self._last_verifier_result = None
        if not self._verifier_enabled:
            return
        name = case.verifier_name or self._default_verifier_name
        if not name:
            return
        config = dict(self._default_verifier_config)
        if case.verifier_options:
            config.update(case.verifier_options)
        try:
            verifier = get_registered_verifier(name, config)
        except ValueError as exc:
            raise ValueError(f"Failed to initialize verifier '{name}' for case '{case.case_id}': {exc}") from exc
        self._active_verifier = verifier
        self._active_verifier_name = name
        self._active_verifier_config = config

    def _append_tool_trace(
        self,
        *,
        step: int,
        tool_name: Optional[str],
        arguments: Mapping[str, Any],
        execution_result: Optional[ValidationResult],
        validator_result: Optional[ValidationResult],
    ) -> None:
        """Persist the latest tool invocation for verifier consumption."""
        if not self._verifier_enabled:
            return
        safe_args = dict(arguments) if isinstance(arguments, Mapping) else {}
        message = ""
        if validator_result and validator_result.message:
            message = validator_result.message
        elif execution_result and execution_result.message:
            message = execution_result.message
        trace = ToolTrace(
            step=step,
            tool_name=tool_name,
            arguments=safe_args,
            execution_success=bool(execution_result.success) if execution_result else False,
            validator_success=bool(validator_result.success) if validator_result else False,
            message=message,
        )
        self._tool_traces.append(trace)

    def _invoke_active_verifier(
        self,
        *,
        case: GoldenCase,
        pre_state: Optional[CrmStateSnapshot],
        post_state: Optional[CrmStateSnapshot],
        validator_result: Optional[ValidationResult],
    ) -> Optional[VerifierResult]:
        """Run the active verifier, capturing exceptions as failure signals."""
        if not self._active_verifier or not self._verifier_enabled or not self._task_sample:
            return None
        try:
            request = VerifierRequest(
                case_id=case.case_id,
                task=case.task,
                utterance=case.utterance,
                expect_success=case.expect_success,
                expected_error_substring=case.expected_error_substring,
                expected_tool=case.expected_tool,
                expected_arguments=dict(self._task_sample.expected_args),
                tool_traces=tuple(self._tool_traces),
                final_response=None,
                validator_result=validator_result,
                pre_state=pre_state,
                post_state=post_state,
            )
            return self._active_verifier.evaluate(request)
        except Exception as exc:  # pragma: no cover - defensive guard
            return VerifierResult(
                score=0.0,
                rationale=f"Verifier '{self._active_verifier_name}' raised an exception: {exc}",
                metadata={"exception": repr(exc)},
            )

    def _empty_reward_breakdown(self) -> Dict[str, Any]:
        return {
            "base": float(self._reward_config.failure),
            "shaping": {
                "enabled": bool(self._shaping_enabled),
                "applied": False,
                "tool_match_bonus": 0.0,
                "partial_progress": 0.0,
                "delta": 0.0,
            },
            "verifier": {
                "enabled": bool(self._verifier_enabled and self._verifier_reward_weight > 0.0),
                "weight": self._verifier_reward_weight,
                "score": 0.0,
                "contribution": 0.0,
            },
            "final": float(self._reward_config.failure),
        }

    @staticmethod
    def _empty_learning_signals() -> Dict[str, Any]:
        return {
            "student": {"summary": "", "score": 0.0},
            "teacher": {"summary": "", "score": 0.0},
            "adapter_events": [],
            "reason": "",
            "drift_notes": "",
        }

    def _empty_validator_metadata(self) -> Dict[str, Any]:
        return {
            "success": False,
            "message": "",
            "details": None,
            "error_category": "unknown",
        }

    def _build_validator_metadata(
        self,
        *,
        validator_result: ValidationResult,
        action_decoding_error: Optional[str],
        tool_correct: bool,
        execution_success: bool,
    ) -> Dict[str, Any]:
        return {
            "success": bool(validator_result.success),
            "message": validator_result.message or "",
            "details": dict(validator_result.details) if validator_result.details else None,
            "error_category": self._classify_validator_error(
                validator_result=validator_result,
                action_decoding_error=action_decoding_error,
                tool_correct=tool_correct,
                execution_success=execution_success,
            ),
        }

    @staticmethod
    def _classify_validator_error(
        *,
        validator_result: ValidationResult,
        action_decoding_error: Optional[str],
        tool_correct: bool,
        execution_success: bool,
    ) -> str:
        if validator_result.success:
            return "success"
        if action_decoding_error:
            return "action_decoding_error"
        if not tool_correct:
            return "wrong_tool"
        if not execution_success:
            return "execution_failure"
        return "validator_failure"

    def _build_learning_signals(
        self,
        *,
        validator_result: ValidationResult,
        verifier_result: Optional[VerifierResult],
        reward_breakdown: Mapping[str, Any],
    ) -> Dict[str, Any]:
        student_score = 1.0 if validator_result.success else 0.0
        student_summary = validator_result.message or ""
        teacher_score = verifier_result.score if verifier_result else 0.0
        teacher_summary = verifier_result.rationale if verifier_result else ""
        reason = validator_result.message or ""
        final_reward = float(reward_breakdown.get("final", 0.0))
        return {
            "student": {"summary": student_summary, "score": float(student_score)},
            "teacher": {"summary": teacher_summary, "score": float(teacher_score)},
            "adapter_events": [],
            "reason": reason,
            "drift_notes": "",
            "reward_observation": final_reward,
        }

    @staticmethod
    def _neutral_last_result() -> Dict[str, Any]:
        return {"tool": "", "tool_index": -1, "arguments": "{}", "success": 0, "error": ""}

    def _setup_backend(self) -> Any:
        if self._backend_mode == "postgres":
            if not self._db_backend:
                self._db_backend = PostgresCrmBackend(self._db_config)
            self._db_backend.begin_session(reset=self._reset_database_each_episode)
            return self._db_backend
        return MockCrmApi()

    def _teardown_backend(self) -> None:
        if self._backend_mode == "postgres" and self._db_backend:
            self._db_backend.rollback_session()

    @property
    def _using_database(self) -> bool:
        return self._backend_mode == "postgres"

    def _summarize_crm(self) -> Dict[str, int]:
        backend = self._backend
        if not backend:
            return {
                "clients": 0,
                "contacts": 0,
                "opportunities": 0,
                "quotes": 0,
                "contracts": 0,
                "documents": 0,
                "notes": 0,
            }
        counts = backend.summarize_counts()
        return {
            "clients": int(counts.get("clients", 0)),
            "contacts": int(counts.get("contacts", 0)),
            "opportunities": int(counts.get("opportunities", 0)),
            "quotes": int(counts.get("quotes", 0)),
            "contracts": int(counts.get("contracts", 0)),
            "documents": int(counts.get("documents", 0)),
            "notes": int(counts.get("notes", 0)),
        }

    def _restore_snapshot(self, snapshot: CrmStateSnapshot) -> None:
        backend = self._backend
        if not backend or self._using_database:
            return

        def _deep_copy(store: Mapping[str, Any]) -> Dict[str, Any]:
            return {key: value.model_copy(deep=True) for key, value in store.items()}

        backend.clients = _deep_copy(snapshot.clients)
        backend.contacts = _deep_copy(snapshot.contacts)
        backend.opportunities = _deep_copy(snapshot.opportunities)
        backend.quotes = _deep_copy(snapshot.quotes)
        backend.contracts = _deep_copy(snapshot.contracts)
        backend.documents = _deep_copy(snapshot.documents)
        backend.notes = _deep_copy(snapshot.notes)

    def _decode_action(
        self,
        action: Union[Mapping[str, Any], str],
    ) -> Tuple[str, Dict[str, Any], int]:
        payload: MutableMapping[str, Any]
        if isinstance(action, str):
            try:
                parsed = json.loads(action)
            except json.JSONDecodeError as exc:
                raise ActionDecodingError(f"Action string must be JSON: {exc}") from exc
            if not isinstance(parsed, Mapping):
                raise ActionDecodingError("Parsed action must be a JSON object.")
            payload = dict(parsed)
        elif isinstance(action, Mapping):
            payload = dict(action)
        else:
            raise ActionDecodingError(f"Unsupported action type: {type(action)!r}")

        if "tool" in payload:
            tool_token = payload.pop("tool")
        elif "tool_index" in payload:
            tool_token = payload.pop("tool_index")
        elif "tool_name" in payload:
            tool_token = payload.pop("tool_name")
        else:
            raise ActionDecodingError("Action missing 'tool' / 'tool_index' / 'tool_name'.")

        tool_name: str
        tool_index: int
        if isinstance(tool_token, str):
            if tool_token not in ALLOWED_TOOLS:
                raise ActionDecodingError(f"Unknown tool '{tool_token}'.")
            tool_name = tool_token
            tool_index = ALLOWED_TOOLS.index(tool_name)
        else:
            try:
                tool_index = int(tool_token)
            except (TypeError, ValueError) as exc:
                raise ActionDecodingError("Tool token must be string or integer.") from exc
            if tool_index < 0 or tool_index >= len(ALLOWED_TOOLS):
                raise ActionDecodingError(f"Tool index {tool_index} out of range.")
            tool_name = ALLOWED_TOOLS[tool_index]

        arguments_token = payload.pop("arguments", payload.pop("arguments_json", None))
        if arguments_token is None:
            arguments: Dict[str, Any] = {}
        elif isinstance(arguments_token, str):
            try:
                parsed_args = json.loads(arguments_token)
            except json.JSONDecodeError as exc:
                raise ActionDecodingError(f"Arguments must be JSON string: {exc}") from exc
            if not isinstance(parsed_args, Mapping):
                raise ActionDecodingError("Decoded arguments must be a JSON object.")
            arguments = dict(parsed_args)
        elif isinstance(arguments_token, Mapping):
            arguments = dict(arguments_token)
        else:
            raise ActionDecodingError("Arguments must be mapping or JSON string.")

        if payload:
            # Surface unexpected extra keys early to ease debugging.
            raise ActionDecodingError(f"Unexpected action fields: {sorted(payload.keys())}.")

        return tool_name, arguments, tool_index

    @staticmethod
    def _execute_tool(api: Any, tool_name: str, arguments: Mapping[str, Any]) -> ValidationResult:
        try:
            tool = getattr(api, tool_name)
        except AttributeError:
            return ValidationResult.fail(f"Unknown tool '{tool_name}'.")

        try:
            # All methods now support uniform calling via **kwargs
            tool(**arguments)
        except Exception as exc:  # pragma: no cover - behaviour covered via validators.
            return ValidationResult.fail(str(exc))

        return ValidationResult.ok()


__all__ = [
    "ALLOWED_TOOLS",
    "CrmEnv",
    "RewardConfig",
    "TaskManager",
]
