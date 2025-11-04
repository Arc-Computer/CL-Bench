"""Atlas SDK integration stubs for continual learning."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def prepare_playbook_payload(conversation_logs: Sequence[Path]) -> None:
    """Placeholder for Atlas playbook integration.

    This function will stream conversation transcripts, verifier rationales, and
    reward signals into the Atlas SDK once the continual learning loop is wired
    up. Implementations should:

    1. Parse the provided JSONL log files.
    2. Summarize improvements/deltas into Atlas playbook entries.
    3. Persist playbooks using `atlas.learning.playbook` APIs.
    """

    raise NotImplementedError("Atlas playbook integration is pending implementation.")


def replay_conversations(dataset_path: Path) -> None:
    """Placeholder hook for Atlas-powered ablations using the benchmark dataset."""

    raise NotImplementedError("Atlas replay integration is pending implementation.")
