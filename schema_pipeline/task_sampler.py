from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TaskDescriptor:
    """Weighted CRM task metadata derived from Agent_tasks.csv."""

    name: str
    verification_mode: str
    weight: float
    intent: str
    typical_user_phrasing: str
    example_sub_actions: str

    @property
    def normalized_actions(self) -> List[str]:
        if not self.example_sub_actions:
            return []
        return [action.strip() for action in self.example_sub_actions.split(",")]


class TaskSampler:
    """Sample CRM tasks based on production-like frequency weights."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self._tasks: List[TaskDescriptor] = self._load_tasks()
        self._weights = [task.weight for task in self._tasks]
        total = sum(self._weights)
        self._probabilities = [weight / total for weight in self._weights]

    def _load_tasks(self) -> List[TaskDescriptor]:
        tasks: List[TaskDescriptor] = []
        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                count_raw = row.get("count", "").strip()
                try:
                    count = float(count_raw)
                except ValueError:
                    count = 1.0
                tasks.append(
                    TaskDescriptor(
                        name=row.get("task_description", "").strip(),
                        verification_mode=row.get("verification_mode", "").strip(),
                        weight=count,
                        intent=row.get("intent", "").strip(),
                        typical_user_phrasing=row.get("typical_user_phrasing", "").strip(),
                        example_sub_actions=row.get("example_sub_actions", "").strip(),
                    )
                )
        if not tasks:
            raise ValueError(f"No tasks found in {self.csv_path}")
        return tasks

    @property
    def tasks(self) -> List[TaskDescriptor]:
        return list(self._tasks)

    def sample(self, k: int, *, rng: Optional[random.Random] = None) -> List[TaskDescriptor]:
        if k <= 0:
            return []
        rng = rng or random.Random()
        cumulative = []
        total = 0.0
        for weight in self._probabilities:
            total += weight
            cumulative.append(total)

        samples: List[TaskDescriptor] = []
        for _ in range(k):
            r = rng.random()
            for idx, threshold in enumerate(cumulative):
                if r <= threshold:
                    samples.append(self._tasks[idx])
                    break
        return samples

    def as_distribution(self) -> List[Dict[str, str]]:
        return [
            {
                "task": task.name,
                "probability": f"{prob:.6f}",
                "intent": task.intent,
                "verification_mode": task.verification_mode,
            }
            for task, prob in zip(self._tasks, self._probabilities)
        ]
