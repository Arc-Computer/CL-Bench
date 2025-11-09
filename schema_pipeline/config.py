from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class GeminiConfig:
    """Model + backend defaults for Gemini-powered Curator blocks."""

    workflow_model: str = "xai/grok-4-fast"
    argument_model: str = "xai/grok-4-fast"
    utterance_model: str = "xai/grok-4-fast"
    judge_model: str = "xai/grok-4-fast"
    backend: str = "litellm"
    backend_params: Dict[str, float] = field(
        default_factory=lambda: {"max_requests_per_minute": 1500, "max_tokens_per_minute": 3_000_000}
    )
    generation_params: Dict[str, float] = field(
        default_factory=lambda: {"temperature": 0.6, "top_p": 0.9}
    )
    viewer_enabled: bool = False


@dataclass
class PipelineConfig:
    """Shared configuration for schema-first generation."""

    schema_path: Path = Path("data/fake_crm_tables_schema.json")
    tasks_csv: Path = Path("data/Agent_tasks.csv")
    output_dir: Path = Path("artifacts/schema_pipeline")
    max_retries: int = 3
    viewer_session_id: Optional[str] = None
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
