"""Lightweight tests for judge utilities."""

from __future__ import annotations

from src.judge import build_case_digest


def test_build_case_digest_contains_case_ids() -> None:
    digest = build_case_digest()
    assert "Golden-Case Coverage" in digest
    assert "CNC-001" in digest
