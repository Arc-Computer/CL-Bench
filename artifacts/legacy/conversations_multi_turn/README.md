# Multi-Turn Conversation Datasets

This directory hosts the validated chained benchmark conversations.

- `20251107T134304Z/` – cleaned evaluation drop (600 eval / 400 holdout success-path conversations + optional stress + reserve). Expected responses were normalized to match tool payloads, and lint/manifest artifacts live under `full/`.
- `20251106T152518Z/` – prior 60/40 success/failure drop retained for audit only.
- Root-level `chains.jsonl` is kept for backward compatibility; prefer the timestamped directories for customer deliverables.

Use the commands recorded in each `<timestamp>/full/` README to regenerate or verify a drop. Keep older timestamps only when auditing historical runs; otherwise promote the most recent validated directory.
