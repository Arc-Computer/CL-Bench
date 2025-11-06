# Baseline Evaluation Outputs

This directory stores evaluation logs and JSONL outputs produced by `src.evaluation.run_baseline`.

- `*.jsonl`: per-conversation results emitted by the harness.
- `*.log`: stdout/stderr from the baseline command (capture LLM judge notices, warnings, etc.).

When collecting metrics, include the timestamp/model name in filenames (e.g., `20251106_gpt4.1_medium.jsonl`).
