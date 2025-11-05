# Chained Dataset Baseline Report

## Dataset Summary
- Source: `artifacts/conversations_chains/20251104T222756Z/full/chains.jsonl` (seed=42, model=gpt-4.1-mini)
- Conversations: 200 (success=200, failed=0)
- Conversation success rate: 100.0%
- Segment success rate: 100.0%

## Chain Performance
| Chain | Conversations | Success Rate | Avg Turns | Avg Turns / Segment | Success Pattern |
| --- | ---: | ---: | ---: | ---: | --- |
| CHAIN-001 | 40 | 100.0% | 15.00 | 5.00 | ✔, ✔, ✔ |
| CHAIN-002 | 40 | 100.0% | 10.00 | 3.33 | ✔, ✔, ✔ |
| CHAIN-003 | 40 | 100.0% | 8.00 | 2.67 | ✔, ✔, ✔ |
| CHAIN-004 | 40 | 100.0% | 12.00 | 4.00 | ✔, ✔, ✔ |
| CHAIN-005 | 40 | 100.0% | 15.00 | 5.00 | ✔, ✔, ✔ |

## Failure Categories
- No conversation-level failures observed in this run.

## Baseline Logs
- No baseline logs provided; refresh scheduled for Phase 6.

## Outstanding Expected-Failure Cases
- Pending enumeration; populate once failure chains are introduced.