# Chained Dataset Baseline Report

## Dataset Summary
- Source: `artifacts/conversations_multi_turn/20251106T152518Z/full/chains.jsonl` (seed=42, model=gpt-4.1-mini)
- Conversations: 1500 (success=900, failed=600)
- Conversation success rate: 60.0%
- Failure ratio: 40.0% (target 40.0% +/- 2.0%)
- Segment success rate: 82.0%

## Chain Performance
| Chain | Conversations | Success % | Failure % | Avg Turns | Avg Turns / Segment | Success Pattern |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CHAIN-006A | 270 | 100.0% | 0.0% | 2.00 | 2.00 | ✔ |
| CHAIN-006B | 180 | 0.0% | 100.0% | 2.00 | 2.00 | ✖ |
| CHAIN-003A | 270 | 100.0% | 0.0% | 8.00 | 2.67 | ✔, ✔, ✔ |
| CHAIN-003B | 180 | 0.0% | 100.0% | 7.00 | 2.33 | ✔, ✖, ✔ |
| CHAIN-002A | 54 | 100.0% | 0.0% | 10.00 | 3.33 | ✔, ✔, ✔ |
| CHAIN-002B | 36 | 0.0% | 100.0% | 10.00 | 3.33 | ✔, ✖, ✔ |
| CHAIN-004A | 54 | 100.0% | 0.0% | 12.00 | 4.00 | ✔, ✔, ✔ |
| CHAIN-004B | 36 | 0.0% | 100.0% | 11.00 | 3.67 | ✔, ✖, ✔ |
| CHAIN-008A | 54 | 100.0% | 0.0% | 7.00 | 3.50 | ✔, ✔ |
| CHAIN-008B | 36 | 0.0% | 100.0% | 6.00 | 3.00 | ✔, ✖ |
| CHAIN-009A | 54 | 100.0% | 0.0% | 7.00 | 3.50 | ✔, ✔ |
| CHAIN-009B | 36 | 0.0% | 100.0% | 6.00 | 3.00 | ✔, ✖ |
| CHAIN-010A | 54 | 100.0% | 0.0% | 8.00 | 4.00 | ✔, ✔ |
| CHAIN-010B | 36 | 0.0% | 100.0% | 8.00 | 4.00 | ✔, ✖ |
| CHAIN-001A | 45 | 100.0% | 0.0% | 15.00 | 5.00 | ✔, ✔, ✔ |
| CHAIN-001B | 30 | 0.0% | 100.0% | 15.00 | 5.00 | ✔, ✖, ✔ |
| CHAIN-005A | 45 | 100.0% | 0.0% | 15.00 | 5.00 | ✔, ✔, ✔ |
| CHAIN-005B | 30 | 0.0% | 100.0% | 15.00 | 5.00 | ✔, ✔, ✖ |

## Failure Categories
- See manifest for aggregated failure signatures.

## Baseline Logs
- `artifacts/conversations_multi_turn/20251106T152518Z/full/run.log` (available)

## Expected Failure Coverage
- CHAIN-006B: 180/180 conversations intentionally fail
- CHAIN-003B: 180/180 conversations intentionally fail
- CHAIN-002B: 36/36 conversations intentionally fail
- CHAIN-004B: 36/36 conversations intentionally fail
- CHAIN-008B: 36/36 conversations intentionally fail
- CHAIN-009B: 36/36 conversations intentionally fail
- CHAIN-010B: 36/36 conversations intentionally fail
- CHAIN-001B: 30/30 conversations intentionally fail
- CHAIN-005B: 30/30 conversations intentionally fail