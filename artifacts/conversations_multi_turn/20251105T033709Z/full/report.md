# Chained Dataset Baseline Report

## Dataset Summary
- Source: `artifacts/conversations_multi_turn/20251105T033709Z/full/chains.jsonl` (seed=42, model=gpt-4.1-mini)
- Conversations: 200 (success=120, failed=80)
- Conversation success rate: 60.0%
- Failure ratio: 40.0% (target 40.0% +/- 2.0%)
- Segment success rate: 86.7%

## Chain Performance
| Chain | Conversations | Success % | Failure % | Avg Turns | Avg Turns / Segment | Success Pattern |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CHAIN-001A | 24 | 100.0% | 0.0% | 15.00 | 5.00 | ✔, ✔, ✔ |
| CHAIN-001B | 16 | 0.0% | 100.0% | 15.00 | 5.00 | ✔, ✖, ✔ |
| CHAIN-002A | 24 | 100.0% | 0.0% | 10.00 | 3.33 | ✔, ✔, ✔ |
| CHAIN-002B | 16 | 0.0% | 100.0% | 10.00 | 3.33 | ✔, ✖, ✔ |
| CHAIN-003A | 24 | 100.0% | 0.0% | 8.00 | 2.67 | ✔, ✔, ✔ |
| CHAIN-003B | 16 | 0.0% | 100.0% | 7.00 | 2.33 | ✔, ✖, ✔ |
| CHAIN-004A | 24 | 100.0% | 0.0% | 12.00 | 4.00 | ✔, ✔, ✔ |
| CHAIN-004B | 16 | 0.0% | 100.0% | 11.00 | 3.67 | ✔, ✖, ✔ |
| CHAIN-005A | 24 | 100.0% | 0.0% | 15.00 | 5.00 | ✔, ✔, ✔ |
| CHAIN-005B | 16 | 0.0% | 100.0% | 15.00 | 5.00 | ✔, ✔, ✖ |

## Failure Categories
- See manifest for aggregated failure signatures.

## Baseline Logs
- `artifacts/conversations_multi_turn/20251105T033709Z/full/run.log` (available)

## Expected Failure Coverage
- CHAIN-001B: 16/16 conversations intentionally fail
- CHAIN-002B: 16/16 conversations intentionally fail
- CHAIN-003B: 16/16 conversations intentionally fail
- CHAIN-004B: 16/16 conversations intentionally fail
- CHAIN-005B: 16/16 conversations intentionally fail