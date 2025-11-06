# Chained Dataset Baseline Report

## Dataset Summary
- Source: `artifacts/conversations_multi_turn/20251105T144453Z/full/chains.jsonl` (seed=42, model=gpt-4.1-mini)
- Conversations: 1500 (success=900, failed=600)
- Conversation success rate: 60.0%
- Failure ratio: 40.0% (target 40.0% +/- 2.0%)
- Segment success rate: 73.3%

## Chain Performance
| Chain | Conversations | Success % | Failure % | Avg Turns | Avg Turns / Segment | Success Pattern |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CHAIN-006A | 540 | 100.0% | 0.0% | 2.00 | 2.00 | ✔ |
| CHAIN-006B | 360 | 0.0% | 100.0% | 2.00 | 2.00 | ✖ |
| CHAIN-007A | 270 | 100.0% | 0.0% | 6.00 | 3.00 | ✔, ✔ |
| CHAIN-007B | 180 | 0.0% | 100.0% | 5.00 | 2.50 | ✔, ✖ |
| CHAIN-001A | 18 | 100.0% | 0.0% | 15.00 | 5.00 | ✔, ✔, ✔ |
| CHAIN-001B | 12 | 0.0% | 100.0% | 15.00 | 5.00 | ✔, ✖, ✔ |
| CHAIN-002A | 18 | 100.0% | 0.0% | 10.00 | 3.33 | ✔, ✔, ✔ |
| CHAIN-002B | 12 | 0.0% | 100.0% | 10.00 | 3.33 | ✔, ✖, ✔ |
| CHAIN-003A | 18 | 100.0% | 0.0% | 8.00 | 2.67 | ✔, ✔, ✔ |
| CHAIN-003B | 12 | 0.0% | 100.0% | 7.00 | 2.33 | ✔, ✖, ✔ |
| CHAIN-004A | 18 | 100.0% | 0.0% | 12.00 | 4.00 | ✔, ✔, ✔ |
| CHAIN-004B | 12 | 0.0% | 100.0% | 11.00 | 3.67 | ✔, ✖, ✔ |
| CHAIN-005A | 18 | 100.0% | 0.0% | 15.00 | 5.00 | ✔, ✔, ✔ |
| CHAIN-005B | 12 | 0.0% | 100.0% | 15.00 | 5.00 | ✔, ✔, ✖ |

## Failure Categories
- See manifest for aggregated failure signatures.

## Baseline Logs
- `artifacts/conversations_multi_turn/20251105T144453Z/full/run.log` (available)

## Expected Failure Coverage
- CHAIN-006B: 360/360 conversations intentionally fail
- CHAIN-007B: 180/180 conversations intentionally fail
- CHAIN-001B: 12/12 conversations intentionally fail
- CHAIN-002B: 12/12 conversations intentionally fail
- CHAIN-003B: 12/12 conversations intentionally fail
- CHAIN-004B: 12/12 conversations intentionally fail
- CHAIN-005B: 12/12 conversations intentionally fail