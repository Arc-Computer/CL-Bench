# Scenario Coverage Report

## Executive Summary

- **Total Scenarios**: 493
- **Success Scenarios**: 296 (60.0%)
- **Failure Scenarios**: 197 (40.0%)
- **Target Ratio**: 60% success / 40% failure
- **Deviation from Target**: 0.0%

## Per-Tool Breakdown

| Tool | Total | Success | Failure | Success Ratio |
|------|-------|---------|---------|---------------|
| add_note | 1 | 1 | 0 | 100.0% |
| cancel_quote | 6 | 2 | 4 | 33.3% |
| client_search | 22 | 19 | 3 | 86.4% |
| clone_opportunity | 6 | 3 | 3 | 50.0% |
| company_search | 2 | 1 | 1 | 50.0% |
| compare_quote_details | 6 | 4 | 2 | 66.7% |
| compare_quotes | 8 | 4 | 4 | 50.0% |
| contact_search | 18 | 10 | 8 | 55.6% |
| create_contract | 4 | 3 | 1 | 75.0% |
| create_new_client | 41 | 22 | 19 | 53.7% |
| create_new_contact | 13 | 7 | 6 | 53.8% |
| create_new_opportunity | 120 | 68 | 52 | 56.7% |
| create_quote | 31 | 17 | 14 | 54.8% |
| delete_opportunity | 5 | 2 | 3 | 40.0% |
| delete_quote | 7 | 1 | 6 | 14.3% |
| modify_client | 21 | 11 | 10 | 52.4% |
| modify_contact | 13 | 8 | 5 | 61.5% |
| modify_opportunity | 54 | 34 | 20 | 63.0% |
| modify_quote | 33 | 23 | 10 | 69.7% |
| opportunity_details | 1 | 1 | 0 | 100.0% |
| opportunity_search | 29 | 18 | 11 | 62.1% |
| quote_details | 12 | 10 | 2 | 83.3% |
| quote_search | 8 | 7 | 1 | 87.5% |
| summarize_opportunities | 5 | 3 | 2 | 60.0% |
| upload_document | 5 | 3 | 2 | 60.0% |
| view_opportunity_details | 22 | 14 | 8 | 63.6% |

## Cross-Reference with Agent Tasks CSV

✅ All high-frequency CSV tasks have corresponding scenarios

### Unexpected Tools (Not in CSV)

create_contract

## Metadata Completeness Analysis

| Entity | Total | Coverage Highlights |
|--------|-------|---------------------|
| Client | 64 | Owner, industry, and branded email populated for 64/64 records |
| Contact | 46 | First/last name, title, and email populated for 46/46 records |
| Opportunity | 52 | Stage, owner, probability, and amount populated for 52/52 records |
| Quote | 30 | Amount and status populated for 30/30 records |

- Deterministic enrichment now derives natural names, roles, and domains from canonical scenario data.
- Mock CRM seeding no longer relies on placeholder owners or contacts; missing metadata raises during generation.
- Scenario tags include per-entity identifiers so chained workflows can maintain context across segments.

## Scenario Tagging for Chained Workflows

- `ScenarioRepository.scenario_tags` surfaces intent, primary entity, stage/status, and contact roles for every scenario.
- `ScenarioRepository.find_scenarios(expected_tool=..., tag_filters=...)` enables chaining to request specific stages (e.g., `{"opportunity_stage": "Negotiation"}`) while honoring success/failure splits.
- `chain_conversation_generator` now forwards tag metadata to the Curator selector, so LLM-based sampling can target precise entities without manual overrides.

## Template References

Scenarios with template references (`{{turn_N.field}}`): 0

## Recommendations

- ✅ Scenario corpus meets all quality targets
