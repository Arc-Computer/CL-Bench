# Scenario Generation Coverage Report

## Summary

- **Total Scenarios**: 100
- **Success Scenarios**: 60 (60.0%)
- **Failure Scenarios**: 40 (40.0%)

## Distribution by Task

| Task | Count |
|------|-------|
| create_new_opportunity | 21 |
| modify_opportunity | 12 |
| client_search | 9 |
| create_new_client | 9 |
| opportunity_search | 5 |
| view_opportunity_details | 5 |
| modify_quote | 4 |
| contact_search | 4 |
| create_new_contact | 4 |
| modify_client | 4 |
| create_quote | 4 |
| quote_details | 4 |
| modify_contact | 3 |
| create_contract | 2 |
| cancel_quote | 2 |
| delete_opportunity | 2 |
| compare_quotes | 2 |
| contract_search | 1 |
| delete_quote | 1 |
| add_note | 1 |
| clone_opportunity | 1 |

## Distribution by Intent Category

| Intent Category | Count |
|----------------|-------|
| Opportunity Management | 46 |
| Client Management | 22 |
| Quote Management | 17 |
| Contact Management | 11 |
| Contract Management | 3 |
| Notes & Collaboration | 1 |

## Distribution by Failure Category

| Failure Category | Count |
|-----------------|-------|

## Enum/Stage Coverage

### Opportunity Stage Distribution

| Stage | Count |
|-------|-------|
| Proposal | 18 |
| Negotiation | 15 |
| Qualification | 9 |
| Closed-Won | 6 |
| Prospecting | 5 |
| Draft | 1 |
| Proposal,Qualification | 1 |

### Quote Status Distribution

| Status | Count |
|--------|-------|
| Sent | 5 |
| Draft | 3 |

### Client Status Distribution

| Status | Count |
|--------|-------|
| Active | 11 |
| Prospect | 8 |

### Contract Status Distribution

| Status | Count |
|--------|-------|
| Pending Signature | 2 |
| Draft | 1 |

## Frequency Alignment with Source Taxonomy

| Task | CSV Frequency | Expected Count | Generated Count | Deviation |
|------|--------------|----------------|-----------------|------------|
| quote_details | 1 | 0 | 4 | +60624.0% ⚠️ |
| delete_opportunity | 1 | 0 | 2 | +30262.0% ⚠️ |
| compare_quotes | 1 | 0 | 2 | +30262.0% ⚠️ |
| modify_contact | 2 | 0 | 3 | +22671.5% ⚠️ |
| modify_quote | 3 | 0 | 4 | +20141.3% ⚠️ |
| modify_client | 3 | 0 | 4 | +20141.3% ⚠️ |
| delete_quote | 1 | 0 | 1 | +15081.0% ⚠️ |
| cancel_quote | 2 | 0 | 2 | +15081.0% ⚠️ |
| add_note | 1 | 0 | 1 | +15081.0% ⚠️ |
| create_new_client | 142 | 0 | 9 | +862.2% ⚠️ |
| client_search | 192 | 1 | 9 | +611.6% ⚠️ |
| create_new_contact | 111 | 0 | 4 | +447.1% ⚠️ |
| contract_search | 40 | 0 | 1 | +279.5% ⚠️ |
| opportunity_search | 2279 | 15 | 5 | -66.7% ⚠️ |
| create_quote | 1804 | 11 | 4 | -66.3% ⚠️ |
| contact_search | 375 | 2 | 4 | +61.9% ⚠️ |
| clone_opportunity | 267 | 1 | 1 | -43.1% ⚠️ |
| modify_opportunity | 2939 | 19 | 12 | -38.0% ⚠️ |
| view_opportunity_details | 878 | 5 | 5 | -13.5% ⚠️ |
| create_new_opportunity | 3683 | 24 | 21 | -13.4% ⚠️ |

