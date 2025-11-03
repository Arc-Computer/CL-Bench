# Scenario Generation Coverage Report

## Summary

- **Total Scenarios**: 1500
- **Success Scenarios**: 900 (60.0%)
- **Failure Scenarios**: 600 (40.0%)

## Distribution by Task

| Task | Count |
|------|-------|
| create_new_opportunity | 334 |
| modify_opportunity | 155 |
| create_quote | 120 |
| create_new_client | 99 |
| opportunity_search | 89 |
| view_opportunity_details | 87 |
| client_search | 84 |
| modify_quote | 68 |
| modify_client | 57 |
| create_new_contact | 56 |
| quote_details | 41 |
| modify_contact | 40 |
| contact_search | 40 |
| compare_quotes | 36 |
| delete_opportunity | 34 |
| compare_quote_details | 24 |
| quote_search | 18 |
| contract_search | 18 |
| cancel_quote | 17 |
| create_contract | 16 |
| delete_quote | 16 |
| clone_opportunity | 14 |
| summarize_opportunities | 12 |
| add_note | 9 |
| company_search | 8 |
| upload_document | 5 |
| opportunity_details | 3 |

## Distribution by Intent Category

| Intent Category | Count |
|----------------|-------|
| Opportunity Management | 728 |
| Quote Management | 340 |
| Client Management | 240 |
| Contact Management | 136 |
| Contract Management | 34 |
| Notes & Collaboration | 9 |
| Company/Account Management | 8 |
| Document Management | 5 |

## Distribution by Failure Category

| Failure Category | Count |
|-----------------|-------|
| unknown_foreign_key | 136 |
| enum_case_mismatch | 69 |
| missing_required_field | 57 |
| blank_string | 48 |
| type_mismatch | 48 |
| negative_amount | 32 |
| invalid_enum | 31 |
| invalid_date_format | 28 |
| malformed_email | 27 |
| probability_out_of_range | 24 |
| past_date | 17 |
| unknown_field_update | 17 |
| modify_closed_opportunity | 15 |
| probability_decimal | 10 |
| amount_exceeds_max | 9 |
| enum_whitespace | 9 |
| zero_amount | 9 |
| extra_field | 8 |
| duplicate_unique_field | 5 |
| unsafe_filename | 1 |

## Enum/Stage Coverage

### Opportunity Stage Distribution

| Stage | Count |
|-------|-------|
| Prospecting | 183 |
| Proposal | 91 |
| Qualification | 73 |
| Negotiation | 42 |
| Negotiations | 31 |
| prospecting | 18 |
| Prospecting  | 5 |

### Quote Status Distribution

| Status | Count |
|--------|-------|
| Draft | 47 |
| Active | 38 |
| Sent | 31 |
| Approved | 22 |
| draft | 11 |
| active | 10 |

### Client Status Distribution

| Status | Count |
|--------|-------|
| Active | 55 |
| Prospect | 18 |
| active | 16 |
| Inactive | 12 |
|    | 4 |

### Contract Status Distribution

| Status | Count |
|--------|-------|
| Active | 12 |
| active | 6 |
| Pending | 5 |

### Company Type Distribution

| Type | Count |
|------|-------|
| Partner | 2 |
| partner | 1 |

## Frequency Alignment with Source Taxonomy

| Task | CSV Frequency | Expected Count | Generated Count | Deviation |
|------|--------------|----------------|-----------------|------------|
| quote_details | 1 | 0 | 41 | +41394.7% ⚠️ |
| compare_quotes | 1 | 0 | 36 | +36334.4% ⚠️ |
| delete_opportunity | 1 | 0 | 34 | +34310.3% ⚠️ |
| compare_quote_details | 1 | 0 | 24 | +24189.6% ⚠️ |
| modify_quote | 3 | 0 | 68 | +22840.2% ⚠️ |
| modify_contact | 2 | 0 | 40 | +20141.3% ⚠️ |
| modify_client | 3 | 0 | 57 | +19129.3% ⚠️ |
| delete_quote | 1 | 0 | 16 | +16093.1% ⚠️ |
| add_note | 1 | 0 | 9 | +9008.6% ⚠️ |
| cancel_quote | 2 | 0 | 17 | +8502.6% ⚠️ |
| company_search | 8 | 0 | 8 | +912.1% ⚠️ |
| create_new_client | 142 | 14 | 99 | +605.6% ⚠️ |
| quote_search | 32 | 3 | 18 | +469.3% ⚠️ |
| create_new_contact | 111 | 10 | 56 | +410.6% ⚠️ |
| opportunity_details | 6 | 0 | 3 | +406.0% ⚠️ |
| contract_search | 40 | 3 | 18 | +355.4% ⚠️ |
| client_search | 192 | 18 | 84 | +342.8% ⚠️ |
| upload_document | 2190 | 216 | 5 | -97.7% ⚠️ |
| opportunity_search | 2279 | 225 | 89 | -60.5% ⚠️ |
| clone_opportunity | 267 | 26 | 14 | -46.9% ⚠️ |
| modify_opportunity | 2939 | 290 | 155 | -46.6% ⚠️ |
| summarize_opportunities | 194 | 19 | 12 | -37.4% ⚠️ |
| create_quote | 1804 | 178 | 120 | -32.7% ⚠️ |
| create_new_opportunity | 3683 | 363 | 334 | -8.2% ✓ |
| contact_search | 375 | 37 | 40 | +8.0% ✓ |
| view_opportunity_details | 878 | 86 | 87 | +0.3% ✓ |

