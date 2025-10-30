-- Seed data for CRM Postgres sandbox.
-- Inserts anonymized records aligned with golden-case happy paths.

-- Clients ------------------------------------------------------------------
INSERT INTO clients (client_id, name, industry, email, phone, address, status, created_date, owner)
VALUES
    ('8b52df1c-fd28-4aef-98a7-4cdb7bb2d1a4', 'Acme Analytics', 'Software', 'ops@acmeanalytics.example', '415-555-0100', '500 Market St, San Francisco, CA', 'Active', '2025-01-12T14:00:00Z', 'luca.carra'),
    ('7b80f463-ccfb-4fff-8e75-932ce8c6f0e4', 'Borealis Labs', 'Biotech', 'hello@borealis.example', NULL, '220 Innovation Way, Boston, MA', 'Prospect', '2025-02-05T16:30:00Z', 'stephanie.wong'),
    ('f3304d3a-a0a3-4e79-baca-86c6a5d819a0', 'Delta Fabrication', 'Manufacturing', 'support@deltafab.example', '312-555-0135', '880 Industrial Blvd, Chicago, IL', 'Active', '2025-03-01T18:45:00Z', 'aman.sharma'),
    ('59b4adc6-2e18-4fc7-869a-6e2155f8d9c2', 'Ember Retail', 'Retail', 'contact@emberretail.example', '646-555-0109', '77 Grove Ave, New York, NY', 'Active', '2025-02-18T12:10:00Z', 'stephanie.wong'),
    ('6e1f3c79-4e65-4649-9a2c-c73d9d9b2a59', 'Fresco Foods', 'Food & Beverage', 'team@frescofoods.example', '503-555-0177', '102 Harvest Rd, Portland, OR', 'Active', '2025-02-25T09:20:00Z', 'gaby.chan'),
    ('0b1b94d8-504e-4f10-954f-8ece66d3bb90', 'Glider Mobility', 'Mobility', 'mobility@glider.example', '206-555-0149', '410 Pioneer Sq, Seattle, WA', 'Prospect', '2025-04-02T11:05:00Z', 'aman.sharma'),
    ('c5f2d1bb-3b7d-4bdf-9350-bb4fe79a7211', 'Harbor Consulting', 'Consulting', 'info@harborconsulting.example', '212-555-0184', '18 Harbor Row, New York, NY', 'Active', '2025-01-28T08:45:00Z', 'aman.sharma'),
    ('af67e410-dc92-4cf7-88cb-6ea82f91ac39', 'Indigo Health', 'Healthcare', 'partnerships@indigohealth.example', NULL, '940 Wellness Dr, Austin, TX', 'Prospect', '2025-03-17T15:55:00Z', 'federico.minutoli'),
    ('68fc16b0-70b4-45c5-9bb9-154d5d9ca7a2', 'Jetstream Logistics', 'Transportation', 'ops@jetstreamlogistics.example', '404-555-0168', '44 Runway Rd, Atlanta, GA', 'Active', '2025-01-05T10:25:00Z', 'luca.carra'),
    ('949779f5-68de-4a67-9dad-536f4f1c1ff1', 'Keystone Supplies', 'Industrial', 'sales@keystonesupplies.example', '646-555-0199', '300 Ridge St, Newark, NJ', 'Inactive', '2025-02-12T13:40:00Z', 'gaby.chan')
ON CONFLICT (client_id) DO NOTHING;


-- Contacts -----------------------------------------------------------------
INSERT INTO contacts (contact_id, first_name, last_name, title, email, phone, client_id, notes)
VALUES
    ('047a2f66-95a3-4d5c-8b43-8858a0a19d3c', 'Jenna', 'Fields', 'Director of Operations', 'jenna.fields@acmeanalytics.example', '415-555-0191', '8b52df1c-fd28-4aef-98a7-4cdb7bb2d1a4', 'Primary operations contact'),
    ('88f7e3ac-1843-4538-8231-b3bebfa3214e', 'Rahul', 'Menon', 'VP Research', 'rahul.menon@borealis.example', NULL, '7b80f463-ccfb-4fff-8e75-932ce8c6f0e4', 'Borealis pilot sponsor'),
    ('6d0e7f24-d659-494c-8001-177d101a66c1', 'Celia', 'Vargas', 'Plant Manager', 'celia.vargas@deltafab.example', '312-555-0192', 'f3304d3a-a0a3-4e79-baca-86c6a5d819a0', 'Prefers morning calls'),
    ('d394a57b-11ba-4dc4-94cd-a71be7583d41', 'Marcus', 'Lee', 'VP Retail Innovation', 'marcus.lee@emberretail.example', '646-555-0110', '59b4adc6-2e18-4fc7-869a-6e2155f8d9c2', NULL),
    ('c2d1d2eb-7d7c-4e7e-b6ae-47d9d5a5a51e', 'Tanya', 'Rios', 'Procurement Lead', 'tanya.rios@frescofoods.example', '503-555-0178', '6e1f3c79-4e65-4649-9a2c-c73d9d9b2a59', 'Handles renewals'),
    ('fbf72f3a-771a-4b28-9af8-cd7418d4ab72', 'Oliver', 'Park', 'CTO', 'oliver.park@glider.example', '206-555-0150', '0b1b94d8-504e-4f10-954f-8ece66d3bb90', 'Interested in pilot metrics'),
    ('a9e3de3e-623d-49e8-bd6f-964025fe1910', 'Elaine', 'Harper', 'COO', 'elaine.harper@harborconsulting.example', '212-555-0185', 'c5f2d1bb-3b7d-4bdf-9350-bb4fe79a7211', NULL),
    ('c53afdd8-f6e3-4465-b86c-7a6fd7b6b9cd', 'Noah', 'Singh', 'Head of Partnerships', 'noah.singh@indigohealth.example', NULL, 'af67e410-dc92-4cf7-88cb-6ea82f91ac39', 'Requested security questionnaire'),
    ('4b06741e-7dce-4ee3-884d-ff0dd4c3e8f9', 'Lauren', 'Chen', 'Logistics Director', 'lauren.chen@jetstreamlogistics.example', '404-555-0169', '68fc16b0-70b4-45c5-9bb9-154d5d9ca7a2', NULL),
    ('261116db-7456-4f49-8d75-96996f2bcbe8', 'Peter', 'Hale', 'Operations Manager', 'peter.hale@keystonesupplies.example', '646-555-0200', '949779f5-68de-4a67-9dad-536f4f1c1ff1', 'Re-engagement target')
ON CONFLICT (contact_id) DO NOTHING;


-- Opportunities -------------------------------------------------------------
INSERT INTO opportunities (
    opportunity_id,
    client_id,
    name,
    stage,
    amount,
    close_date,
    owner,
    probability,
    notes,
    created_at
)
VALUES
    ('1d111992-7c0d-4bbe-a68a-d36bf5be8c3e', '8b52df1c-fd28-4aef-98a7-4cdb7bb2d1a4', 'Acme Renewal', 'Prospecting', 85000, NULL, 'luca.carra', 25, 'Renewal cycle kickoff', '2025-04-10T15:00:00Z'),
    ('2a0ceecc-97bc-4dcd-978b-4484a1c2b74d', '7b80f463-ccfb-4fff-8e75-932ce8c6f0e4', 'Borealis Pilot', 'Qualification', 120000, NULL, 'stephanie.wong', 30, 'Awaiting lab validation', '2025-04-15T16:30:00Z'),
    ('3e672caa-5711-4bf6-8a3c-3776b1c8a3c4', 'f3304d3a-a0a3-4e79-baca-86c6a5d819a0', 'Delta Fabrication Upgrade', 'Proposal', 210000, NULL, 'aman.sharma', 45, 'Proposal shared 2025-04-01', '2025-04-18T11:00:00Z'),
    ('4f8320c4-1e82-4fa9-84cb-f5f9b2585fa2', '59b4adc6-2e18-4fc7-869a-6e2155f8d9c2', 'Ember Retail Expansion', 'Negotiation', 330000, '2025-11-30', 'stephanie.wong', 60, 'Negotiating payment schedule', '2025-04-22T09:10:00Z'),
    ('5a657971-7bb7-4f4c-8b37-1b8391f5dd5a', '6e1f3c79-4e65-4649-9a2c-c73d9d9b2a59', 'Fresco Foods Renewal', 'Closed-Won', 95000, '2025-05-15', 'gaby.chan', 95, 'Signed master renewal', '2025-03-30T13:45:00Z'),
    ('661c1f32-6efc-4cfa-a4bd-ff1c983b2984', '0b1b94d8-504e-4f10-954f-8ece66d3bb90', 'Glider Mobility Pilot', 'Closed-Lost', 140000, '2025-03-20', 'aman.sharma', 5, 'Lost to Hawk Systems', '2025-03-25T14:25:00Z'),
    ('7f5cb47a-5829-4bb2-8b67-6e94c386327a', 'c5f2d1bb-3b7d-4bdf-9350-bb4fe79a7211', 'Harbor Consulting Expansion', 'Negotiation', 175000, NULL, 'aman.sharma', 55, 'Reviewing security addendum', '2025-04-05T10:15:00Z'),
    ('891bb0bd-0b5a-4f5a-b48c-58035c50a648', 'af67e410-dc92-4cf7-88cb-6ea82f91ac39', 'Indigo Health Pilot', 'Qualification', 210000, NULL, 'federico.minutoli', 40, 'Security questionnaire in progress', '2025-04-08T17:20:00Z'),
    ('9a27aeb3-0b99-4e65-b167-df2ebf082e9c', '68fc16b0-70b4-45c5-9bb9-154d5d9ca7a2', 'Jetstream Logistics Modernization', 'Prospecting', 420000, '2025-12-15', 'luca.carra', 20, 'Discovery workshop scheduled', '2025-04-12T09:50:00Z'),
    ('a3fbde5a-c7bc-4923-93b7-0970ed2236c3', '949779f5-68de-4a67-9dad-536f4f1c1ff1', 'Keystone Supplies Expansion', 'Negotiation', 260000, NULL, 'gaby.chan', 50, 'Awaiting pricing approval', '2025-04-14T12:05:00Z')
ON CONFLICT (opportunity_id) DO NOTHING;


-- Quotes -------------------------------------------------------------------
INSERT INTO quotes (quote_id, opportunity_id, version, amount, status, valid_until, created_date, quote_prefix)
VALUES
    ('11f2d102-1076-44c9-8f25-f0eaa90bf34e', '1d111992-7c0d-4bbe-a68a-d36bf5be8c3e', 'v1', 85000, 'Draft', NULL, '2025-04-12T18:05:00Z', 'ACME-REN'),
    ('22c6b69e-465f-4f70-b248-31553378ea98', '2a0ceecc-97bc-4dcd-978b-4484a1c2b74d', NULL, 120000, 'Sent', NULL, '2025-04-16T19:20:00Z', 'BP-2025'),
    ('33791a1b-9bc4-4645-8e35-bbfa6e7e10bf', '3e672caa-5711-4bf6-8a3c-3776b1c8a3c4', NULL, 210000, 'Approved', '2025-11-30', '2025-04-19T10:00:00Z', NULL),
    ('449e72f3-7b75-4c6a-9a10-885770126923', '4f8320c4-1e82-4fa9-84cb-f5f9b2585fa2', NULL, 330000, 'Rejected', NULL, '2025-04-23T08:40:00Z', NULL),
    ('55c2004f-1f06-44f5-9800-85080f9cb40f', '5a657971-7bb7-4f4c-8b37-1b8391f5dd5a', NULL, 95000, 'Canceled', NULL, '2025-03-31T14:10:00Z', NULL),
    ('66a31f3d-aa87-47cd-801b-552b0da41d69', '661c1f32-6efc-4cfa-a4bd-ff1c983b2984', 'v2', 140000, 'Draft', NULL, '2025-03-26T11:15:00Z', NULL),
    ('77b81be5-4ad1-4524-aa69-c9c7cdbaff47', '7f5cb47a-5829-4bb2-8b67-6e94c386327a', '1.1', 175000, 'Sent', NULL, '2025-04-06T10:20:00Z', 'HC-EXP'),
    ('889bb0b4-3b21-4d18-9af1-68451f3d4de4', '891bb0bd-0b5a-4f5a-b48c-58035c50a648', NULL, 210000, 'Approved', '2025-12-31', '2025-04-09T17:55:00Z', NULL),
    ('99cc3df1-7c7c-4ad9-bdd7-a0eade43b437', '9a27aeb3-0b99-4e65-b167-df2ebf082e9c', NULL, 420000, 'Draft', NULL, '2025-04-13T09:55:00Z', 'JET-2025'),
    ('aa0d5172-7b48-4c5a-b5bb-9989bb3b0277', 'a3fbde5a-c7bc-4923-93b7-0970ed2236c3', 'V3', 260000, 'Approved', NULL, '2025-04-15T12:30:00Z', NULL)
ON CONFLICT (quote_id) DO NOTHING;


-- Contracts ----------------------------------------------------------------
INSERT INTO contracts (contract_id, client_id, opportunity_id, start_date, end_date, value, status, document_url)
VALUES
    ('16fd2d88-32e2-44f5-bc89-a0748e06be02', '59b4adc6-2e18-4fc7-869a-6e2155f8d9c2', '4f8320c4-1e82-4fa9-84cb-f5f9b2585fa2', '2025-05-01', '2026-04-30', 330000, 'Active', 'https://docs.example.com/contracts/ember-expansion'),
    ('2f51a0fe-7ccb-4ad3-8795-80d43299cb3a', '6e1f3c79-4e65-4649-9a2c-c73d9d9b2a59', '5a657971-7bb7-4f4c-8b37-1b8391f5dd5a', '2025-04-01', '2026-03-31', 95000, 'Active', 'https://docs.example.com/contracts/fresco-renewal'),
    ('3a67f59f-7cd5-493f-8b19-3370c0ac553a', '8b52df1c-fd28-4aef-98a7-4cdb7bb2d1a4', NULL, '2024-11-01', '2025-10-31', 120000, 'Expired', 'https://docs.example.com/contracts/acme-2024'),
    ('4b1125f1-0a3d-40fb-8c89-7d94d8fbe94b', '0b1b94d8-504e-4f10-954f-8ece66d3bb90', '661c1f32-6efc-4cfa-a4bd-ff1c983b2984', '2024-07-01', '2025-06-30', 140000, 'Expired', 'https://docs.example.com/contracts/glider-pilot')
ON CONFLICT (contract_id) DO NOTHING;


-- Companies ----------------------------------------------------------------
INSERT INTO companies (company_id, name, type, industry, address, contacts)
VALUES
    ('8d2e2f17-e772-4d37-8a20-0ce543917bb6', 'Atlas Consulting Partners', 'Partner', 'Consulting', '101 Atlas Way, Denver, CO', ARRAY['047a2f66-95a3-4d5c-8b43-8858a0a19d3c']),
    ('c6fb5657-f44f-4bf6-b5a6-94ffd8cf5da2', 'NovaEdge Systems', 'Competitor', 'Software', '75 Edge Plaza, Phoenix, AZ', ARRAY['fbf72f3a-771a-4b28-9af8-cd7418d4ab72']),
    ('d8abbd94-fc01-4e37-9f71-45b7607388b6', 'Hawk Systems Group', 'Competitor', 'Mobility', '60 Hawk Ridge Rd, Seattle, WA', ARRAY['fbf72f3a-771a-4b28-9af8-cd7418d4ab72'])
ON CONFLICT (company_id) DO NOTHING;


-- Documents ----------------------------------------------------------------
INSERT INTO documents (document_id, entity_type, entity_id, file_name, uploaded_by, uploaded_at, file_url)
VALUES
    ('d1b5f3c3-34b0-45ae-81d9-e6c0a8f7996a', 'Client', '8b52df1c-fd28-4aef-98a7-4cdb7bb2d1a4', 'acme-onboarding.pdf', 'luca.carra', '2025-04-11T10:05:00Z', 'https://docs.example.com/acme/acme-onboarding.pdf'),
    ('a4a279fe-d6fa-4c75-a93b-5c3c0a2e5e56', 'Opportunity', '2a0ceecc-97bc-4dcd-978b-4484a1c2b74d', 'borealis-roi.xlsx', 'stephanie.wong', '2025-04-16T12:45:00Z', 'https://docs.example.com/borealis/roi.xlsx'),
    ('b7e3218b-f595-40dd-85de-f6e21493bd69', 'Quote', '33791a1b-9bc4-4645-8e35-bbfa6e7e10bf', 'delta-upgrade-quote.pdf', 'aman.sharma', '2025-04-19T10:05:00Z', 'https://docs.example.com/delta/upgrade-quote.pdf'),
    ('c3f51242-1862-4b01-924e-2e8fdc1cff7f', 'Contract', '16fd2d88-32e2-44f5-bc89-a0748e06be02', 'ember-expansion-contract.pdf', 'stephanie.wong', '2025-04-23T09:05:00Z', 'https://docs.example.com/ember/expansion-contract.pdf')
ON CONFLICT (document_id) DO NOTHING;


-- Notes --------------------------------------------------------------------
INSERT INTO notes (note_id, entity_type, entity_id, content, created_by, created_at)
VALUES
    ('5f0d87dc-8bb5-42bd-bc10-41a2f4f2a40c', 'Opportunity', '1d111992-7c0d-4bbe-a68a-d36bf5be8c3e', 'Customer wants to review roadmap before next call.', 'luca.carra', '2025-04-11T09:00:00Z'),
    ('7a9d4090-0b9c-4cfa-b630-3bfb73bc01af', 'Opportunity', '4f8320c4-1e82-4fa9-84cb-f5f9b2585fa2', 'Legal review scheduled for May 5.', 'stephanie.wong', '2025-04-22T10:30:00Z'),
    ('81d1ad9d-0c0f-47d0-8f60-75a08b0be8da', 'Client', '6e1f3c79-4e65-4649-9a2c-c73d9d9b2a59', 'Procurement approved updated SLA.', 'gaby.chan', '2025-04-01T14:15:00Z'),
    ('9fc11e24-e46c-48d5-8b2f-0568ad139abd', 'Quote', '99cc3df1-7c7c-4ad9-bdd7-a0eade43b437', 'Ready for executive signature.', 'luca.carra', '2025-04-13T10:05:00Z'),
    ('a72164c3-e8a8-403f-ba8c-60a778559975', 'Contract', '2f51a0fe-7ccb-4ad3-8795-80d43299cb3a', 'Renewal auto-renews annually unless canceled 60 days prior.', 'gaby.chan', '2025-04-02T09:30:00Z')
ON CONFLICT (note_id) DO NOTHING;
