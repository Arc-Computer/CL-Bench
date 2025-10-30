-- Schema definition for the CRM Postgres sandbox.
-- Mirrors data/fake_crm_tables_schema.json with relational structure and constraints.

-- Enable UUID generation helper.
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'client_status') THEN
        CREATE TYPE client_status AS ENUM ('Active', 'Prospect', 'Inactive');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'opportunity_stage') THEN
        CREATE TYPE opportunity_stage AS ENUM (
            'Prospecting',
            'Qualification',
            'Proposal',
            'Negotiation',
            'Closed-Won',
            'Closed-Lost'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'quote_status') THEN
        CREATE TYPE quote_status AS ENUM ('Draft', 'Sent', 'Approved', 'Rejected', 'Canceled');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'contract_status') THEN
        CREATE TYPE contract_status AS ENUM ('Active', 'Pending', 'Expired');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'document_entity_type') THEN
        CREATE TYPE document_entity_type AS ENUM ('Opportunity', 'Contract', 'Quote', 'Client');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'note_entity_type') THEN
        CREATE TYPE note_entity_type AS ENUM ('Opportunity', 'Client', 'Contact', 'Quote', 'Contract');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'company_type') THEN
        CREATE TYPE company_type AS ENUM ('Partner', 'Vendor', 'Competitor');
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS clients (
    client_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    industry TEXT,
    email TEXT UNIQUE,
    phone TEXT,
    address TEXT,
    status client_status NOT NULL,
    created_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    owner TEXT,
    CHECK (name <> '')
);

CREATE TABLE IF NOT EXISTS contacts (
    contact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    title TEXT,
    email TEXT,
    phone TEXT,
    client_id UUID NOT NULL REFERENCES clients(client_id) ON DELETE CASCADE,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS opportunities (
    opportunity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL REFERENCES clients(client_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    stage opportunity_stage NOT NULL,
    amount NUMERIC(14, 2) NOT NULL CHECK (amount > 0),
    close_date DATE,
    owner TEXT,
    probability INTEGER,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (probability IS NULL OR (probability BETWEEN 1 AND 99))
);

CREATE INDEX IF NOT EXISTS opportunities_client_id_idx ON opportunities(client_id);

CREATE TABLE IF NOT EXISTS quotes (
    quote_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    opportunity_id UUID NOT NULL REFERENCES opportunities(opportunity_id) ON DELETE CASCADE,
    version TEXT,
    amount NUMERIC(14, 2) NOT NULL CHECK (amount > 0),
    status quote_status NOT NULL,
    valid_until DATE,
    created_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    quote_prefix TEXT
);

CREATE INDEX IF NOT EXISTS quotes_opportunity_id_idx ON quotes(opportunity_id);

CREATE TABLE IF NOT EXISTS contracts (
    contract_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL REFERENCES clients(client_id) ON DELETE CASCADE,
    opportunity_id UUID REFERENCES opportunities(opportunity_id) ON DELETE SET NULL,
    start_date DATE,
    end_date DATE,
    value NUMERIC(14, 2),
    status contract_status NOT NULL,
    document_url TEXT
);

CREATE TABLE IF NOT EXISTS companies (
    company_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    type company_type NOT NULL,
    industry TEXT,
    address TEXT,
    contacts TEXT[]
);

CREATE TABLE IF NOT EXISTS documents (
    document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type document_entity_type NOT NULL,
    entity_id UUID NOT NULL,
    file_name TEXT NOT NULL,
    uploaded_by TEXT,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    file_url TEXT
);

CREATE INDEX IF NOT EXISTS documents_entity_idx ON documents(entity_type, entity_id);

CREATE TABLE IF NOT EXISTS notes (
    note_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type note_entity_type NOT NULL,
    entity_id UUID NOT NULL,
    content TEXT NOT NULL,
    created_by TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS notes_entity_idx ON notes(entity_type, entity_id);
