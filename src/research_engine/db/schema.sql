CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY,
    question TEXT NOT NULL,
    status VARCHAR(32) NOT NULL,
    error TEXT,
    report_md TEXT,
    report_html TEXT,
    provider VARCHAR(32) NOT NULL DEFAULT '',
    model VARCHAR(128) NOT NULL DEFAULT '',
    support_score DOUBLE PRECISION,
    stats JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS jobs_created_at_idx ON jobs (created_at DESC);

CREATE TABLE IF NOT EXISTS job_events (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    agent VARCHAR(64) NOT NULL,
    kind VARCHAR(64) NOT NULL,
    message TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (job_id, seq)
);

CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    ref_number INTEGER NOT NULL,
    source_type VARCHAR(32) NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL DEFAULT '',
    authors TEXT NOT NULL DEFAULT '',
    year VARCHAR(16) NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    text_origin VARCHAR(16) NOT NULL DEFAULT 'abstract',
    pdf_url TEXT NOT NULL DEFAULT '',
    UNIQUE (job_id, ref_number)
);

CREATE INDEX IF NOT EXISTS sources_job_idx ON sources (job_id);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    source_uid VARCHAR(32) NOT NULL,
    source_ref INTEGER NOT NULL DEFAULT 0,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    section TEXT NOT NULL DEFAULT '',
    page INTEGER
);

CREATE INDEX IF NOT EXISTS chunks_job_idx ON chunks (job_id);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

ALTER TABLE sources ADD COLUMN IF NOT EXISTS text_origin VARCHAR(16) NOT NULL DEFAULT 'abstract';
ALTER TABLE sources ADD COLUMN IF NOT EXISTS pdf_url TEXT NOT NULL DEFAULT '';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section TEXT NOT NULL DEFAULT '';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page INTEGER;
