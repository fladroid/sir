-- SIR Projekat — PostgreSQL šema za Skill Persistence
-- Flavio & Claude | Mart 2026

-- Apstrahovane procedure koje sistem "pamti"
CREATE TABLE sir_procedures (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    lang        VARCHAR(3),           -- NULL = važi za sve jezike
    direction   VARCHAR(10),          -- eng2local / local2eng / NULL
    description TEXT,
    steps       JSONB NOT NULL,
    version     INTEGER DEFAULT 1,
    active      BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMP DEFAULT NOW(),
    updated_at  TIMESTAMP DEFAULT NOW()
);

-- Svaki pokušaj rešavanja zadatka
CREATE TABLE sir_trajectories (
    id               SERIAL PRIMARY KEY,
    procedure_id     INTEGER REFERENCES sir_procedures(id),
    lang             VARCHAR(3),
    direction        VARCHAR(10),
    original_text    TEXT,
    iterations       INTEGER,
    final_similarity FLOAT,
    delta_similarity FLOAT,           -- poboljšanje vs. baseline
    steps_log        JSONB,           -- detaljan log svakog koraka
    success          BOOLEAN,
    created_at       TIMESTAMP DEFAULT NOW()
);

-- Metapodaci o procesu — "explain plan" za SIR sistem
CREATE TABLE sir_metadata (
    id          SERIAL PRIMARY KEY,
    session_id  VARCHAR(50),
    lang        VARCHAR(3),
    metric      VARCHAR(50),          -- npr. 'avg_delta_similarity'
    value       FLOAT,
    details     JSONB,                -- distribucije, mape slabih konstrukcija
    created_at  TIMESTAMP DEFAULT NOW()
);
