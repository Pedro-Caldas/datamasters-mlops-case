-- Tabelas para persistir tanto os dados de treino, quanto os de inferência
CREATE TABLE IF NOT EXISTS training_data (
    id SERIAL PRIMARY KEY,
    run_id TEXT,
    model_version TEXT,
    features JSONB,
    target INTEGER,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS inference_logs (
    id SERIAL PRIMARY KEY,
    run_id TEXT,
    model_version TEXT,
    input JSONB,
    prediction REAL,
    timestamp TIMESTAMP DEFAULT NOW()
);
