CREATE TABLE IF NOT EXISTS training_data (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMP DEFAULT NOW(),
    source VARCHAR(50),
    rows INT,
    cols INT
);

CREATE TABLE IF NOT EXISTS inference_logs (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMP DEFAULT NOW(),
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    input_shape VARCHAR(50),
    prediction JSONB
);
