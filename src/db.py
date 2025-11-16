import json
import os

import psycopg2


def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
    )


def save_training_row(run_id, model_version, features: dict, target: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO training_data (run_id, model_version, features, target)
        VALUES (%s, %s, %s::jsonb, %s)
        """,
        (run_id, model_version, json.dumps(features), target),
    )
    conn.commit()
    cur.close()
    conn.close()


def save_inference_row(run_id, model_version, features: dict, prediction: float):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO inference_logs (run_id, model_version, input, prediction)
        VALUES (%s, %s, %s::jsonb, %s)
        """,
        (run_id, model_version, json.dumps(features), prediction),
    )
    conn.commit()
    cur.close()
    conn.close()
