import json
import os

import psycopg2


def get_conn():
    """
    Conecta tanto localmente quanto dentro do Docker.
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    if host == "localhost" and os.getenv("RUNNING_IN_DOCKER") == "1":
        host = "postgres"

    return psycopg2.connect(
        host=host,
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
        VALUES (%s, %s, %s::jsonb, %s::jsonb)
        """,
        (run_id, model_version, json.dumps(features), json.dumps(target)),
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
