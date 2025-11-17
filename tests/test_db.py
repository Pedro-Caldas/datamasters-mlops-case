# tests/test_db.py
import json
from unittest.mock import MagicMock, patch

import src.db as db


def test_get_conn_uses_env_vars(monkeypatch):
    """
    Garante que get_conn usa as variáveis de ambiente esperadas
    para montar a conexão.
    """
    # Configura envs "fake"
    monkeypatch.setenv("POSTGRES_HOST", "fake-host")
    monkeypatch.setenv("POSTGRES_PORT", "9999")
    monkeypatch.setenv("POSTGRES_USER", "fake-user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "fake-pwd")
    monkeypatch.setenv("POSTGRES_DB", "fake-db")
    monkeypatch.delenv("RUNNING_IN_DOCKER", raising=False)

    fake_connect = MagicMock()

    with patch("src.db.psycopg2.connect", fake_connect):
        db.get_conn()

    fake_connect.assert_called_once_with(
        host="fake-host",
        port="9999",
        user="fake-user",
        password="fake-pwd",
        dbname="fake-db",
    )


def test_save_inference_row_inserts_and_closes(monkeypatch):
    """
    Garante que save_inference_row:
      - chama INSERT na tabela inference_logs
      - commita e fecha cursor/conn
    """
    fake_conn = MagicMock()
    fake_cursor = MagicMock()
    fake_conn.cursor.return_value = fake_cursor

    # get_conn vai devolver nossa conexão fake
    with patch("src.db.get_conn", return_value=fake_conn):
        db.save_inference_row(
            run_id="run-123",
            model_version="1",
            features={"age": 40, "balance": 1000},
            prediction=0.73,
        )

    # Verifica se execute foi chamado
    fake_cursor.execute.assert_called_once()
    sql, params = fake_cursor.execute.call_args[0]

    assert "INSERT INTO inference_logs" in sql

    # Params: (run_id, model_version, features_json, prediction)
    assert params[0] == "run-123"
    assert params[1] == "1"
    # o terceiro é JSON string
    assert json.loads(params[2]) == {"age": 40, "balance": 1000}
    assert params[3] == 0.73

    # Verifica commit e fechamento
    fake_conn.commit.assert_called_once()
    fake_cursor.close.assert_called_once()
    fake_conn.close.assert_called_once()
