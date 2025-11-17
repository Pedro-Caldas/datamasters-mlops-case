# tests/test_monitor_bank.py

from unittest.mock import MagicMock

import pandas as pd


def test_fetch_latest_training_snapshot(monkeypatch):
    """
    Testa se fetch_latest_training_snapshot:
      - chama get_conn()
      - executa SELECT
      - converte saída corretamente para dict
    """

    # ----- Fake row returned by database -----
    fake_row = (
        "RUN123",  # run_id
        "5",  # model_version
        "roc_auc",  # metric_name
        0.91,  # metric_value
        1000,  # n_train
        250,  # n_test
        42,  # n_features
        '{"age": {"mean": 45.0}}',  # feature_stats (JSON string)
    )

    # ----- Fake cursor -----
    fake_cursor = MagicMock()
    fake_cursor.fetchone.return_value = fake_row

    # ----- Fake connection -----
    fake_conn = MagicMock()
    fake_conn.cursor.return_value = fake_cursor

    # ----- Patch get_conn → retorna fake_conn -----
    monkeypatch.setattr("src.monitor_bank.get_conn", lambda: fake_conn)

    import src.monitor_bank as mb

    result = mb.fetch_latest_training_snapshot()

    assert result["run_id"] == "RUN123"
    assert result["model_version"] == "5"
    assert result["metric_name"] == "roc_auc"
    assert result["metric_value"] == 0.91
    assert result["n_train"] == 1000
    assert result["n_test"] == 250
    assert result["n_features"] == 42
    assert result["feature_stats"]["age"]["mean"] == 45.0

    # cursor e conexão fechados
    fake_cursor.close.assert_called_once()
    fake_conn.close.assert_called_once()


def test_fetch_recent_inferences(monkeypatch):
    """
    Testa se fetch_recent_inferences:
      - executa SELECT
      - retorna listas de inputs e preds
    """

    fake_rows = [
        ({"age": 30, "balance": 500}, 0.7),
        ({"age": 45, "balance": 900}, 0.9),
    ]

    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = fake_rows

    fake_conn = MagicMock()
    fake_conn.cursor.return_value = fake_cursor

    monkeypatch.setattr("src.monitor_bank.get_conn", lambda: fake_conn)

    import src.monitor_bank as mb

    inputs, preds = mb.fetch_recent_inferences("RUN123", limit=2)

    assert inputs == [
        {"age": 30, "balance": 500},
        {"age": 45, "balance": 900},
    ]
    assert preds == [0.7, 0.9]

    fake_cursor.close.assert_called_once()
    fake_conn.close.assert_called_once()


def test_compute_simple_stats():
    """
    Testa cálculo de mean/std/count em colunas específicas.
    """

    import src.monitor_bank as mb

    df = pd.DataFrame(
        {
            "age": [30, 40, 50],
            "balance": [100, 200, 300],
        }
    )

    feature_keys = ["age", "balance", "missing_col"]

    result = mb.compute_simple_stats(df, feature_keys)

    # age
    assert abs(result["age"]["mean"] - 40.0) < 1e-6
    assert result["age"]["count"] == 3
    assert result["age"]["std"] > 0

    # balance
    assert abs(result["balance"]["mean"] - 200.0) < 1e-6
    assert result["balance"]["count"] == 3

    # missing_col não aparece
    assert "missing_col" not in result
