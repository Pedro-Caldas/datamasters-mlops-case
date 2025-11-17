# tests/test_train_bank_marketing.py
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import src.train_bank_marketing as tbm


def test_compute_metric_f1_and_roc_auc():
    """
    Testa se compute_metric calcula corretamente F1 (com threshold 0.5)
    e ROC AUC em um caso simples.
    """
    y_true = [0, 1, 1, 0]
    y_scores = [0.1, 0.9, 0.8, 0.2]

    # Converte para numpy array para compatibilidade com o código de produção
    y_scores_np = np.array(y_scores)

    f1 = tbm.compute_metric(y_true, y_scores_np, "f1")
    roc = tbm.compute_metric(y_true, y_scores_np, "roc_auc")

    assert f1 == 1.0
    assert roc == 1.0


def test_train_and_log_uses_mlflow(monkeypatch):
    """
    Testa se train_and_log:
      - chama fit e predict_proba do modelo
      - chama mlflow.log_params, mlflow.log_metric
      - chama mlflow.sklearn.log_model
      - retorna dict com run_id vindo do mlflow.active_run()
    """

    X_train = pd.DataFrame({"f1": [0, 1, 2], "f2": [3, 4, 5]})
    y_train = [0, 1, 0]
    X_test = pd.DataFrame({"f1": [10, 11], "f2": [12, 13]})
    y_test = [0, 1]

    class DummyModel:
        def __init__(self):
            self._params = {"C": 1.0}

        def fit(self, X, y):
            self._fitted = True

        def predict_proba(self, X):
            # Retorna numpy array para suportar [:,1] no código de produção
            return np.array([[0.4, 0.6]] * len(X))

        def get_params(self, deep=True):
            return self._params

    dummy_model = DummyModel()

    mock_run_ctx = MagicMock()
    mock_run_ctx.__enter__.return_value = MagicMock()
    mock_run_ctx.__exit__.return_value = False

    monkeypatch.setattr(tbm.mlflow, "start_run", lambda run_name=None: mock_run_ctx)

    active_run_mock = MagicMock()
    active_run_mock.info.run_id = "fake-run-id"
    monkeypatch.setattr(tbm.mlflow, "active_run", lambda: active_run_mock)

    mock_log_params = MagicMock()
    mock_log_metric = MagicMock()
    mock_log_model = MagicMock()

    monkeypatch.setattr(tbm.mlflow, "log_params", mock_log_params)
    monkeypatch.setattr(tbm.mlflow, "log_metric", mock_log_metric)
    monkeypatch.setattr(tbm.mlflow.sklearn, "log_model", mock_log_model)

    mock_signature = MagicMock()
    monkeypatch.setattr(tbm, "infer_signature", MagicMock(return_value=mock_signature))

    result = tbm.train_and_log(
        model_name="dummy_model",
        model=dummy_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        metric_name="roc_auc",
    )

    mock_log_params.assert_called_once()
    mock_log_metric.assert_called_once()
    mock_log_model.assert_called_once()

    assert result["model_name"] == "dummy_model"
    assert result["run_id"] == "fake-run-id"
    assert isinstance(result["metric"], float)


def test_log_training_metadata_to_db_inserts_and_closes(monkeypatch):
    """
    Garante que log_training_metadata_to_db:
      - abre conexão com psycopg2.connect
      - executa INSERT na tabela training_data
      - commita e fecha cursor/conn
      - salva feature_stats como JSON
    """

    X_train = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
    X_test = pd.DataFrame({"f1": [7, 8], "f2": [9, 10]})
    y_train = [0, 1, 0]
    y_test = [1, 0]

    fake_conn = MagicMock()
    fake_cursor = MagicMock()
    fake_conn.cursor.return_value = fake_cursor

    with patch("src.train_bank_marketing.psycopg2.connect", return_value=fake_conn):
        tbm.log_training_metadata_to_db(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            run_id="run-xyz",
            model_version=2,
            metric_name="roc_auc",
            metric_value=0.88,
        )

    fake_conn.cursor.assert_called_once()
    fake_cursor.execute.assert_called_once()
    fake_conn.commit.assert_called_once()
    fake_cursor.close.assert_called_once()
    fake_conn.close.assert_called_once()

    sql, params = fake_cursor.execute.call_args[0]
    assert "INSERT INTO training_data" in sql

    assert params[0] == "run-xyz"
    assert params[1] == "2"
    assert params[2] == "roc_auc"
    assert params[3] == 0.88
    assert params[4] == len(X_train)
    assert params[5] == len(X_test)
    assert params[6] == X_train.shape[1]

    feature_stats = json.loads(params[7])
    assert "f1" in feature_stats
    assert "f2" in feature_stats
