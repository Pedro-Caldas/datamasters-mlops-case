# tests/test_predict_bank.py

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

import src.predict_bank as pb


def test_load_input_reads_csv_and_samples_20(monkeypatch):
    """
    Garante que load_input:
      - lê o CSV correto
      - retorna exatamente 20 linhas
    """
    fake_df = pd.DataFrame({"a": range(100)})
    monkeypatch.setattr(pd, "read_csv", lambda path: fake_df)

    result = pb.load_input()
    assert len(result) == 20


def test_load_production_model(monkeypatch):
    """
    Garante que load_production_model:
      - chama MlflowClient
      - pega última versão Production
      - carrega o modelo com mlflow.sklearn.load_model
    """

    # Fake response do MlflowClient
    fake_version = MagicMock()
    fake_version.run_id = "RUN123"

    fake_client = MagicMock()
    fake_client.get_latest_versions.return_value = [fake_version]

    # Mock do MlflowClient() → retorna fake_client
    monkeypatch.setattr(pb, "MlflowClient", lambda: fake_client)

    # Mock do loader do modelo
    fake_model = MagicMock()
    monkeypatch.setattr(pb.mlflow.sklearn, "load_model", lambda uri: fake_model)

    model, run_id, model_uri = pb.load_production_model("bank-model")

    assert model is fake_model
    assert run_id == "RUN123"
    assert model_uri == "runs:/RUN123/model"


def test_main_runs_full_flow(monkeypatch, capsys):
    """
    Testa o fluxo completo do main:
      - load_input → df fake
      - load_production_model → modelo fake
      - predict_proba é chamado
      - save_inference_row é chamado N vezes
      - imprime JSON final
    """

    # ---- Mock load_input ----
    df_fake = pd.DataFrame({"age": [10, 20], "balance": [100, 200]})
    monkeypatch.setattr(pb, "load_input", lambda: df_fake)

    # ---- Mock modelo ----
    class DummyModel:
        def predict_proba(self, X):
            return np.array([[0.2, 0.8]] * len(X))

    dummy_model = DummyModel()

    # ---- Mock load_production_model ----
    monkeypatch.setattr(
        pb,
        "load_production_model",
        lambda model_name: (dummy_model, "RUN999", "runs:/RUN999/model"),
    )

    # ---- Mock save_inference_row ----
    mock_save = MagicMock()
    monkeypatch.setattr(pb, "save_inference_row", mock_save)

    # ---- Executa main ----
    pb.main()

    # save_inference_row deve ser chamado uma vez por linha
    assert mock_save.call_count == len(df_fake)

    # ---- Verifica impressão JSON ----
    captured = capsys.readouterr().out
    output = json.loads(captured)

    assert "predictions" in output
    assert "input_shape" in output
    assert output["input_shape"] == list(df_fake.shape)
    assert output["model_uri"] == "runs:/RUN999/model"
