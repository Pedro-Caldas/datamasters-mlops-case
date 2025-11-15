import json
import os

import mlflow


def load_model_uri():
    """
    Lê variáveis de ambiente, define tracking URI e monta o caminho
    do modelo no Model Registry, usando o stage solicitado.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))

    model_name = os.getenv("MODEL_NAME", "datamasters-model")
    stage = os.getenv("STAGE", "Production")

    return f"models:/{model_name}/{stage}"


def main():
    """
    Exemplo simples de predição. Atualmente usa um batch fixo,
    mas será atualizado no Bloco 2 com o dataset Bank Marketing.
    """
    from sklearn.datasets import load_diabetes

    X, _ = load_diabetes(return_X_y=True)
    X = X[:5]  # batch de exemplo

    model_uri = load_model_uri()
    model = mlflow.pyfunc.load_model(model_uri)
    preds = model.predict(X)

    print(json.dumps({"model_uri": model_uri, "preds": preds.tolist()}, indent=2))


if __name__ == "__main__":
    main()
