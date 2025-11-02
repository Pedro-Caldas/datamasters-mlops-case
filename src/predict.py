import json
import os

import mlflow


def load_model_uri():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))
    model_name = os.getenv("MODEL_NAME", "datamasters-elasticnet")
    # Aqui pode ser "Staging" ou "Production", caso alguém queira fazer um teste A/B ou algo assim
    stage = os.getenv("PREDICT_STAGE", "Production")
    return f"models:/{model_name}/{stage}"


def main():
    from sklearn.datasets import load_diabetes

    X, _ = load_diabetes(return_X_y=True)
    X = X[:5]  # batch exemplo

    model_uri = load_model_uri()
    # Pega a versão vigente do stage definido ao carregar o modelo
    model = mlflow.pyfunc.load_model(model_uri)
    preds = model.predict(X)

    print(json.dumps({"model_uri": model_uri, "preds": preds.tolist()}, indent=2))


if __name__ == "__main__":
    main()
