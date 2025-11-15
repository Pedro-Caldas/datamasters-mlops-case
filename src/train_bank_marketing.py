import json
import os
import pathlib

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

# Usar servidor local do MLflow se tiver, senão fallback (caso do CI)

if os.getenv("CI"):
    # CI do GitHub usa tracking local
    mlflow.set_tracking_uri("file:./mlruns-ci")
else:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri("http://localhost:5050")

mlflow.set_experiment("bank-marketing")

# Carregar dados processados no script de data_bank_marketing

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def load_data():
    X_train = pd.read_csv(PROCESSED / "X_train.csv")
    X_test = pd.read_csv(PROCESSED / "X_test.csv")
    y_train = pd.read_csv(PROCESSED / "y_train.csv").values.ravel()
    y_test = pd.read_csv(PROCESSED / "y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test


# Métrica parametrizável: ROC AUC (default) ou F1 se o usuário preferir


def compute_metric(y_true, y_pred, metric_name):
    if metric_name == "f1":
        return f1_score(y_true, (y_pred > 0.5).astype(int))
    # default = ROC AUC
    return roc_auc_score(y_true, y_pred)


# Treinar e avaliar um modelo com MLflow


def train_and_log(model_name, model, X_train, y_train, X_test, y_test, metric_name):
    with mlflow.start_run(run_name=model_name):
        # Treino
        model.fit(X_train, y_train)

        # Predição
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Métrica
        metric_value = compute_metric(y_test, y_pred_proba, metric_name)

        # Log de hiperparâmetros
        mlflow.log_params(model.get_params())
        mlflow.log_metric(metric_name, metric_value)

        # Logar modelo
        mlflow.sklearn.log_model(model, artifact_path="model")

        return {
            "model_name": model_name,
            "metric": metric_value,
            "run_id": mlflow.active_run().info.run_id,
        }


# Pipeline principal


def main():
    # Métrica parametrizável
    metric_name = os.getenv("METRIC", "roc_auc")
    model_registry_name = os.getenv("MODEL_NAME", "bank-model")

    print(f"Usando métrica: {metric_name}")
    print(f"Registrando melhor modelo como: {model_registry_name}")

    X_train, X_test, y_train, y_test = load_data()

    # Modelos simples e clássicos
    models = {
        "log_reg": LogisticRegression(max_iter=500),
        "xgb": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=1.0,
            colsample_bytree=1.0,
            eval_metric="logloss",
            n_jobs=2,
        ),
    }

    results = []

    for name, model in models.items():
        print(f"\nTreinando modelo: {name}")
        res = train_and_log(name, model, X_train, y_train, X_test, y_test, metric_name)
        results.append(res)

    # Escolher o melhor
    best = max(results, key=lambda r: r["metric"])
    print("\nMelhor modelo:", best)

    # Registrar versão no Model Registry
    model_uri = f"runs:/{best['run_id']}/model"
    mlflow.register_model(model_uri, model_registry_name)

    print("\nModelo registrado no MLflow:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
