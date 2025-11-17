import json
import os
import pathlib
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
import psycopg2
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

# Limpar warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuração do MLflow
if os.getenv("CI"):
    mlflow.set_tracking_uri("file:./mlruns-ci")
else:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri("http://localhost:5050")

mlflow.set_experiment("bank-marketing")

# Caminho dos dados
ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def load_data():
    X_train = pd.read_csv(PROCESSED / "X_train.csv")
    X_test = pd.read_csv(PROCESSED / "X_test.csv")
    y_train = pd.read_csv(PROCESSED / "y_train.csv").values.ravel()
    y_test = pd.read_csv(PROCESSED / "y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test


def compute_metric(y_true, y_pred, metric_name):
    if metric_name == "f1":
        return f1_score(y_true, (y_pred > 0.5).astype(int))
    return roc_auc_score(y_true, y_pred)


def train_and_log(model_name, model, X_train, y_train, X_test, y_test, metric_name):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metric_value = compute_metric(y_test, y_pred_proba, metric_name)

        mlflow.log_params(model.get_params())
        mlflow.log_metric(metric_name, metric_value)

        signature = infer_signature(X_train, y_pred_proba)
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)

        return {
            "model_name": model_name,
            "metric": metric_value,
            "run_id": mlflow.active_run().info.run_id,
        }


# Persistência no Postgres — agora com feature_stats
def log_training_metadata_to_db(
    X_train,
    X_test,
    y_train,
    y_test,
    run_id,
    model_version,
    metric_name,
    metric_value,
):
    host = os.getenv("POSTGRES_HOST", "localhost")
    user = os.getenv("POSTGRES_USER")
    pwd = os.getenv("POSTGRES_PASSWORD")
    db = os.getenv("POSTGRES_DB")
    port = os.getenv("POSTGRES_PORT", "5432")

    n_train = len(X_train)
    n_test = len(X_test)
    n_features = X_train.shape[1]

    # Estatísticas de drift — super simples e super úteis
    feature_stats = X_train.describe().to_dict()

    try:
        conn = psycopg2.connect(
            host=host,
            user=user,
            password=pwd,
            dbname=db,
            port=port,
        )
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO training_data (
                run_id,
                model_version,
                metric_name,
                metric_value,
                n_train,
                n_test,
                n_features,
                feature_stats
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                run_id,
                str(model_version),
                metric_name,
                float(metric_value),
                int(n_train),
                int(n_test),
                int(n_features),
                json.dumps(feature_stats),
            ),
        )

        conn.commit()
        cur.close()
        conn.close()

        print("Metadados + feature_stats persistidos no Postgres.")

    except Exception as e:
        print("Erro ao salvar metadados de treino no Postgres:", e)


# Pipeline principal
def main():
    metric_name = os.getenv("METRIC", "roc_auc")
    model_registry_name = os.getenv("MODEL_NAME", "bank-model")

    print(f"Usando métrica: {metric_name}")
    print(f"Registrando melhor modelo como: {model_registry_name}")

    X_train, X_test, y_train, y_test = load_data()

    models = {
        "log_reg": LogisticRegression(max_iter=500),
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    for name, model in models.items():
        print(f"\nTreinando modelo: {name}")
        res = train_and_log(name, model, X_train, y_train, X_test, y_test, metric_name)
        results.append(res)

    best = max(results, key=lambda r: r["metric"])
    print("\nMelhor modelo:", best)

    # Registra apenas 1 versão
    model_uri = f"runs:/{best['run_id']}/model"
    registered = mlflow.register_model(model_uri, model_registry_name)
    version_number = registered.version

    # Salva metadados + feature_stats
    log_training_metadata_to_db(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        run_id=best["run_id"],
        model_version=version_number,
        metric_name=metric_name,
        metric_value=best["metric"],
    )

    print("\nModelo registrado no MLflow:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
