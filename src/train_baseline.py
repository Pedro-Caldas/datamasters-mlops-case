import os

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# --- Se o usuário rodar o arquivo direto (sem `make train`), mapeia variáveis
if not os.getenv("AWS_ACCESS_KEY_ID"):
    if os.getenv("S3_ACCESS_KEY"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY", "")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY", "")
        os.environ["AWS_DEFAULT_REGION"] = os.getenv("S3_REGION", "us-east-1")
    if os.getenv("S3_ENDPOINT_EXTERNAL") and not os.getenv("MLFLOW_S3_ENDPOINT_URL"):
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("S3_ENDPOINT_EXTERNAL")
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")


def main():
    if os.getenv("CI"):
        # CI do GitHub usa store local
        mlflow.set_tracking_uri("file:./mlruns-ci")
    else:
        # Ambiente local usa MLFLOW_TRACKING_URI se houver
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("http://localhost:5050")

    mlflow.set_experiment("baseline")

    model_name = os.getenv("MODEL_NAME", "datamasters-elasticnet")
    target_stage = os.getenv(
        "MODEL_STAGE", "None"
    )  # Podendo ser "Staging", ou "Production" ou "None"

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    alpha = float(os.getenv("ALPHA", 0.5))
    l1_ratio = float(os.getenv("L1_RATIO", 0.5))

    with mlflow.start_run(run_name="elasticnet-baseline") as run:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id

        print(f"RMSE={rmse:.4f}")

        # --- Registro no Model Registry
        client = MlflowClient()
        # cria registro de modelo se ainda não existir
        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(model_name)

        artifact_uri = mlflow.get_artifact_uri("model")

        # versão nova a partir da run atual
        mv = client.create_model_version(
            name=model_name,
            source=artifact_uri,
            run_id=run_id,
        )
        print(f"Model version created: name={model_name} version={mv.version}")

        # promove stage se o usuário pedir e já arquiva versão anterior que esteja no mesmo estágio
        if target_stage in {"Staging", "Production"}:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage=target_stage,
                archive_existing_versions=True,
            )
            print(f"Promoted {model_name} v{mv.version} -> {target_stage}")


if __name__ == "__main__":
    main()
