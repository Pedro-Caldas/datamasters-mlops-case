import os

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# --- Shim: se o usuário rodar direto (sem `make train`), mapeia variáveis agnósticas S3_* -> AWS_*
if not os.getenv("AWS_ACCESS_KEY_ID"):
    if os.getenv("S3_ACCESS_KEY"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY", "")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY", "")
        os.environ["AWS_DEFAULT_REGION"] = os.getenv("S3_REGION", "us-east-1")
    if os.getenv("S3_ENDPOINT_EXTERNAL") and not os.getenv("MLFLOW_S3_ENDPOINT_URL"):
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("S3_ENDPOINT_EXTERNAL")
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))
    mlflow.set_experiment("baseline")

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    alpha = float(os.getenv("ALPHA", 0.5))
    l1_ratio = float(os.getenv("L1_RATIO", 0.5))

    with mlflow.start_run(run_name="elasticnet-baseline"):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
