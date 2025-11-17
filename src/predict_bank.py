import json
import os
import pathlib

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.db import save_inference_row

# Carregar variáveis de ambiente
ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "infra" / ".env"
load_dotenv(ENV_PATH)

# Configurar acesso ao MinIO/S3
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("S3_REGION")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("S3_ENDPOINT_EXTERNAL")
os.environ["AWS_S3_ADDRESSING_STYLE"] = "path"
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

# Configurar MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))

# Caminho dos dados processados
PROCESSED = ROOT / "data" / "processed"


def load_input():
    df = pd.read_csv(PROCESSED / "X_test.csv")
    return df.sample(20, random_state=42)


def load_production_model(model_name, stage="Production"):
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[stage])

    if not versions:
        raise ValueError(f"Nenhuma versão em {stage} para o modelo {model_name}")

    version = versions[0]
    run_id = version.run_id
    model_uri = f"runs:/{run_id}/model"

    model = mlflow.sklearn.load_model(model_uri)
    return model, run_id, model_uri


def main():
    df = load_input()

    model_name = os.getenv("MODEL_NAME", "bank-model")
    model, run_id, model_uri = load_production_model(model_name)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[:, 1]
    else:
        preds = model.predict(df)
        proba = preds.astype(float)

    for i in range(len(df)):
        save_inference_row(
            run_id=run_id,
            model_version="production",
            features=df.iloc[i].to_dict(),
            prediction=float(proba[i]),
        )

    print(
        json.dumps(
            {
                "predictions": proba.tolist(),
                "input_shape": list(df.shape),
                "model_uri": model_uri,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
