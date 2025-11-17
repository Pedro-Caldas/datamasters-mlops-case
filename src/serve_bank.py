import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

from src.db import save_inference_row

# Carregar envs
load_dotenv("infra/.env")

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("S3_REGION")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("S3_ENDPOINT_EXTERNAL")
os.environ["AWS_S3_ADDRESSING_STYLE"] = "path"
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))

# Colunas booleanas
BOOLEAN_COLS = [
    "job_blue-collar",
    "job_entrepreneur",
    "job_housemaid",
    "job_management",
    "job_retired",
    "job_self-employed",
    "job_services",
    "job_student",
    "job_technician",
    "job_unemployed",
    "job_unknown",
    "marital_married",
    "marital_single",
    "education_secondary",
    "education_tertiary",
    "education_unknown",
    "default_yes",
    "housing_yes",
    "loan_yes",
    "contact_telephone",
    "contact_unknown",
    "month_aug",
    "month_dec",
    "month_feb",
    "month_jan",
    "month_jul",
    "month_jun",
    "month_mar",
    "month_may",
    "month_nov",
    "month_oct",
    "month_sep",
    "poutcome_other",
    "poutcome_success",
    "poutcome_unknown",
]


def ensure_boolean_columns(df):
    for col in BOOLEAN_COLS:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


# Carregamento do modelo
def load_model():
    model_name = os.getenv("MODEL_NAME", "bank-model")

    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])

    if not versions:
        raise RuntimeError(f"Nenhum modelo em Production para {model_name}")

    v = versions[0]
    model_uri = f"runs:/{v.run_id}/model"

    # Carregar modelo sklearn diretamente (para ter predict_proba)
    model = mlflow.sklearn.load_model(model_uri)

    return model, v.run_id, v.version


model, model_run_id, model_version = load_model()

# API
app = FastAPI(title="Bank Marketing Model API")


class PredictRequest(BaseModel):
    input: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest):
    df = pd.DataFrame([payload.input])
    df = ensure_boolean_columns(df)

    # Calcular probabilidade e classe
    proba = float(model.predict_proba(df)[0, 1])
    pred_class = int(proba >= 0.5)

    # Salvar no Postgres
    save_inference_row(
        run_id=model_run_id,
        model_version=str(model_version),
        features=df.iloc[0].to_dict(),
        prediction=proba,
    )

    return {
        "class": pred_class,
        "probability": proba,
        "n_features": df.shape[1],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
