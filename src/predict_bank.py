import json
import os
import pathlib

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from db import save_inference_row

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def load_input():
    df = pd.read_csv(PROCESSED / "X_test.csv")
    return df.sample(20, random_state=42)


def load_production_model(model_name, stage="Production"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))

    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[stage])

    if not versions:
        raise ValueError(f"No versions for model {model_name} in stage {stage}")

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

    # Persistir inferências
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
