import json
import pathlib

import pandas as pd
from dotenv import load_dotenv

from db import get_conn

# Carregar infra/.env (para rodar direto via python -m)
ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "infra" / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


def fetch_latest_training_snapshot():
    """
    Busca o último registro de treino na tabela training_data.
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            run_id,
            model_version,
            metric_name,
            metric_value,
            n_train,
            n_test,
            n_features,
            feature_stats
        FROM training_data
        ORDER BY timestamp DESC
        LIMIT 1;
        """
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise RuntimeError("Nenhum registro encontrado em training_data.")

    run_id, model_version, metric_name, metric_value, n_train, n_test, n_features, feature_stats = (
        row
    )

    # feature_stats vem como dict ou string (json)
    if isinstance(feature_stats, str):
        feature_stats = json.loads(feature_stats)

    return {
        "run_id": run_id,
        "model_version": model_version,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_features": int(n_features),
        "feature_stats": feature_stats,
    }


def fetch_recent_inferences(run_id: str, limit: int = 500):
    """
    Busca as últimas N inferências para um dado run_id,
    retornando lista de inputs (dict) e lista de predições.
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT input, prediction
        FROM inference_logs
        WHERE run_id = %s
        ORDER BY id DESC
        LIMIT %s;
        """,
        (run_id, limit),
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    inputs = []
    preds = []
    for inp, pred in rows:
        inputs.append(inp or {})
        preds.append(float(pred))

    return inputs, preds


def compute_simple_stats(df: pd.DataFrame, feature_keys):
    """
    Calcula estatísticas simples (mean, std, count) para as
    colunas numéricas presentes em feature_keys.
    """
    stats = {}

    for feat in feature_keys:
        if feat not in df.columns:
            continue

        s = df[feat].dropna()
        if s.empty:
            continue

        stats[feat] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "count": int(s.count()),
        }

    return stats


def main():
    print("\n=== Monitor de Drift - Bank Marketing ===\n")

    # 1) Buscar snapshot de treino
    print("Buscando snapshot de treino mais recente...")
    train = fetch_latest_training_snapshot()
    feature_stats_train = train["feature_stats"]

    print("\n[Snapshot de treino]")
    print(f"  run_id        : {train['run_id']}")
    print(f"  model_version : {train['model_version']}")
    print(f"  metric        : {train['metric_name']} = {train['metric_value']:.4f}")
    print(f"  n_train       : {train['n_train']}")
    print(f"  n_test        : {train['n_test']}")
    print(f"  n_features    : {train['n_features']}")

    # 2) Buscar últimas inferências para esse run_id
    print("\nBuscando últimas inferências para esse run_id...")
    inputs, preds = fetch_recent_inferences(train["run_id"], limit=500)

    if not inputs:
        print("⚠ Nenhuma inferência encontrada ainda para esse run_id.")
        return

    df_inf = pd.DataFrame(inputs)
    pred_series = pd.Series(preds)

    print(f"  Inferências carregadas: {len(df_inf)}")
    print("\n[Estatísticas das predições recentes]")
    print(
        f"  mean={pred_series.mean():.4f}  "
        f"std={pred_series.std():.4f}  "
        f"min={pred_series.min():.4f}  "
        f"max={pred_series.max():.4f}"
    )

    # 3) Comparar estatísticas de features numéricas
    print("\n[Comparação de features numéricas - treino vs inferência]")
    print("  Threshold para DRIFT: |Δmédia relativa| > 20%\n")

    inf_stats = compute_simple_stats(df_inf, feature_stats_train.keys())
    threshold = 0.20  # 20%

    for feat, train_stats in feature_stats_train.items():
        if feat not in inf_stats:
            continue

        train_mean = train_stats.get("mean")
        inf_mean = inf_stats[feat]["mean"]

        if train_mean is None:
            continue

        # Evita divisão por zero
        if train_mean == 0:
            rel_delta = None
        else:
            rel_delta = (inf_mean - train_mean) / abs(train_mean)

        status = "[OK]"
        extra = ""

        if rel_delta is not None and abs(rel_delta) > threshold:
            status = "[DRIFT]"
            extra = f" | Δrel={rel_delta * 100:.1f}%"

        print(
            f"{status} {feat:10s}  "
            f"mean_train={train_mean:8.3f}  "
            f"mean_infer={inf_mean:8.3f}{extra}"
        )

    print("\n=== Fim do relatório de monitoramento ===\n")


if __name__ == "__main__":
    main()
