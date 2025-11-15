import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "bank-full.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main(test_size=0.2, random_state=42):
    print("Carregando dataset...")
    df = pd.read_csv(RAW_PATH, sep=";")

    # Converter target
    df["y"] = df["y"].map({"no": 0, "yes": 1})

    # Split features/target
    X = df.drop(columns=["y"])
    y = df["y"]

    # Faz um one-hot encoding só com o get_dummies
    X = pd.get_dummies(X, drop_first=True)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Salvar
    print("Salvando arquivos processados...")
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print("Processamento concluído com sucesso.")


if __name__ == "__main__":
    main()
