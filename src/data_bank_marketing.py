import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split

# Caminhos base do projeto
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "bank-full.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main(test_size: float = 0.2, random_state: int = 42) -> None:
    """
    Lê o arquivo bank-full.csv, faz um split simples em treino e teste
    e salva os arquivos prontos em data/processed/.
    """

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # O arquivo original usa ; como separador
    df = pd.read_csv(RAW_PATH, sep=";")

    # Alvo: coluna "y" (yes/no) -> 1/0
    df["y"] = df["y"].map({"no": 0, "yes": 1})

    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Salvamos tudo em CSV simples, sem complicar
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print("Arquivos salvos em:", PROCESSED_DIR)


if __name__ == "__main__":
    main()
