# tests/test_data_bank_marketing.py
import pandas as pd

import src.data_bank_marketing as dbm


def test_main_creates_processed_files(tmp_path, monkeypatch):
    """
    Testa se:
      - o script lê o CSV de entrada (RAW_PATH)
      - faz o split e o one-hot encoding
      - salva X_train, X_test, y_train, y_test em PROCESSED_DIR
    usando um dataset pequeno fake em diretório temporário.
    """

    # ----- 1) Cria um CSV fake parecido com o esperado -----
    # Precisamos pelo menos da coluna "y" e algumas features
    data = {
        "age": [30, 40, 50, 35, 45, 32, 38, 41, 29, 55],
        "job": [
            "admin.",
            "blue-collar",
            "technician",
            "services",
            "admin.",
            "technician",
            "services",
            "blue-collar",
            "admin.",
            "services",
        ],
        "balance": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "y": ["no", "yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes"],
    }
    df = pd.DataFrame(data)

    raw_csv_path = tmp_path / "bank-full.csv"
    df.to_csv(raw_csv_path, sep=";", index=False)

    # ----- 2) Aponta o módulo para usar nossos caminhos temporários -----
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(dbm, "RAW_PATH", raw_csv_path)
    monkeypatch.setattr(dbm, "PROCESSED_DIR", processed_dir)

    # ----- 3) Executa o main -----
    dbm.main(test_size=0.2, random_state=42)

    # ----- 4) Verifica se os arquivos foram criados -----
    x_train_path = processed_dir / "X_train.csv"
    x_test_path = processed_dir / "X_test.csv"
    y_train_path = processed_dir / "y_train.csv"
    y_test_path = processed_dir / "y_test.csv"

    assert x_train_path.exists()
    assert x_test_path.exists()
    assert y_train_path.exists()
    assert y_test_path.exists()

    # ----- 5) Verificações básicas de consistência -----
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0]
    y_test = pd.read_csv(y_test_path).iloc[:, 0]

    # Mesmas colunas em X_train e X_test (depois do get_dummies)
    assert list(X_train.columns) == list(X_test.columns)

    # Tamanhos batendo
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(df)

    # Target só com 0 e 1
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})
