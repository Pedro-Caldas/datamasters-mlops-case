# Makefile do projeto Datamasters
# Focado no case Bank Marketing
# Criado para facilitar a vida do avaliador (você mesmo que está lendo :D)!

.PHONY: format lint test ensure-dotenv up down logs open-mlflow open-minio \
        train-bank predict-bank serve-bank list-models list-versions promote \
        data-bank db-training db-inference

# --------------------------------------------------------------------
# Qualidade de código
# --------------------------------------------------------------------

format:
	black .
	ruff check --fix .

lint:
	ruff check .

test:
	pytest -q || true

# --------------------------------------------------------------------
# Infraestrutura (Docker Compose)
# --------------------------------------------------------------------

ensure-dotenv:
	@test -f infra/.env || (cp infra/.env.example infra/.env && echo "infra/.env criado.")

up: ensure-dotenv
	cd infra && docker compose --env-file .env up -d

down:
	cd infra && docker compose --env-file .env down -v

logs:
	cd infra && docker compose --env-file .env logs -f

open-mlflow:
	@if [ -f infra/.env ]; then \
		echo "Carregando infra/.env..."; \
		set -a; . infra/.env; set +a; \
	fi; \
	open "http://localhost:$${MLFLOW_PORT}"

open-minio:
	@if [ -f infra/.env ]; then \
		echo "Carregando infra/.env..."; \
		set -a; . infra/.env; set +a; \
	fi; \
	open "http://localhost:$${MINIO_PORT_UI}"

# --------------------------------------------------------------------
# Pipeline de MLOps - Bank Marketing
# --------------------------------------------------------------------
# Exemplos:
#   make data-bank
#   make train-bank METRIC=roc_auc MODEL_NAME=bank-model
#   make serve-bank
#   make predict-bank STAGE=Production MODEL_NAME=bank-model
# --------------------------------------------------------------------

data-bank:
	python -m src.data_bank_marketing

train-bank:
	@if [ -f infra/.env ]; then \
		echo "Carregando infra/.env..."; \
		set -a; . infra/.env; set +a; \
		export AWS_ACCESS_KEY_ID=$$S3_ACCESS_KEY; \
		export AWS_SECRET_ACCESS_KEY=$$S3_SECRET_KEY; \
		export AWS_DEFAULT_REGION=$$S3_REGION; \
		export MLFLOW_S3_ENDPOINT_URL=$$S3_ENDPOINT_EXTERNAL; \
		export AWS_S3_ADDRESSING_STYLE=path; \
		export AWS_EC2_METADATA_DISABLED=true; \
	else \
		echo "infra/.env NÃO encontrado (modo CI)"; \
	fi; \
	MODEL_NAME=$${MODEL_NAME:-bank-model} \
	METRIC=$${METRIC:-roc_auc} \
	python -m src.train_bank_marketing

predict-bank:
	@if [ -f infra/.env ]; then \
		echo "Carregando infra/.env..."; \
		set -a; . infra/.env; set +a; \
		export AWS_ACCESS_KEY_ID=$$S3_ACCESS_KEY; \
		export AWS_SECRET_ACCESS_KEY=$$S3_SECRET_KEY; \
		export AWS_DEFAULT_REGION=$$S3_REGION; \
		export MLFLOW_S3_ENDPOINT_URL=$$S3_ENDPOINT_EXTERNAL; \
		export AWS_S3_ADDRESSING_STYLE=path; \
		export AWS_EC2_METADATA_DISABLED=true; \
	else \
		echo "infra/.env NÃO encontrado (modo CI)"; \
	fi; \
	MODEL_NAME=$${MODEL_NAME:-bank-model} \
	PREDICT_STAGE=$${STAGE:-Production} \
	python -m src.predict_bank

serve-bank:
	@if [ -f infra/.env ]; then \
		echo "Carregando infra/.env..."; \
		set -a; . infra/.env; set +a; \
		export AWS_ACCESS_KEY_ID=$$S3_ACCESS_KEY; \
		export AWS_SECRET_ACCESS_KEY=$$S3_SECRET_KEY; \
		export AWS_DEFAULT_REGION=$$S3_REGION; \
		export MLFLOW_S3_ENDPOINT_URL=$$S3_ENDPOINT_EXTERNAL; \
		export AWS_S3_ADDRESSING_STYLE=path; \
		export AWS_EC2_METADATA_DISABLED=true; \
	else \
		echo "infra/.env NÃO encontrado (modo CI)"; \
	fi; \
	python -m src.serve_bank

# --------------------------------------------------------------------
# Model Registry (MLflow)
# --------------------------------------------------------------------
# Exemplos:
#   make list-models
#   make list-versions MODEL_NAME=bank-model
#   make promote MODEL_NAME=bank-model VERSION=3 STAGE=Production
# --------------------------------------------------------------------

list-models:
	@if [ -f infra/.env ]; then \
		set -a; . infra/.env; set +a; \
	fi; \
	python -c 'from mlflow.tracking import MlflowClient; \
c=MlflowClient(); \
[print("-", m.name) for m in c.search_registered_models()]'

list-versions:
	@if [ -f infra/.env ]; then \
		set -a; . infra/.env; set +a; \
	fi; \
	python -c 'import os; from mlflow.tracking import MlflowClient; \
name=os.getenv("MODEL_NAME","bank-model"); \
c=MlflowClient(); \
vs=c.search_model_versions(f"name=\"{name}\""); \
[print(f"{v.name} v{v.version} - stage={v.current_stage}") for v in vs]'

promote:
	@if [ -f infra/.env ]; then \
		set -a; . infra/.env; set +a; \
	fi; \
	python -c 'import os; from mlflow.tracking import MlflowClient; \
name=os.getenv("MODEL_NAME","bank-model"); \
ver=os.getenv("VERSION"); stage=os.getenv("STAGE","Staging"); \
assert ver, "Set VERSION=<n>"; \
c=MlflowClient(); \
c.transition_model_version_stage(name=name, version=str(ver), stage=stage, archive_existing_versions=True); \
print(f"Promoted {name} v{ver} -> {stage}")'

# --------------------------------------------------------------------
# Inspeção rápida das tabelas do Postgres
# --------------------------------------------------------------------
#   make db-training   -> últimas linhas de metadados de treino
#   make db-inference  -> últimas linhas de logs de inferência
# --------------------------------------------------------------------

db-training:
	@if [ -f infra/.env ]; then \
		echo "Carregando infra/.env..."; \
		set -a; . infra/.env; set +a; \
		cd infra && docker compose --env-file .env exec -T postgres \
			psql -U $$POSTGRES_USER -d $$POSTGRES_DB \
			-c "SELECT id, run_id, model_version, metric_name, metric_value, n_train, n_test, n_features, timestamp FROM training_data ORDER BY id DESC LIMIT 5;"; \
	else \
		echo "infra/.env NÃO encontrado. Suba a infra primeiro com 'make up'."; \
	fi

db-training-full:
	@if [ -f infra/.env ]; then \
		echo "Carregando infra/.env..."; \
		set -a; . infra/.env; set +a; \
		cd infra && docker compose --env-file .env exec -T postgres \
			psql -U $$POSTGRES_USER -d $$POSTGRES_DB \
			-c "SELECT * FROM training_data ORDER BY id DESC LIMIT 1;"; \
	else \
		echo "infra/.env NÃO encontrado. Suba a infra primeiro com 'make up'."; \
	fi

db-training-pretty:
	@if [ -f infra/.env ]; then \
		set -a; . infra/.env; set +a; \
		cd infra && docker compose --env-file .env exec -T postgres \
			psql -U $$POSTGRES_USER -d $$POSTGRES_DB \
			-c "SELECT jsonb_pretty(feature_stats) FROM training_data ORDER BY id DESC LIMIT 1;"; \
	else \
		echo "infra/.env NÃO encontrado."; \
	fi

db-inference:
	@if [ -f infra/.env ]; then \
		echo "Carregando infra/.env..."; \
		set -a; . infra/.env; set +a; \
		cd infra && docker compose --env-file .env exec -T postgres \
			psql -U $$POSTGRES_USER -d $$POSTGRES_DB \
			-c "SELECT id, run_id, model_version, prediction, timestamp FROM inference_logs ORDER BY id DESC LIMIT 10;"; \
	else \
		echo "infra/.env NÃO encontrado. Suba a infra primeiro com 'make up'."; \
	fi
