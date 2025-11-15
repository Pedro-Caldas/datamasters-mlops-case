# Makefile do projeto Datamasters
# Criado para facilitar a vida do avaliador!

.PHONY: format lint test ensure-dotenv up down logs open-mlflow open-minio \
        train predict promote data-bank

# Ferramentas de qualidade

format:
	black .
	ruff check --fix .

lint:
	ruff check .

test:
	pytest -q || true

# Infraestrutura

ensure-dotenv:
	@test -f infra/.env || (cp infra/.env.example infra/.env && echo "infra/.env criado.")

up: ensure-dotenv
	cd infra && docker compose --env-file .env up -d

down:
	cd infra && docker compose --env-file .env down -v

logs:
	cd infra && docker compose logs -f

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

# Pipeline de MLOps
# --
# Exemplos:
#   make train MODEL_NAME=bank-model
#   make predict STAGE=Staging MODEL_NAME=bank-model

train:
	@set -a; . infra/.env; set +a; \
	MODEL_NAME=$${MODEL_NAME:-datamasters-model} python src/train_baseline.py

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
		echo "⚠️  infra/.env NÃO encontrado (modo CI)"; \
	fi; \
	MODEL_NAME=$${MODEL_NAME:-bank-model} METRIC=$${METRIC:-roc_auc} python -m src.train_bank_marketing


predict:
	@set -a; . infra/.env; set +a; \
	export AWS_ACCESS_KEY_ID=$$S3_ACCESS_KEY; \
	export AWS_SECRET_ACCESS_KEY=$$S3_SECRET_KEY; \
	export AWS_DEFAULT_REGION=$$S3_REGION; \
	export MLFLOW_S3_ENDPOINT_URL=$$S3_ENDPOINT_EXTERNAL; \
	export AWS_S3_ADDRESSING_STYLE=path; \
	export AWS_EC2_METADATA_DISABLED=true; \
	STAGE=$${STAGE:-Production} MODEL_NAME=$${MODEL_NAME:-datamasters-model} python src/predict.py

# Model Registry
# Exemplo:
#   make promote VERSION=3 STAGE=Production MODEL_NAME=bank-model

list-models:
	python -c 'from mlflow.tracking import MlflowClient; c=MlflowClient(); \
	[print("-", m.name) for m in c.search_registered_models()]'

list-versions:
	python -c 'import os; from mlflow.tracking import MlflowClient; \
	name=os.getenv("MODEL_NAME","datamasters-model"); \
	c=MlflowClient(); vs=c.search_model_versions(f"name=\"{name}\""); \
	[print(f"{v.name} v{v.version} - stage={v.current_stage}") for v in vs]'

promote:
	python -c 'import os; from mlflow.tracking import MlflowClient; \
	name=os.getenv("MODEL_NAME","datamasters-model"); \
	ver=os.getenv("VERSION"); stage=os.getenv("STAGE","Staging"); \
	assert ver, "Set VERSION=<n>"; \
	c=MlflowClient(); \
	c.transition_model_version_stage(name=name, version=str(ver), stage=stage, archive_existing_versions=True); \
	print(f"Promoted {name} v{ver} -> {stage}")'

# Dados do case Bank Marketing

data-bank:
	python -m src.data_bank_marketing
