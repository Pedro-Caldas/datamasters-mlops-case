# Makefile do projeto Datamasters
# Focado no case Bank Marketing
# Criado para facilitar a vida do avaliador!

.PHONY: format lint test ensure-dotenv up down logs open-mlflow open-minio \
        train-bank predict-bank serve-bank \
        list-models list-versions promote \
        data-bank db-training db-training-full db-training-pretty db-inference \
        monitor-bank

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

# make up: só infra :)
up: ensure-dotenv
	cd infra && docker compose --env-file .env up -d postgres postgres-migrate minio minio-setup mlflow

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

# Serviço local de inferência via FastAPI
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

db-training:
	@if [ -f infra/.env ]; then \
		echo "Carregando infra/.env..."; \
		set -a; . infra/.env; set +a; \
		cd infra && docker compose --env-file .env exec -T postgres \
		  psql -U $$POSTGRES_USER -d $$POSTGRES_DB \
		  -c "SELECT id, run_id, model_version, metric_name, metric_value, n_train, n_test, n_features, timestamp FROM training_data ORDER BY id DESC LIMIT 5;"; \
	else \
		echo "infra/.env NÃO encontrado."; \
	fi

db-training-full:
	@if [ -f infra/.env ]; then \
		set -a; . infra/.env; set +a; \
		cd infra && docker compose --env-file .env exec -T postgres \
		  psql -U $$POSTGRES_USER -d $$POSTGRES_DB \
		  -c "SELECT * FROM training_data ORDER BY id DESC LIMIT 1;"; \
	fi

db-training-pretty:
	@if [ -f infra/.env ]; then \
		set -a; . infra/.env; set +a; \
		cd infra && docker compose --env-file .env exec -T postgres \
		  psql -U $$POSTGRES_USER -d $$POSTGRES_DB \
		  -c "SELECT jsonb_pretty(feature_stats) FROM training_data ORDER BY id DESC LIMIT 1;"; \
	fi

db-inference:
	@if [ -f infra/.env ]; then \
		set -a; . infra/.env; set +a; \
		cd infra && docker compose --env-file .env exec -T postgres \
		  psql -U $$POSTGRES_USER -d $$POSTGRES_DB \
		  -c "SELECT id, run_id, model_version, prediction, timestamp FROM inference_logs ORDER BY id DESC LIMIT 10;"; \
	fi

# --------------------------------------------------------------------
# Monitoramento simples de drift
# --------------------------------------------------------------------

monitor-bank:
	python -m src.monitor_bank

# --------------------------------------------------------------------
# Testes
# --------------------------------------------------------------------

test:
	pytest -q

coverage:
	coverage run -m pytest -q
	coverage report -m
	coverage html
	@echo "📊 Abra o relatório em: htmlcov/index.html"
