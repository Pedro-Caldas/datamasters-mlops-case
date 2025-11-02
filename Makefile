.PHONY: fmt lint type test env-host up down rebuild logs train mlflow minio train-reg promote-staging promote-prod predict-staging predict-prod

fmt:
	black .
	ruff check --fix .

lint:
	ruff check .

type:
	mypy src || true

test:
	pytest -q || true

env-host:
	set -a; . infra/.env; set +a; \
	export AWS_ACCESS_KEY_ID=$$S3_ACCESS_KEY; \
	export AWS_SECRET_ACCESS_KEY=$$S3_SECRET_KEY; \
	export AWS_DEFAULT_REGION=$$S3_REGION; \
	export MLFLOW_S3_ENDPOINT_URL=$$S3_ENDPOINT_EXTERNAL; \
	export AWS_S3_ADDRESSING_STYLE=path; \
	export AWS_EC2_METADATA_DISABLED=true; \
	python -c 'import os; print("Tracking:", os.getenv("MLFLOW_TRACKING_URI")); print("Artifacts:", os.getenv("MLFLOW_S3_ENDPOINT_URL")); print("AWS key present?", bool(os.getenv("AWS_ACCESS_KEY_ID")))'

ensure-dotenv:
	@test -f infra/.env || (cp infra/.env.example infra/.env && echo "infra/.env criado a partir de .env.example")

up:
	cd infra && docker compose --env-file .env up -d

down:
	cd infra && docker compose --env-file .env down -v

rebuild:
	cd infra && docker compose --env-file .env build --no-cache mlflow

logs:
	cd infra && docker compose logs -f

mlflow:
	open http://localhost:$(MLFLOW_PORT)

minio:
	open http://localhost:9001

train:
	set -a; . infra/.env; set +a; \
	export AWS_ACCESS_KEY_ID=$$S3_ACCESS_KEY; \
	export AWS_SECRET_ACCESS_KEY=$$S3_SECRET_KEY; \
	export AWS_DEFAULT_REGION=$$S3_REGION; \
	export MLFLOW_S3_ENDPOINT_URL=$$S3_ENDPOINT_EXTERNAL; \
	export AWS_EC2_METADATA_DISABLED=true; \
	python src/train_baseline.py

# Treina e registra, mas não promove
train-reg:
	set -a; . infra/.env; set +a; \
	export AWS_ACCESS_KEY_ID=$$S3_ACCESS_KEY; \
	export AWS_SECRET_ACCESS_KEY=$$S3_SECRET_KEY; \
	export AWS_DEFAULT_REGION=$$S3_REGION; \
	export MLFLOW_S3_ENDPOINT_URL=$$S3_ENDPOINT_EXTERNAL; \
	export AWS_EC2_METADATA_DISABLED=true; \
	MODEL_NAME=$${MODEL_NAME:-datamasters-elasticnet} MODEL_STAGE=None python src/train_baseline.py

# Lista modelos existentes
list-models:
	python -c 'from mlflow.tracking import MlflowClient; c=MlflowClient(); [print("-",m.name) for m in c.search_registered_models()]'

# Lista versões do mesmo modelo
list-versions:
	python -c 'import os; from mlflow.tracking import MlflowClient; \
	name=os.getenv("MODEL_NAME","datamasters-elasticnet"); \
	c=MlflowClient(); vs=sorted(c.search_model_versions(f"name=\"{name}\""), key=lambda v:int(v.version)); \
	[print(f"{name} v{v.version}  stage={v.current_stage}  run_id={v.run_id}") for v in vs]'

promote-staging:
	python -c 'import os; from mlflow.tracking import MlflowClient; \
	name=os.getenv("MODEL_NAME","datamasters-elasticnet"); \
	c=MlflowClient(); vs=sorted(int(v.version) for v in c.search_model_versions(f"name=\"{name}\"")); \
	assert vs, f"No versions for {name}"; \
	v=str(max(vs)); c.transition_model_version_stage(name=name, version=v, stage="Staging", archive_existing_versions=True); \
	print(f"Promoted {name} v{v} -> Staging")'

promote-prod:
	python -c 'import os; from mlflow.tracking import MlflowClient; \
	name=os.getenv("MODEL_NAME","datamasters-elasticnet"); \
	c=MlflowClient(); vs=sorted(int(v.version) for v in c.search_model_versions(f"name=\"{name}\"")); \
	assert vs, f"No versions for {name}"; \
	v=str(max(vs)); c.transition_model_version_stage(name=name, version=v, stage="Production", archive_existing_versions=True); \
	print(f"Promoted {name} v{v} -> Production")'

# Promove versão específica de forma parametrizável
# Ex: make promote VERSION=3 STAGE=Staging MODEL_NAME=meu-modelo
promote:
	python -c 'import os; from mlflow.tracking import MlflowClient; \
	name=os.getenv("MODEL_NAME","datamasters-elasticnet"); \
	ver=os.getenv("VERSION"); stage=os.getenv("STAGE","Staging"); \
	assert ver, "Set VERSION=<n>"; c=MlflowClient(); \
	c.transition_model_version_stage(name=name, version=str(ver), stage=stage, archive_existing_versions=True); print(f"Promoted {name} v{ver} -> {stage}")'

predict-staging:
	set -a; . infra/.env; set +a; \
	export AWS_ACCESS_KEY_ID=$$S3_ACCESS_KEY; \
	export AWS_SECRET_ACCESS_KEY=$$S3_SECRET_KEY; \
	export AWS_DEFAULT_REGION=$$S3_REGION; \
	export MLFLOW_S3_ENDPOINT_URL=$$S3_ENDPOINT_EXTERNAL; \
	export AWS_S3_ADDRESSING_STYLE=path; \
	export AWS_EC2_METADATA_DISABLED=true; \
	MODEL_NAME=datamasters-elasticnet PREDICT_STAGE=Staging python src/predict.py

predict-prod:
	set -a; . infra/.env; set +a; \
	export AWS_ACCESS_KEY_ID=$$S3_ACCESS_KEY; \
	export AWS_SECRET_ACCESS_KEY=$$S3_SECRET_KEY; \
	export AWS_DEFAULT_REGION=$$S3_REGION; \
	export MLFLOW_S3_ENDPOINT_URL=$$S3_ENDPOINT_EXTERNAL; \
	export AWS_S3_ADDRESSING_STYLE=path; \
	export AWS_EC2_METADATA_DISABLED=true; \
	MODEL_NAME=datamasters-elasticnet PREDICT_STAGE=Production python src/predict.py
