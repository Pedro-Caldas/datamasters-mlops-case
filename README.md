# Data Masters — Case de Engenharia de Machine Learning

## 1. Objetivo do Case
## 2. Arquitetura de Solução e Arquitetura Técnica
## 3. Explicação do Case (Plano de Implementação)
## 4. Melhorias e Considerações Finais

## Como Rodar Localmente
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
make test


### `Makefile`
```bash
cat > Makefile << 'EOF'
.PHONY: fmt lint type test

fmt:
\tblack .
\truff check --fix .

lint:
\truff check .

type:
\tmypy src || true

test:
\tpytest -q || true

### Segurança e variáveis de ambiente

Este projeto inclui um arquivo `infra/.env.example` **apenas como template** para reproduzir a stack localmente e no CI.
- Em ambientes reais, **não versionar** `.env` com credenciais válidas.
- As variáveis sensíveis devem ser injetadas via **GitHub Secrets** ou cofre de segredos (ex.: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault).
- Aqui usamos **MinIO/Postgres locais e efêmeros** no Docker/CI; as credenciais são **dummy** e o estado é destruído ao final do job.
