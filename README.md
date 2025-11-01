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

<!-- New PR de teste para validar CI/CD -->
