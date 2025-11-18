# DataMasters - Case de Engenharia de Machine Learning

**Autor**: Pedro Neris Luiz Caldas


## Conteúdo  
1. [Objetivo do Case e Como Executá-lo](#1-objetivo-do-case-e-como-executá-lo)  
2. [Arquitetura de Solução e Arquitetura Técnica](#2-arquitetura-de-solução-e-arquitetura-técnica)  
   - [2.1 Visão Geral](#21-visão-geral)  
   - [2.2 Justificativas Técnicas](#22-justificativas-técnicas)
   - [2.3 Diagrama básico com fluxo principal da solução](#23-diagrama-básico)  
3. [Plano de Implementação](#3-plano-de-implementação)  
   - [3.1 Levantamento de requisitos](#31-levantamento-de-requisitos)  
   - [3.2 Aquisição e Preparação de Dados](#32-aquisição-e-preparação-de-dados)  
   - [3.3 Desenvolvimento, Tracking e Registro do Modelo](#33-desenvolvimento-tracking-e-registro-do-modelo)  
   - [3.4 Integração Contínua e Implantação](#34-integração-contínua-e-implantação)  
   - [3.5 Monitoramento e Métricas de Produção](#35-monitoramento-e-métricas-de-produção)  
   - [3.5.1 Serving do Modelo](#351-serving-do-modelo)  
   - [3.6 Escalabilidade e Ambientes de Produção](#36-escalabilidade-e-ambientes-de-produção)  
4. [Melhorias e Considerações Finais](#4-melhorias-e-considerações-finais)




### 1. Objetivo do Case e Como Executá-lo

Neste repositório consta a solução que desenvolvi para o Case de Engenharia de Machine Learning do programa Data. O foco foi projetar e implementar uma arquitetura completa de MLOps — capaz de receber dados brutos, treinar e versionar modelos, automatizar o pipeline, disponibilizar inferência online e monitorar em produção —, sempre com escalabilidade e em mente.

Para ilustrar o projeto, escolhi trabalhar com o dataset Bank Marketing (da UCI), focado em Marketing/CRM em um cenário bancário. O objetivo é, basicamente, prever se um cliente assinará um depósito a prazo ou não. Aqui eu tentei fugir um pouco dos clássicos modelos de risco, crédito etc. e ir pra algo diferente. Mesmo sendo um conjunto de dados de tamanho relativamente pequeno, ele permite demonstrar todos os principais blocos do ciclo de vida de um modelo de ML: ingestão de dados, transformação e gerenciamento de features, treino com diferentes algoritmos, tracking e registro de modelo, inferência batch e online, monitoramento de produção e execução via ambiente conteinerizado com CI/CD.

Acho importante deixar claro que a minha estratégia no desafio foi priorizar a amplitude em vez de profundidade. Ou seja: a ideia era mostrar o máximo de estágios possíveis de uma solução de ML/MLOps — do dado bruto ao serviço em produção sendo monitorado — em vez de me aprofundar excessivamente em um único ponto e acabar deixando de lado alguma etapa relevante. Isso me permitiu demonstrar (espero!) competência em engenharia de MLOps de ponta a ponta. Alguns componentes estão em nível de prova de conceito (ex.: uma Feature Store offiline bem superficial ou a falta de complexidade na etapa de treinamento dos modelos), mas essa escolha foi consciente, dado que, se eu me aprofundasse muito, provavelmente não conseguiria entregar os outros componentes.

Nas próximas seções você vai encontrar a arquitetura adotada, as decisões técnicas, o plano de implementação, os trade-offs que fiz e as melhorias futuras previstas — todos feitos tentando manter ao máximo o alinhamento aos requisitos do edital da Academia Santander.

#### 1.2 Como Executar a Solução Localmente

Há no projeto um arquivo Makefile com diversos comandos criados para facilitar a interação. Vamos usá-los como guia para a execução do projeto.

<span style="font-size: 0.9em;"> **PS**: Por default, o código atribuirá a um modelo treinado o nome 'bank-model' caso o usuário não atribua um específico. Portanto usaremos esse mesmo nome como exemplo no decorrer do guia. </span> 

**1. Clone o repositório**: \
`git clone https://github.com/Pedro-Caldas/datamasters-mlops-case.git` \

E acesse a pasta \
`cd datamasters-mlops-case`

**2. Crie uma env e instale as dependências**: \
<span style="font-size: 0.9em;"> Recomendado: ambiente virtual `Python 3.11` </span>

`python -m venv venv` \
`source venv/bin/activate` ou `venv\Scripts\activate` no Windows \
`pip install -e ".[dev]"`

<span style="font-size: 0.9em;"> Opcional: você pode usar o `uv` para criar e usar o ambiente também. </span>
    
**3. Levante a infraestrutura**: (MLflow, MinIO e PostgreSQL) via Docker Compose \
`make up`

**4. Processe os dados**: (que já estão presentes no repositório) e os deixe prontos para serem treinados \
`make data-bank`

**5. Treine os modelos** (Regressão Logística e Random Forest) e automaticamente registre aquele com melhor performance no registry de modelos: \
`make train-bank MODEL_NAME=bank-model`

<span style="font-size: 0.9em;"> **Opcionais**:
- <span style="font-size: 0.9em;"> Você pode escolher o nome do modelo. Se deixar em branco, ele usa 'bank-model' por default. </span>
- <span style="font-size: 0.9em;"> Por default, usamos roc_auc como métrica, mas você pode passar `METRIC=f1` após o `MODEL_NAME` para usar f1 em vez de roc_auc. </span>
- <span style="font-size: 0.9em;"> Após o treinamento, você pode checar os dados de treinamento salvos no DB com o comando \
`make db-training` ou `db-training-pretty` (para todas as estatísticas) </span>

**6. Liste os modelos registrados no MLflow** e você verá o seu: \
`make list-models`

**7. Liste as versões registradas de um modelo** para ver todas as versões de um mesmo modelo: \
`make list-versions MODEL_NAME=bank-model`

<span style="font-size: 0.9em;"> **PS**: você também verá o STAGE do modelo. </span>

**8. Promova o modelo registrado à produção** (apenas modelos em produção podem ser usados para inferência): \
`make promote MODEL_NAME=bank-model VERSION=[versão-escolhida] STAGE=Production`

<span style="font-size: 0.9em;"> **Opcional**: você pode promover um modelo ao estágio de Staging usando `STAGE=Staging`, no entanto, não poderá servir ele </span>

**9. Rode o script para inferências batch** \
`make predict-bank`

**10. Inicie o serviço de inferência online** \
`make serve-bank`

**11. Acesse a API** (via navegador ou ferramentas como Postman) em \
<span style="font-size: 0.9em;">http://localhost:8000/health </span>\
<span style="font-size: 0.9em;">http://localhost:8000/predict </span>

Para abrir o Swagger da API no navegador, use \
<span style="font-size: 0.9em;">http://localhost:8000/docs </span>

**12. Envie requisições ao modelo** via UI do `Swagger` ou `curl`
    
<span style="font-size: 0.9em;"> <details> <summary> Clique para ver exemplos de requisição </summary>

**Via UI do Swagger**:

**/health**:
`Apenas clique em "Try it out" e então em "Execute"`

**/predict**:
`{
"input": {
    "age": 40,
    "balance": 640,
    "day": 8,
    "duration": 347,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "job_blue-collar": true,
    "job_entrepreneur": false,
    "job_housemaid": false,
    "job_management": false,
    "job_retired": false,
    "job_self-employed": false,
    "job_services": false,
    "job_student": false,
    "job_technician": false,
    "job_unemployed": false,
    "job_unknown": false,
    "marital_married": true,
    "marital_single": false,
    "education_secondary": false,
    "education_tertiary": false,
    "education_unknown": false,
    "default_yes": false,
    "housing_yes": true,
    "loan_yes": true,
    "contact_telephone": false,
    "contact_unknown": true,
    "month_aug": false,
    "month_dec": false,
    "month_feb": false,
    "month_jan": false,
    "month_jul": false,
    "month_jun": false,
    "month_mar": false,
    "month_may": true,
    "month_nov": false,
    "month_oct": false,
    "month_sep": false,
    "poutcome_other": false,
    "poutcome_success": false,
    "poutcome_unknown": true
}
}`


**Via curl**: 

**/health**: 
`curl -X GET http://localhost:8000/health`

**/predict**: 
`curl -X POST http://localhost:8000/predict 
-H "Content-Type: application/json" \
-d '{
"input": {
    "age": 40,
    "balance": 640,
    "day": 8,
    "duration": 347,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "job_blue-collar": true,
    "job_entrepreneur": false,
    "job_housemaid": false,
    "job_management": false,
    "job_retired": false,
    "job_self-employed": false,
    "job_services": false,
    "job_student": false,
    "job_technician": false,
    "job_unemployed": false,
    "job_unknown": false,
    "marital_married": true,
    "marital_single": false,
    "education_secondary": false,
    "education_tertiary": false,
    "education_unknown": false,
    "default_yes": false,
    "housing_yes": true,
    "loan_yes": true,
    "contact_telephone": false,
    "contact_unknown": true,
    "month_aug": false,
    "month_dec": false,
    "month_feb": false,
    "month_jan": false,
    "month_jul": false,
    "month_jun": false,
    "month_mar": false,
    "month_may": true,
    "month_nov": false,
    "month_oct": false,
    "month_sep": false,
    "poutcome_other": false,
    "poutcome_success": false,
    "poutcome_unknown": true
}
}'`
</details> 

<span style="font-size: 0.9em;"> **Opcional**: Após as inferências (batch ou online), você pode checar os dados persistidos no DB com o comando \
`make db-inference`</span>

**13. Rode o módulo de observabilidade para checar drift em produção** \
`make monitor-bank`

**14. Rode a bateria de testes**, caso queira \
`make coverage`

**15. Derrube a infra** \
`make down`

### 2. Arquitetura de Solução e Arquitetura Técnica
#### 2.1 Visão Geral

A solução está organizada em camadas integradas para cobrir todo o ciclo de vida de modelo de ML Ops. Isso inclui:

**Ingestão, Pré-processamento e Gerenciamento de Features**: O módulo de dados lê o CSV bruto, mapeia o target, aplica one-hot encoding e divide em treino/teste. Além disso, há um registro estático das colunas de entrada (via o arquivo _feature_registry.yaml_), que funciona como uma versão simplificada de uma feature store offline — ou seja, define um “contrato de features” reutilizável entre treino e inferência, garantindo consistência no espaço vetorial.

**Treinamento & Registro**: Toda vez que o vez que um modelo é treinado, dois algoritmos são testados (nesse caso, Regressão Logística e Random Forest) visando garantir o melhor resultado baseado na métrica escolhida pelo usuário. O modelo vencedor é registrado via MLflow Model Registry. Dados de treino (metadados, número de features e estatísticas) são salvos em banco de dados para rastreabilidade e futura observabilidade.

**Inferência Batch & Online**: O pipeline batch faz predições sobre os dados que chegam via arquivo local e grava os logs completos. A API online (FastAPI) expõe endpoint para predição em tempo real e grava logs de cada requisição.

**Monitoramento & Observabilidade**: Um módulo compara estatísticas de treino vs inferência (drift simples) e alerta quando há divergência relevante em alguma feature específica.

**Infraestrutura & Automação**: A solução está containerizada (Docker Compose), com artefatos armazenados em MinIO (S3-compatible), banco de dados PostgreSQL para o back-end do MLflow e persistência de metadados de treino e inferência, e o pipeline de CI/CD via GitHub Actions que faz lint, testes, geração de dados, treino automático e promoção de modelo.

**Testes Automatizados**: Contamos com uma suíte robusta de testes implementada com pytest que cobre quase 90 % do código-fonte. Os testes são executados automaticamente no pipeline de CI (via GitHub Actions), garantindo que alterações no código ou nas configurações não quebrem funcionalidades existentes.

#### 2.2 Justificativas Técnicas

**1.** Optei por utilizar o CSV como formato de entrada para facilitar a dinâmica do case -- por ser mais leve e amplamente compatível com diferentes plataformas. Fiz questão de escolher um _corpus_ que coubesse dentro do próprio repositório Git para que vocês (avaliadores) não precisassem baixar os dados à parte.

**2.** Usei MLflow para rastrear experimentos e versionar modelos porque atende diretamente ao requisito de gerenciamento de artefatos e reprodutibilidade. É open-source e disparado a solução mais usada no mercado (inclusive no próprio banco via Gutenberg e Databricks).

**3.** O backend de metadados está em PostgreSQL — isso permite que todos os parâmetros, métricas, versões de modelo e estatísticas de treino/inferência fiquem persistidos de forma segura, auditável e escalável. Esse arranjo também apoia a transição de ambiente de protótipo para produção.

**4.** O armazenamento de artefatos no MinIO permite que o sistema seja facilmente portado para ambientes cloud ou escalado horizontalmente com o mínimo de modificação.

**5.** A escolha de Python + scikit‑learn para os algoritmos garante rapidez de implementação e facilita a compreensão dos resultados no contexto de protótipo. É uma escolha que abrange muitas possibilidades.

**6.** A arquitetura baseada em containers via Docker Compose permite que qualquer pessoa reproduza o ambiente localmente. Aqui a solução usa um único arquivo docker-compose, com configuração mínima de rede e de volumes (para simplificar o desenvolvimento do case). No entanto, quando for necessário escalar, essa mesma imagem pode ser, inclusive, usada como base para criar manifestos Kubernetes (Deployments/Services/ConfigMaps).

**7.** Aproveito pra reforçar que a escolha da estratégia de amplitude (abranger o máximo de camadas possível) em vez de profundidade permite demonstrar domínio amplo de engenharia de ML/MLOps, alinhado ao objetivo do case.

#### 2.3 Diagrama básico com fluxo principal da solução

                                      ┌───────────────────────────┐
                                      │     Dados Brutos (.csv)   │
                                      └────────────┬──────────────┘
                                                   │
                                                   ▼
                              ┌──────────────────────────────────────────┐
                              │ Ingestão & Pré-processamento             │
                              │ (data_bank_marketing.py)                 │
                              │ • mapeia target, one-hot encoding        │
                              │ • split treino/teste                     │
                              │ • salva em data/processed                │
                              └────────────┬─────────────────────────────┘
                                           │
                                           ▼
                             ┌────────────────────────────────────────────┐
                             │ Treinamento & Registro de Modelo           │
                             │ (train_bank_marketing.py)                  │
                             │ • experimentos (LogReg, RF)                │
                             │ • métrica (roc_auc / f1)                   │
                             │ • registro no ─► MLflow Registry           │
                             │ • meta-dados no PostgreSQL training_data   │
                             └────────────┬───────────────────────────────┘
                                          │
                                          ▼
                ┌────────────────────────────────────────────────────────────┐
                │ Gerenciamento de Artefatos                                 │
                │ • MLflow server + Model Registry                           │
                │ • MinIO / S3-style para artefatos                          │
                │ • PostgreSQL para metadados (training_data, inference_logs)│
                └────────────┬───────────────────────────────────────────────┘
                             │
                             ▼
          ┌────────────────────────────────────────────────────────────────────────┐
          │ Inferência Batch & Online                                              │
          │ • Batch: predict_bank.py → carrega X_test, faz predições, grava logs   │
          │ • Online: serve_bank.py → FastAPI endpoint /health & /predict          │
          │   grava logs no PostgreSQL inference_logs                              │
          └────────────┬───────────────────────────────────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────────────────────────────────────────────────────┐
          │ Monitoramento & Observabilidade                                        │
          │ (monitor_bank.py)                                                      │
          │ • pega snapshot de treino (feature_stats)                              │
          │ • pega últimas inferências                                             │
          │ • compara médias desvio, cheque drift                                  │
          └────────────┬───────────────────────────────────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────────────────────────────────────────────────────┐
          │ Infraestrutura & Automação                                             │
          │ • Docker Compose (Postgres, MinIO, MLflow)                             │
          │ • CI/CD via GitHub Actions (lint, tests, data-bank, train-bank etc)    │
          │ • Artefatos versionados + pipeline preparada para Kubernetes           │
          └────────────────────────────────────────────────────────────────────────┘

### 3. Plano de Implementação

#### 3.1 Levantamento de requisitos  
Na fase inicial, definiu-se:  
- Funcionais: Criar um “contrato de features” reutilizável, receber dados brutos em CSV, processar para treino/teste, treinar e registrar modelos de classificação binária que preveem se o cliente assinará depósito a prazo, disponibilizar inferência batch e online, monitorar produção (desempenho e drift).  
- Não-funcionais: Reprodutibilidade (rodar em máquina diferente), versionamento de modelo, CI/CD automatizado, cobertura ampla de testes, escalabilidade-horizontal e observabilidade em produção.

#### 3.2 Aquisição e Preparação de Dados  
- Os dados brutos foram obtidos a partir do dataset bancário em `https://archive.ics.uci.edu/dataset/222/bank+marketing` (UCI) e armazenados em `data/raw/bank-full.csv`.  
- O módulo `src/data_bank_marketing.py` lê esse CSV, mapeia o target (“yes” → 1, “no” → 0), aplica one-hot encoding via `pd.get_dummies(drop_first=True)`, e então separa em treino/teste com `train_test_split(test_size=0.2, stratify=y)`.  
- Os artefatos gerados (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`) são salvos em `data/processed/`.  
- **Poderia ter sido feito**: uso de `ColumnTransformer` + `Pipeline` do scikit-learn para modularizar melhor, logging da estatística de feature preprocessing, tratamento de valores faltantes/outliers mais sofisticado, e armazenamento desses artefatos em sistema de arquivos distribuído ou data lake para grande volume.
- **Motivo do trade-off**: O foco foi mostrar todos os estágios da solução em menor escala, então optou-se por uma pipeline direta e compreensível, em vez de construir toda a infraestrutura de dados de produção. Sei que no banco temos uma enorme cadeia de dados que envolve diferentes áreas, fluxos, armazenamento em formato Medallion (bronze, silver e gold), governança de acesso etc., mas não caberia aqui ir nessa direção.

#### 3.3 Desenvolvimento, Tracking e Registro do Modelo de Machine Learning  
- Os modelos testados foram: `LogisticRegression(max_iter=500)` e `RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)`, definidos no script `src/train_bank_marketing.py`.  
- A métrica padrão de avaliação foi `roc_auc`, com possibilidade de trocar para `f1` via variável de ambiente `METRIC` se for da preferência do usuário.
- O código registra parâmetros, métrica, e modelo em execução do MLflow (versão 2.16.0) e então os compara e registra o vencedor no Model Registry com nome definido por `MODEL_NAME`.
- Estatísticas do treino (n_train, n_test, n_features, feature_stats=descrição de X_train) são persistidas em banco PostgreSQL na tabela `training_data`.
- Os modelos podem ser versionados e levados a estágios como Staging e Production. Sempre que uma nova versão de um modelo entra em estágio de Production, a anterior é arquivada.
- **Poderia ter sido feito**: aqui poderíamos ir muito longe. Hyper-tuning automatizado (GridSearch/RandomSearch/Optuna), comparação com algoritmos mais avançados (XGBoost, LightGBM), cross-validation mais extensa, pipeline de features customizadas, automação de seleção de features. Tudo isso de forma dinâmica e abstraída para o usuário (ex.: usuário poderia criar um arquivo de config com os modelos que quer usar, métricas, features etc.).
- **Motivo do trade-off**: A opção por dois modelos clássicos e parâmetros fixos permitiu focar mais na orquestração e na engenharia de MLOps do que em ajuste fino de modelo. Posso estar errado, mas penso que o aprofundamento no treinamento do modelo (apesar de muito bem-vindo a um Engenheiro de ML), é avaliado na carreira de Cientista de Dados do programa Datamaster.

#### 3.4 Integração Contínua e Implantação  
- O repositório possui arquivo `.github/workflows/ci.yaml` que define pipeline de CI: a cada push ou pull request para qualquer branch, são executados checkout, setup Python 3.11, instalação das dependências (`pip install -e ".[dev]"`), criação de `.env` de exemplo, lint (`make lint`), testes (`make test`), geração de dados (`make data-bank`), treino em modo CI (`make train-bank MODEL_NAME=ci-test`), e promoção de modelo para “Production” no ambiente local. Como parte da estratégia de governança de modelos, o pipeline permite promover versões via MLflow também para o estágio ‘Staging’ antes da transição para ‘Production’. Em ambiente real é recomendável que apenas modelos com estágio Production possam servir de fato, garantindo controle de segurança, aprovação de versões e compliance organizacional, por isso aplicamos essa política aqui também no projeto.
- Durante o desenvolvimento utilizei um arquivo `Makefile` que oferece comandos para levantar a infraestrutura (`make up`), derrubar (`make down`), verificar logs (`make logs`), abrir interfaces MLflow/MinIO (`make open-mlflow`, `make open-minio`), processar dados (`make data-bank`), treino (`make train-bank`), inferência (`make predict-bank`), serviço (`make serve-bank`), monitorar (`make monitor`) etc. Esse comandos, além de úteis também na avaliação, servem para delinear o pipeline de MLOps.
- **Poderia ter sido feito**: pipeline de CD real (deploy automático em ambiente de produção ou staging, rollback automático, blue/green deployment), monitoração de pipelines, deploy em cluster orquestrado, Kubernetes etc.  
- **Motivo do trade-off**: O tempo e escopo requeriam uma entrega funcional (que passasse pelo máximo de blocos possível); portanto, a entrega de CI real foi priorizada, enquanto CD foi simulada/local.

#### 3.5 Monitoramento e Métricas de Produção
- A tabela `training_data` registra para cada execução de treino: run_id, model_version, métrica de desempenho, número de observações de treino/teste, número de features, estatísticas das features (JSONB) e timestamp. A tabela `inference_logs` registra para cada predição: run_id, model_version, input JSON, prediction float, timestamp.
- O módulo `src/monitor_bank.py` acessa o último snapshot de treino (via `training_data`), lê as últimas inferências (`inference_logs`), calcula estatísticas (média, std, count) das features detectadas nas inferências, compara com estatísticas de treino e sinaliza drift se variação relativa média > 20%.  
- **Poderia ter sido feito**: dashboards de observabilidade em tempo real (inclusive com métricas mais sofisticadas como KS e PSI), alertas in-line (e-mail/SMS), monitores de performance de latência, taxa de erro, saturação, e retenção automática de modelo retrain-trigger. Além, é claro, de um super retreino automatizado baseado em detecção automática de algum threshold ultrapassado.
- **Motivo do trade-off**: A implementação visa demonstrar o bloco de observabilidade de forma funcional, com código simples e compreensível. A complexidade em alertas e dashboards ficou fora do escopo primário para manter foco em amplitude.

#### 3.5.1 Serving do Modelo  
- O serviço de inferência online é implementado com FastAPI em `src/serve_bank.py`, expondo endpoints `/health` e `/predict`. O módulo carrega o modelo em produção (apenas os modelos em Stage Production do Model Registry podem ser usados) em uma API REST, recebe payloads JSON com recursos, garante o tipo correto das colunas booleanas, calcula probabilidade e classe, grava o log da inferência no banco, e retorna a resposta com `class`, `probability` e `n_features`.  
- Para inferência em batch, o script `src/predict_bank.py` carrega uma amostra de `X_test.csv`, usa o modelo em produção, grava logs de inferência e imprime resultado JSON com `predictions`, `input_shape`, `model_uri` e `model_version`.  
- **Poderia ter sido feito**: versionamento de endpoints (ex.: v1/v2), deploy blue/green ou canary, monitoramento de latência por endpoint, escala automática de serviço em produção. No caso da inferência batch, hoje usamos os próprios dados de teste chumbados, mas em produção deveríamos conseguir receber qualquer batch de input.  
- **Motivo do trade-off**: Foi priorizado um serviço funcional e reproduzível que mostra claramente o caminho das inferências online e batch, dentro do escopo do case.

#### 3.6 Escalabilidade e Ambientes de Produção  
- A infraestrutura é definida em `infra/docker-compose.yaml`, com serviços: Postgres (backend MLflow + metadados e estatísticas), MinIO (S3 compatible para artefatos), MLflow server (modelo + artefatos).  
- A solução é executável localmente com `make up` (que já roda tudo por trás), e tem suporte para rodar em outros ambientes (via Docker).  
- Como dito anteriormente, o uso de artefatos via S3 (MinIO) e tracking de modelos via MLflow permite implantação em ambiente cloud ou cluster horizontalizado com mínima modificação.  
- Embora não tenha sido implementado um orquestrador completo neste caso (ex.: via Airflow), a arquitetura foi desenhada para suportar essa camada: pipelines de tratamento, treino, inferência e promoção de modelo já são acionados via Makefile + CI.
- **Poderia ter sido feito**: ambiente de produção em Kubernetes (com autoscaling, replica sets), orquestração de pipeline em cluster (Kubernetes, Airflow), tratamento de dados em escala (Spark, Dask, BigQuery), streaming real-time.
- **Motivo do trade-off**: Como o foco era demonstrar toda a cadeia de ML Ops de ponta a ponta, optou-se por ambiente containerizado leve e reproduzível localmente, o que atende ao requisito de escalabilidade conceitual para o case.

### 4. Melhorias e Considerações Finais

#### Principais Aprendizado
Durante o desenvolvimento deste projeto, ficou mais claro do que nunca que construir uma solução de Engenharia de ML vai muito além do treinamento e disponibilização de um modelo — requer atenção aos pipelines, à infraestrutura, à governança de dados e à manutenção constante. No meu caso, venho de alguns anos trabalhando na Pltaforma Gutenberg, onde tive a noção da amplitude (e profundidade!) do tema. Aqui, pude novamente aplicar diversos desses conceitos e aprender muito mais sobre eles: treinamento, feature store (mesmo que em versão bem inicial) rastreamento de modelo via MLflow, serving online, logs de inferência persistidos para observabilidade, dockerização da infraestrutura, além de muitos outros.

#### Melhorias previstas  
| Área                                  | Ação planejada                                                                 | Justificativa                                                                                  |
|---------------------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| Feature engineering & pipeline de pré-processamento | Implantar `ColumnTransformer` + `Pipeline` no scikit-learn e persistir artefatos de pré-processamento | Aumenta a modularidade e permite reaplicar a pipeline em produção sem “drift” por inconsistência |
| Feature Store & Governança de Features | Evoluir a “feature_registry.yaml” atual para uma implementação de Feature Store (offline + online) com versionamento de features e lookup em produção | Facilita reutilização de features, evita *training-serving skew* e prepara o pipeline para escala real de ML |
| Kedro para pipeline de ciência de dados | Implementar um framework (ex.: Kedro) para estruturar pipelines modulares que permitam aos cientistas de dados criar modelos já prontos para produção | Garante modularidade, reutilização e transição mais suave dos protótipos para a produção |
| Orquestração de fluxo de trabalho     | Integrar um orquestrador como Apache Airflow ou Prefect para agendamento de dados e monitoramento de pipeline | Facilita automação real de ETL, treino, inferência e re-treino, o que eleva a maturidade do sistema |
| Implantação em produção com escalabilidade real | Migrar da configuração Docker Compose para um cluster Kubernetes com autoscaling, replicação de serviço e balanceamento de carga | Garante que a solução suporte picos, seja resiliente e tenha arquitetura cloud-native           |
| Gestão de Secrets & Configs em Kubernetes | Evoluir o uso atual de variáveis de ambiente (hoje com .env.example) para, por exemplo, modelos de ConfigMaps e Secrets no Kubernetes | Fortalece segurança, versionamento de configurações sensíveis, facilita implantação em cluster e segue boas práticas de orquestração de containers|
| Monitoramento avançado & alertas      | Adicionar dashboard de métricas (ex.: Grafana + Prometheus), alertas de drift, latência, erro e gatilhos de re-treino automático | Eleva o nível de observabilidade para produção real, reduz risco de degradação silenciosa      |
| Hiper-tuning e A/B testing            | Introduzir frameworks para otimização de hiper-parâmetros (ex.: Optuna) e comparar versões de modelos em produção | Permite melhoria contínua da performance do modelo, aumentando valor de negócio                |

Essas são algumas melhorias que demonstram o fluxo de evolução da solução. A ideia é mostrar que as decisões tomadas foram conscientes e estratégicas, mas que também tenho clareza sobre as **diversas possibilidades** de incrementos e melhorias contínuas.

#### Considerações finais  
A solução entregue atende de forma abrangente aos requisitos do edital: inclui processamento de dados, comparação de algoritmos, persistência de metadados, CI/CD, inferência batch e online, monitoramento simples, e infraestrutura dockerizada. A escolha consciente de priorizar amplitude faz com que a solução evidencie boa parte das camadas de engenharia de ML, mesmo que algumas funcionalidades estejam em nível de protótipo.

Além disso, a arquitetura foi pensada para ser **reprodutível**: os avaliadores com acesso ao repositório poderão levantar os serviços (via `docker-compose`), gerar dados, treinar modelos, executar inferências e visualizar logs com poucos comandos.

Por fim, vale reforçar que **MLOps é uma jornada** durante a qual aprendi (aprendo e aprenderei) muito. Espero que esta solução demonstre capacidade técnica, clareza de propósito e maturidade de engenharia — e que a banca enxergue tanto a solução entregue como o plano de evolução futuro.

**Muito obrigado!**
