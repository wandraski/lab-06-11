# AGENTS.md — Guia operacional para agentes de IA

Este documento orienta **qualquer agente de IA** (Claude Code, opencode, Copilot, Cursor, Codex etc.) a trabalhar neste
repositório sem quebrar os pipelines de CI, CT e CD. Ele descreve o estado-alvo do projeto já adaptado para a triagem de
diabetes e as regras para chegar lá.

> **Contexto**: este repo é a adaptação do projeto didático `ci_cd_ct_mlops` (classificação de tumores de mama) para a
> **triagem de diabetes** (dataset Pima Indians Diabetes), como exigido pela atividade. A adaptação segue a ordem das
> etapas da Parte C (seção "Definição de pronto" abaixo).

---

## 1. Visão geral

- **O que o projeto é**: um pipeline MLOps completo com três workflows do GitHub Actions:
  - **CI** (`ci.yml`): lint (ruff) + testes de código, dados, modelo e API — dispara em push na `main` e em todo PR.
  - **CT** (`ct.yml`): detecção de drift (teste KS), retreino versionado com DVC, portão de qualidade, promoção do campeão.
  - **CD** (`cd.yml`): tag `v*.*.*` → roda o CI, constrói a imagem Docker, teste de fumaça, publica no `ghcr.io`.
- **O problema**: classificação binária tabular — prever diabetes (`Outcome` = 1) a partir de 8 atributos numéricos
  clínicos (Pima Indians Diabetes, 768 linhas).
- **O contrato em uma frase**: só entra em produção um modelo com **recall da classe diabética ≥ 0,85**,
  **precisão ≥ 0,80** e que **não piore o recall do campeão atual**, medido no mesmo conjunto.

## 2. Regras de ouro (invioláveis)

1. **NUNCA renomeie o pacote `mlops_demo`**. O nome está acoplado a `pyproject.toml`, `dvc.yaml`, `Dockerfile`, aos três
   workflows e aos scripts de `demo/`. Renomear é a fonte de erro nº 1 desta atividade.
2. **Antes de commitar ou dar push, rode `python demo/local.py ci` e ele deve terminar tudo `[ok]`.** Só dê push com o
   CI local verde. Para mudanças em dados/pipeline, rode também `python demo/local.py ct`.
3. **Não edite `.github/workflows/*.yml`** a menos que o usuário peça explicitamente. Os três workflows são o coração
   da avaliação e já funcionam.
4. **`dvc.lock` NUNCA vai para o `.gitignore`** (o `dvc repro` falha com "is git-ignored"). **`models/champion/` é
   sempre versionado no Git** (é o modelo em produção). O que é ignorado: `data/processed`, `models/candidate`, `reports/`.
5. **Nenhum segredo, token ou senha no repositório.** Nada de chaves em código, tests ou docs. O único token usado é o
   `GITHUB_TOKEN` nativo, referenciado apenas nos workflows.
6. **Sem fork.** O repo da equipe é público, criado do zero. O histórico do projeto original foi removido.
7. **Um commit por pessoa, da conta de cada integrante.** O histórico entra na avaliação — não concentre tudo em uma
   conta nem commite como `github-actions[bot]` manualmente.
8. **Mudou limiar em `config.py`? Justifique no PR** (custo do erro). Limiares são contrato de negócio, não detalhe técnico.
9. **Não use fork, não reative histórico do projeto original, não commite artefatos gerados** (`data/processed/`,
   `models/candidate/`, `reports/`, `.venv/`, `__pycache__/`).
10. **Nada de dados pessoais ou sensíveis reais.** O dataset é anonimizado e de domínio público; não traga outros.

## 3. Ambiente e dependências

Pré-requisitos: **Python ≥ 3.10** e **Git**. Docker é opcional (só para o CD).

```bash
# ambiente virtual (obrigatório)
python -m venv .venv
source .venv/bin/activate          # Linux/macOS; no Git Bash: source .venv/Scripts/activate

# instalação única com todos os extras
pip install -e ".[dev,ct]"
```

Grupos de dependências (definidos no `pyproject.toml` — **não mexa**):

| Grupo | Pacotes | Para quê |
|---|---|---|
| runtime | `numpy>=1.26`, `pandas>=2.1`, `scikit-learn>=1.4`, `scipy>=1.11`, `joblib>=1.3`, `pyyaml>=6.0`, `fastapi>=0.110`, `pydantic>=2.5`, `uvicorn>=0.27` | modelo, API e pipeline |
| `dev` | `pytest>=8.0`, `httpx>=0.27`, `ruff>=0.6` | testes, lint e cliente HTTP do TestClient |
| `ct` | `dvc>=3.50` | pipeline versionado do treinamento contínuo |

**Armadilha**: o `dvc repro` chama o `python` do PATH. Com o venv inativo ele falha com `No module named 'mlops_demo'`.

## 4. Mapa do repositório

| Caminho | Papel | Quando mexer |
|---|---|---|
| `src/mlops_demo/config.py` | **Fonte única de verdade**: caminhos, `FEATURES`, `ALVO`, `CLASSE_DIABETICA`, `COLUNAS_DRIFT`, limiares | ao adaptar o domínio ou o contrato |
| `src/mlops_demo/data.py` | divisão da base, `validar_dataframe`, `aplicar_drift`, `gerar_lote`, `preparar` | etapas 2, 4, 7 |
| `src/mlops_demo/features.py` | `separar_xy` (ordem das colunas) e `construir_pipeline` (StandardScaler + LogisticRegression) | etapa 5 |
| `src/mlops_demo/train.py` | treina candidato (pipeline DVC) ou campeão inicial (`--campeao`) | etapa 11 (só para gerar) |
| `src/mlops_demo/evaluate.py` | `calcular_metricas` (recall/precisão da classe diabética) e `avaliar` | etapa 6 |
| `src/mlops_demo/gate.py` | portão de qualidade: compara candidato x campeão, escreve `reports/gate.*`, `--promover` | raramente (regras derivam de `config.py`) |
| `src/mlops_demo/monitor.py` | drift por teste KS, escreve `reports/drift.md`, output `retreinar` para o workflow | etapa 7 (só se mudar o gatilho) |
| `src/mlops_demo/api.py` | FastAPI: `/health` e `/predict` servindo o campeão | etapa 8 |
| `src/mlops_demo/__init__.py` | `__version__` do pacote | ao publicar versão nova |
| `tests/conftest.py` | fixtures de sessão: bases, params, campeão, `modelo_atual` | raramente |
| `tests/test_codigo.py` (marker `codigo`) | funções: `separar_xy`, pipeline, métricas, `decidir` do gate | etapa 9 |
| `tests/test_dados.py` (marker `dados`) | esquema/valores de todos os CSVs, proporção de classes, vazamento, duplicatas | etapa 9 |
| `tests/test_modelo.py` (marker `modelo`) | contrato do campeão, baseline, determinismo, probabilidades, invariância, teste direcional | etapa 9 |
| `tests/test_api.py` (marker `api`) | contrato da API: `/health`, formato do `/predict`, 422, casos claros | etapa 9 |
| `data/external/diabetes.csv` | fonte original (768 linhas), versionada | nunca editar à mão |
| `data/raw/base_treino.csv` / `base_holdout.csv` | bases finais (538/150), geradas pelo script, versionadas | etapa 1 (gerar, nunca editar à mão) |
| `data/raw/lotes/` | lotes novos do CT; **vazia por padrão** (só `.gitkeep`) | cenários e evidências E3/E4 |
| `models/champion/model.joblib` + `metrics.json` | **o modelo em produção, versionado** | gerado por `train --campeao` ou promoção do CT |
| `params.yaml` | hiperparâmetros (`semente`, `modelo.C`, `modelo.max_iter`, `modelo.class_weight`) | etapa 5 |
| `dvc.yaml` | estágios `preparar` e `treinar` | nunca (mantido do original) |
| `dvc.lock` | hashes do DVC — versionado, **não ignorar** | regenerado pelo `dvc repro` |
| `Dockerfile` | imagem da API com o campeão (`MODEL_PATH` env) | nunca |
| `demo/local.py` | espelha os workflows na máquina (`ci` / `ct`) | nunca |
| `demo/cenarios.py` | aplica os cenários 1–5 | etapa 10 |
| `demo/gerar_lote.py` | gera lote em `data/raw/lotes/` (com `--drift`) | evidências E3/E4 |
| `demo/smoke_test.py` | teste de fumaça contra API rodando | etapa 8 (ajustar para `Outcome`) |
| `.github/workflows/ci.yml`, `ct.yml`, `cd.yml` | os três pipelines | **nunca**, sem pedido explícito |
| `.github/PULL_REQUEST_TEMPLATE.md` | checklist de ML no PR | raramente |
| `.gitattributes` | LF obrigatório (`* text=auto eol=lf`), `*.joblib binary` | nunca |
| `pyproject.toml` | pacote, deps, config ruff e markers pytest | nunca (pacote mantém o nome) |
| `ADAPTACAO.md` | relatório da entrega | ao final |
| `README.md` | documentação do projeto | etapa 12 |

## 5. Arquitetura de dados (fluxo do modelo)

```
data/external/diabetes.csv (fonte, 768 linhas)
        │  python -m mlops_demo.data base   (semente fixa, stratify, imputação de zeros)
        ▼
base_treino.csv (538) ──────┐
base_holdout.csv (150) ─────┤
reserva (80) ──┐            │
               ▼            ▼
        data/raw/lotes/  ─► preparar (dvc stage) ─► data/processed/
                                                    treino.csv          ─► treinar ─► models/candidate/
                                                    avaliacao.csv       ─► gate ─► (aprovado?) ─► models/champion/
```

Regras que o agente precisa entender antes de mexer:

- **Campeão e candidato são avaliados no MESMO conjunto**: o holdout fixo + a metade de avaliação de cada lote. Nunca
  altere essa lógica em `preparar()`.
- **`SEMENTE_DADOS = 7` é fixa.** Mudar a divisão invalida a referência do campeão versionado (`metrics.json`).
- **O holdout nunca participa do treino** — o teste `test_nao_ha_vazamento_entre_treino_e_holdout` garante isso.
- **Zeros são missing no Pima**: zeros em `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` são
  clinicamente impossíveis. A imputação (mediana da base de treino) acontece no script de divisão. Depois disso, o
  validador deve **rejeitar** zeros nessas colunas — nunca "consertar" o validador para aceitar dados sujos.
- **`MIN_FEATURES_DRIFT = 3`** (de 8 atributos): o drift simulado afeta exatamente 3 colunas (`COLUNAS_DRIFT`), o que
  dispara o retreino no CT. Se `FEATURES` mudar, recalcule esse número.

## 6. Fonte única de verdade: `config.py`

Todas as constantes do domínio vivem aqui. **Nunca repita valores mágicos em outros arquivos** — importe de `config`:

| Constante | Valor-alvo | Nota |
|---|---|---|
| `FEATURES` | 8 colunas do Pima | str puro, não numpy.str_ |
| `ALVO` | `"Outcome"` | 1 = diabético, 0 = não diabético |
| `CLASSE_DIABETICA` | `1` | renomeada de `CLASSE_MALIGNA`; usada como `pos_label` em `evaluate.py` e na API |
| `COLUNAS_DRIFT` | `["Glucose", "Insulin", "BMI"]` | renomeada de `COLUNAS_TAMANHO`; drift + teste direcional |
| `RECALL_MINIMO` | `0.85` | contrato (falso negativo é o erro caro) |
| `PRECISAO_MINIMA` | `0.80` | impede "chutar tudo diabético" (recall 1,0) |
| `TOLERANCIA_CAMPEAO` | `0.0` | candidato não pode ter recall menor que o campeão |
| `PISO_SANIDADE` | `0.70` | piso do campeão no holdout fixo (abaixo do contrato de propósito) |
| `LIMITE_DRIFT_P` | `0.01` | p-valor do teste KS |
| `MIN_FEATURES_DRIFT` | `3` | gatilho do CT |

## 7. Comandos

Sempre com o venv ativo, a partir da raiz:

| Comando | Para quê |
|---|---|
| `python demo/local.py ci` | CI inteiro local: lint + validar dados + 4 camadas de teste. **Rode antes de todo commit** |
| `python demo/local.py ct` | CT inteiro local: monitor + `dvc repro` + gate |
| `ruff check .` e `ruff format --check .` | lint e formatação isolados |
| `pytest -m codigo` / `-m dados` / `-m modelo` / `-m api` | uma camada de teste isolada |
| `python -m mlops_demo.data base` | regenera treino/holdout/reserva a partir da fonte (semente fixa) |
| `python -m mlops_demo.data validar` | valida todos os CSVs de `data/raw/` (exit code 1 se houver problema) |
| `python -m mlops_demo.monitor` | relatório de drift + decisão de retreino |
| `dvc repro` | pipeline `preparar` + `treinar` (só refaz o que mudou) |
| `python -m mlops_demo.gate` | portão: exit 0 aprovado, 1 reprovado; escreve `reports/gate.json` e `gate.md` |
| `python -m mlops_demo.gate --promover` | copia candidato para `models/champion/` (só se aprovado) |
| `python -m mlops_demo.train --campeao` | **uso único**: treina e grava o primeiro campeão |
| `python demo/cenarios.py listar` / `aplicar N` | cenários de prática (sempre em branch nova) |
| `python demo/gerar_lote.py lote_02 --drift 0.75` | lote novo em `data/raw/lotes/` |
| `uvicorn mlops_demo.api:app --reload` | sobe a API localmente |
| `python demo/smoke_test.py http://localhost:8000` | fumaça contra a API rodando |

## 8. Convenções de código

- **Linter**: ruff com `line-length = 100`, `target-version = "py310"`, regras `E, F, I, B, UP`. Rode `ruff format .`
  antes de commitar. Imports ordenados (regra `I`).
- **Idioma**: todo o código, docstrings, mensagens, nomes de teste e identificadores em **português**, seguindo o
  estilo do repo (ex.: `RECALL_MINIMO`, `validar_dataframe`, `aplicar_drift`, `test_gate_reprova_recall_abaixo_do_limiar`).
  Não misture inglês.
- **Docstrings**: todo módulo público tem docstring explicando o papel e o uso na linha de comando (veja os módulos
  existentes como modelo). Funções públicas têm docstring curta.
- **Markers pytest obrigatórios**: todo teste novo precisa de marker (`codigo`, `dados`, `modelo`, `api`) — os markers
  já são herdados pelo `pytestmark` no topo de cada arquivo. `--strict-markers` está ligado no `pyproject.toml`.
- **Tipagem**: anotações de retorno nas funções públicas (padrão do repo), tipos prontos do Python 3.10+.
- **Sem comentários de código desnecessários**: prefira nomes claros e docstrings. Comentários só para armadilhas
  reais (como o aviso sobre a ordem das classes em `predict_proba` no original).

## 9. As quatro camadas de teste

O CI roda cada camada como um **step separado** (`pytest -m <camada>`). Ao adaptar, mantenha essa separação e
**acrescente pelo menos 1 teste novo por camada**, específico do domínio (exigência da atividade):

| Camada | Arquivo | O que já cobre | Sugestão de teste novo (domínio) |
|---|---|---|---|
| código | `test_codigo.py` | `separar_xy` (ordem, coluna faltante, imutabilidade), pipeline, métricas, regras do gate | `aplicar_drift` multiplica exatamente as 3 colunas de `COLUNAS_DRIFT` pelo fator |
| dados | `test_dados.py` | esquema/valores de todos os CSVs, proporção de classes, vazamento treino/holdout, duplicatas, validador detecta sujeira | zero em `Glucose`/`BMI` é rejeitado pelo `validar_dataframe`; proporção de diabéticos entre 0,25 e 0,55 |
| modelo | `test_modelo.py` | contrato do `metrics.json`, piso de sanidade, treino atual cumpre contrato, supera baseline, determinismo, memorização, probabilidades válidas, invariância a ruído, **teste direcional** | glicose alta isolada não reduz a probabilidade prevista de diabetes |
| API | `test_api.py` | `/health`, formato do `/predict`, 422 para payload errado, casos claros de cada classe | payload com 7 atributos → 422; `/health` expõe as métricas do campeão |

**Teste direcional**: no original, "tumor maior não reduz a chance de maligno". Aqui: **pior perfil metabólico**
(`COLUNAS_DRIFT` × 1,10) **não pode reduzir a probabilidade média de diabetes**. O sentido esperado mudou com o
domínio — ajuste o teste, não o sentido do modelo.

## 10. Receitas de tarefas comuns

### 10.1 Trocar o schema/FEATURES (ex.: renomear colunas)

1. Atualize `FEATURES` em `config.py`.
2. Regeneração dos dados: ajuste o script de divisão em `data.py` (nomes das colunas da fonte) e rode
   `python -m mlops_demo.data base`.
3. `validar_dataframe`: atualize faixas válidas e regras de zeros impossíveis.
4. Busque por literais de coluna em `tests/` (cenário 2 usa nomes de colunas) e em `demo/cenarios.py` — atualize todos.
5. `python demo/local.py ci` deve passar verde.

### 10.2 Ajustar limiares do contrato

1. Edite em `config.py` (`RECALL_MINIMO`, `PRECISAO_MINIMA`, ...).
2. Verifique se o modelo atual ainda cumpre: `pytest -m modelo`.
3. **Justifique no PR** (custo do erro). O checklist do template de PR cobra isso.

### 10.3 Adicionar um teste novo

1. Escolha o arquivo da camada certa (`test_codigo/dados/modelo/api.py`).
2. Escreva o teste em português, com marker herdado pelo `pytestmark`.
3. Prefira fixtures do `conftest.py` (`base_treino`, `base_holdout`, `params`, `campeao`, `modelo_atual`, `xy_holdout`).
4. Rode a camada isolada (`pytest -m dados -q`) e depois o CI local completo.

### 10.4 Gerar lote com drift (evidência E3)

1. `git switch -c lote-drift`
2. `python demo/gerar_lote.py lote_01 --drift 0.75 --n 60`
3. `python -m mlops_demo.data validar` (lote deve ser válido — drift não é sujeira)
4. Commit, push, abra o PR. O **CT** roda no PR: veja o **Summary** com a tabela de drift e o portão.
5. Após o merge, o CT roda de novo na `main` e abre o PR **"ct: novo modelo campeão aprovado no portão de qualidade"**
   (só `models/champion/` muda). Ele fica em **Action required** — clique em **Approve and run workflows**.

### 10.5 Recriar o campeão inicial

```bash
python -m mlops_demo.train --campeao   # treina na base e avalia no holdout
```

Use uma única vez (etapa 11) ou quando o domínio mudar por completo. O `metrics.json` do campeão é versionado e o CI
confere se ele cumpre o contrato (`test_campeao_versionado_foi_certificado_pelo_portao`).

### 10.6 Adaptar os cenários 1–5 (`demo/cenarios.py`)

Reescreva `_substituir`/corrupções com as colunas do Pima:

1. remover última coluna em `features.py` (`FEATURES[:-1]`) → CI falha em `codigo`.
2. nulos em `Glucose` + `BMI` negativos em `base_treino.csv` → CI falha em `dados` (zeros impossíveis são erro de domínio).
3. `params.yaml`: `C: 1.0` → `C: 0.00005` → CI falha em `modelo` (recall < 0,85).
4. `gerar_lote("lote_01", n=120, fator_drift=0.75, semente=1)` (amostra os 80 registros da reserva) → CI verde; CT dispara.
5. `params.yaml`: `class_weight: balanced` → `{0: 1, 1: 50}` → recall 1,0, precisão < 0,80; CI e portão reprovam.

Cada cenário deve quebrar **exatamente** onde a tabela do README diz.

### 10.7 Adaptar a API (30 → 8 atributos)

1. `api.py`: `atributos: list[float] = Field(min_length=8, max_length=8)` — use `len(config.FEATURES)` em vez de
   literais, se possível.
2. Resposta: `classe` ∈ {"diabetico", "nao_diabetico"}, `probabilidade_diabetes` (renomeie `probabilidade_maligno`).
3. `smoke_test.py`: remova a coluna com `linha.pop(config.ALVO)` (era `target`).
4. `tests/test_api.py`: atualize `set(corpo)` e os valores esperados de `classe`.
5. A leitura de `predict_proba` usa `list(modelo.classes_).index(config.CLASSE_DIABETICA)` — mantenha esse padrão
   (a ordem de `classes_` não é garantida).

### 10.8 Renomear métricas (recall_maligno → recall_diabetico)

As chaves de métricas aparecem em: `evaluate.py` (`calcular_metricas`), `gate.py` (`decidir`, `resumo_markdown`),
`tests/test_codigo.py` (helper `_metricas`), `tests/test_modelo.py`, `api.py`. Renomeie **todas** de uma vez
(`rg "recall_maligno|precisao_maligna|probabilidade_maligno"` e zere as ocorrências). Rode o CI local completo.

## 11. Armadilhas conhecidas

| # | Armadilha | Como evitar |
|---|---|---|
| 1 | O `30` de atributos está **hardcoded** em `api.py` e no teste de código | use `len(config.FEATURES)`; ao adaptar, troque todo literal |
| 2 | Venv inativo → `dvc repro` falha com `No module named 'mlops_demo'` | ative o venv; confirme com `which python` |
| 3 | `dvc.lock` no `.gitignore` → `dvc repro` falha com "is git-ignored" | nunca ignore `dvc.lock` |
| 4 | Zeros do Pima tratados como valor real | impute no script de divisão; o validador deve rejeitar zeros impossíveis |
| 5 | `pos_label` errado nas métricas | sempre `CLASSE_DIABETICA` (1) vindo de `config`, nunca literal |
| 6 | Ordem de `classes_` assumida em `predict_proba` | use `list(modelo.classes_).index(config.CLASSE_DIABETICA)` |
| 7 | Cenário "não aplica" (`trecho não encontrado`) | a mudança já foi aplicada na branch; volte à `main` e crie outra |
| 8 | CI do PR do campeão fica em **Action required** | normal: PR aberto pelo `GITHUB_TOKEN`; clique em **Approve and run workflows** |
| 9 | CT não abre PR | permissão "Allow GitHub Actions to create and approve pull requests" desligada |
| 10 | `data/raw/lotes/` com lote esquecido → CT retreina sem querer | a pasta fica vazia por padrão; apague lotes de teste antes de commitar em outra feature |
| 11 | Mudança de `params.yaml` dispara CT | é o comportamento esperado (o CI retreina e confere os limiares) |
| 12 | Line endings no Windows quebram hashes do DVC | `git config --local core.autocrlf false` + `.gitattributes` já força LF |
| 13 | Commitar `reports/`, `data/processed/`, `models/candidate/` | são gerados e ignorados; só `models/champion/` é versionado |

## 12. Definição de pronto (checklist da Parte C)

Marque cada etapa só quando `python demo/local.py ci` (ou `ct`) passar **verde** na etapa:

- [ ] **Etapa 1** — `base_treino.csv`/`base_holdout.csv` substituídos, sem sobreposição; `lotes/` vazia
- [ ] **Etapa 2** — sem `load_breast_cancer`; base reproduzível por script com semente fixa
- [ ] **Etapa 3** — `FEATURES`, `ALVO`, `CLASSE_DIABETICA`, `COLUNAS_DRIFT`, limiares em `config.py`
- [ ] **Etapa 4** — `validar_dataframe` com regras do domínio (faixas, zeros impossíveis, tipos)
- [ ] **Etapa 5** — modelo/hiperparâmetros consistentes (`features.py`, `train.py`, `params.yaml`)
- [ ] **Etapa 6** — métricas com `pos_label` da classe diabética
- [ ] **Etapa 7** — drift nas colunas certas; `MIN_FEATURES_DRIFT` coerente com nº de atributos
- [ ] **Etapa 8** — schema da API com 8 atributos; mensagens/respostas do domínio
- [ ] **Etapa 9** — 4 arquivos de teste adaptados + **≥ 1 teste novo por camada**
- [ ] **Etapa 10** — cenários 1–5 reescritos com as colunas do projeto
- [ ] **Etapa 11** — primeiro campeão gerado (`train --campeao`) e versionado
- [ ] **Etapa 12** — README atualizado
- [ ] **Evidências** — E1 a E5 geradas e linkadas no README/`ADAPTACAO.md`
- [ ] **Relatório** — `ADAPTACAO.md` com problema, contrato, mudanças, evidências, limites, contribuição e uso de IA

## 13. Fluxo de trabalho (como entregar mudanças)

1. `git switch main && git pull`
2. `git switch -c <nome-da-mudanca>` (branch curta e descritiva)
3. Faça a mudança menor possível (uma preocupação por PR)
4. Rode `python demo/local.py ci` (e `ct` se mexeu em dados/pipeline) — **tudo `[ok]`**
5. `git add -A && git commit -m "<descrição>"` — **commite da sua conta** (o histórico entra na avaliação)
6. `git push -u origin <branch>` e abra o PR
7. Preencha o template de PR (o que mudou + checklist de ML + impacto no modelo, colando `reports/gate.md` se mudou
   dados ou treino)
8. Acompanhe o **Checks** do PR; se vermelho, leia o step que falhou (cada camada é um step separado)
9. Merge só com CI verde. Feche PRs de cenário **sem merge**

**Modo de promoção**: `PROMOCAO_AUTOMATICA=false` durante o desenvolvimento (CT propõe, pessoa aprova). O modo
automático (`true`) é o desafio opcional (+5 pontos) e exige `main` sem proteção.

## 14. O que NÃO fazer

- Não renomear o pacote `mlops_demo` nem mexer nos três workflows sem pedido explícito.
- Não editar os CSVs de `data/raw/` à mão — sempre via script com semente fixa.
- Não "suavizar" testes ou validação para passar (ex.: baixar limiar sem justificativa, aceitar zeros impossíveis,
  pular marker de teste).
- Não commitar artefatos gerados, `dvc.lock` no ignore, segredos, ou alterar `models/champion/` à mão.
- Não usar fork, não reaproveitar histórico do projeto original, não deletar o `.gitattributes`.
- Não responder perguntas do usuário com mudanças de código sem antes rodar o CI local.
