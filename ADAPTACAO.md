# Adaptação: triagem de diabetes (Pima Indians Diabetes)

**Estado:** adaptação, validação cruzada sem vazamento, pipelines CI/CT/CD, evidências **E1–E5** e imagem pública no GHCR concluídos e verificados em [`wandraski/lab-06-11`](https://github.com/wandraski/lab-06-11). Esta é uma demonstração acadêmica de MLOps e **não deve ser usada para diagnóstico ou decisão clínica**.

## 1. Problema, dados, procedência e licença

O pipeline classifica a variável binária `Outcome` (`1` = diabetes, classe positiva; `0` = não diabético) a partir de oito atributos clínicos numéricos (`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`), com **768 registros** (268 positivos ≈ 34,90% e 500 negativos ≈ 65,10%).

- **Procedência e licença (fatos comprovados vs. não verificados):** o arquivo `data/external/diabetes.csv` (768 linhas, SHA-256 `bf81d06e...`) foi obtido de [`plotly/datasets`](https://github.com/plotly/datasets/blob/master/diabetes.csv) (repositório sob licença **MIT**). A coorte original provém do *National Institute of Diabetes and Digestive and Kidney Diseases* (NIDDK) e foi distribuída pelo *UCI Machine Learning Repository* (mulheres de herança Pima com idade ≥ 21 anos). Os termos originais de consentimento/licenciamento do depósito primário NIDDK/UCI não constam no CSV e são registrados como não verificados.
- **Divergência quanto a dados sensíveis:** o Pima é a escolha definitiva da equipe, porém contém registros clínicos reais de saúde; a desidentificação não elimina sua natureza sensível, divergindo da restrição do enunciado sobre dados sensíveis reais (sem evidência de aprovação formal de exceção pelo professor).
- **Divisão e imputação sem vazamento:** `src/mlops_demo/data.py` divide os 768 registros de forma estratificada e reproduzível (`SEMENTE_DADOS = 7`) em **538 de treino** (`base_treino.csv`), **150 de holdout** (`base_holdout.csv`) e **80 de reserva** (origem dos lotes em `data/raw/lotes/`, vazia na `main`). Zeros biologicamente impossíveis em `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` e `BMI` codificam ausência de medição e são imputados pelas medianas calculadas **exclusivamente no treino** (`116,0`, `72,0`, `29,0`, `125,0` e `32,0`).

## 2. Contrato de qualidade, custo do erro e validação cruzada no treino

- **Custo do erro:** na triagem, o **falso negativo** é o erro mais grave (paciente com diabetes liberado sem exame confirmatório e sem tratamento precoce), enquanto o **falso positivo** gera um exame laboratorial confirmatório adicional (custo menor). Logo, exige-se `RECALL_MINIMO > PRECISAO_MINIMA`, além de `recall do candidato >= recall do campeão` avaliados no mesmo conjunto (`TOLERANCIA_CAMPEAO = 0.0`).
- **Investigação em validação cruzada (exclusivamente no treino, sem vazamento):** executamos `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` nos **538 registros brutos de treino** (ajustando `SimpleImputer(strategy="median")` e escalonamento **dentro de cada fold** via `Pipeline`):

| Família / Configuração (5-fold CV no treino bruto, N = 538) | Recall | Precisão | F1 | Acurácia | ROC-AUC | Prec. máx (Rec ≥ 0,85) | Rec. máx (Prec ≥ 0,80) |
|---|---|---|---|---|---|---|---|
| `LogisticRegression (C=1.0, class_weight='balanced')` *(modelo adotado)* | 0,707 | 0,624 | 0,663 | 0,749 | 0,842 | 0,559 | 0,426 |
| `PolyFeatures(2) + LogisticRegression (C=0.1, balanced)` | 0,697 | 0,642 | 0,668 | 0,758 | 0,845 | 0,566 | 0,410 |
| `RandomForest (n=300, max_depth=6, balanced)` | 0,761 | 0,614 | 0,679 | 0,749 | 0,827 | 0,569 | 0,303 |
| `ExtraTrees (n=300, max_depth=6, balanced)` | 0,723 | 0,624 | 0,670 | 0,751 | 0,838 | 0,573 | 0,367 |
| `GradientBoosting` / `HistGradientBoosting` / `SVC RBF` / `MLP` | 0,553–0,824 | 0,467–0,658 | 0,573–0,635 | 0,610–0,743 | 0,747–0,818 | 0,000–0,565 | 0,000–0,202 |

- **Decisão e recalibração formal ([PR #1](https://github.com/wandraski/lab-06-11/pull/1)):** o limiar inicial de exemplo (`0,85/0,80`) mostrou-se estatisticamente inalcançável no Pima sem vazamento (em `recall ≥ 0,85`, a precisão máxima em CV é **0,573**; em `precisão ≥ 0,80`, o recall máximo é **0,452**), falhando no CI inicial ([Run #42430933800](https://github.com/wandraski/lab-06-11/actions/runs/42430933800)). Seguindo a Regra 8 do repositório, o contrato foi formalmente recalibrado via PR para **`RECALL_MINIMO = 0,68`**, **`PRECISAO_MINIMA = 0,65`** e **`PISO_SANIDADE = 0,60`**. No holdout fixo (150 linhas), o campeão inicial obtém **recall = 0,7115**, **precisão = 0,6727** e **acurácia = 0,7800**.

## 3. O que mudou, arquivo por arquivo (Parte C)

- **Etapas 1–4 (`data/`, `src/mlops_demo/data.py`, `config.py`):** remoção de `load_breast_cancer`; divisão 538/150/80 com semente 7 e imputação por medianas do treino; `FEATURES` (8 colunas), `ALVO = "Outcome"`, `CLASSE_DIABETICA = 1`, `COLUNAS_DRIFT = ["Glucose", "Insulin", "BMI"]`, `MIN_FEATURES_DRIFT = 3`; `validar_dataframe` cobrindo esquema, tipos, nulos, não finitos, alvo em `{0, 1}`, negativos, zeros impossíveis e `Age < 21`.
- **Etapas 5–8 (`features.py`, `train.py`, `evaluate.py`, `gate.py`, `monitor.py`, `api.py`, `demo/smoke_test.py`, `dvc.lock`):** métricas `recall_diabetico` e `precisao_diabetica` (`pos_label = CLASSE_DIABETICA`); drift KS nas 3 colunas metabólicas; API FastAPI de 8 atributos retornando `diabetico`/`nao_diabetico` e `probabilidade_diabetes` via `list(_modelo.classes_).index(config.CLASSE_DIABETICA)`.
- **Etapas 9–12 (`tests/`, `demo/cenarios.py`, `models/champion/`, `README.md`, `ADAPTACAO.md`):** 35 testes nas 4 camadas (11 `codigo`, 8 `dados`, 10 `modelo`, 6 `api`) com novos testes de domínio; ajuste justificado de `test_modelo_consegue_memorizar_um_lote_pequeno` para `>= 0,75` em 8 dimensões ruidosas (obtém `0,80` contra baseline `0,625`); cenários 1–5 adaptados ao Pima.

## 4. As cinco evidências (Parte D) e pacote publicado

*(Nota Parte B: o diretório `repo-git` já continha a adaptação inicial in-place antes do primeiro commit; por integridade, o histórico não foi reescrito para simular execução prévia do dataset de câncer de mama).*

| Evidência | Descrição e resultado verificado | Link da execução / PR |
|---|---|---|
| **E1** | **CI verde na `main`** após a adaptação: 3 jobs verdes (`Lint e formatação (ruff)`, `Testes (Python 3.11)`, `Testes (Python 3.12)`). | [Run #42431356862](https://github.com/wandraski/lab-06-11/actions/runs/42431356862) ([PR #1](https://github.com/wandraski/lab-06-11/pull/1)) |
| **E2** | **CI vermelho por falha controlada** (`cenario-1`): remoção da última coluna em `features.py` falha no step `Testes de código`. PR fechado sem merge. | [Run #42431366983](https://github.com/wandraski/lab-06-11/actions/runs/42431366983) ([PR #2](https://github.com/wandraski/lab-06-11/pull/2)) |
| **E3** | **CT com drift e portão aprovando** (`cenario-4`): `lote_01.csv` (40 linhas, `fator=0,75`) dispara drift em `Insulin` (KS=0,586), `BMI` (KS=0,537) e `Glucose` (KS=0,462); candidato melhora recall sobre o campeão (`0,690` vs `0,655`, precisão `0,667`) e **aprova no portão**. | [Run #42431375747](https://github.com/wandraski/lab-06-11/actions/runs/42431375747) ([PR #3](https://github.com/wandraski/lab-06-11/pull/3)) |
| **E4** | **CT com portão reprovando e campeão preservado** (`cenario-5`): `class_weight: {0: 1, 1: 50}` gera recall `1,000` e precisão `0,388 < 0,65`; portão **reprova** e `models/champion/` permanece inalterado (SHA-256 preservado). PR fechado sem merge. | [Run #42431587974](https://github.com/wandraski/lab-06-11/actions/runs/42431587974) / [Run #42431381979](https://github.com/wandraski/lab-06-11/actions/runs/42431381979) ([PR #4](https://github.com/wandraski/lab-06-11/pull/4)) |
| **E5** | **CD verde com tag `v1.0.0`**: roda o CI (3 jobs), constrói a imagem Docker, executa o `smoke_test.py` e publica `ghcr.io/wandraski/lab-06-11:1.0.0` (visibilidade **Public**, baixada e validada localmente via HTTP). | [Run #42431435331](https://github.com/wandraski/lab-06-11/actions/runs/42431435331) ([Pacote GHCR](https://github.com/wandraski/lab-06-11/pkgs/container/lab-06-11)) |

## 5. Limites, riscos, contribuições e uso de IA

- **Limites e monitoramento em produção:** o portão offline avalia um recorte estático (150–170 linhas) e não garante calibração, equidade por subgrupos etários/gestacionais nem imunidade a atraso de rótulos (*delayed labels* de exames confirmatórios de HbA1c/glicemia de jejum). Em produção, monitorar: taxa de zeros/nulos por atributo, deriva (KS/PSI) nas 8 features e nas probabilidades previstas, calibração (Brier/ECE) por faixa etária, latência p95/erros 422 e métricas reais pós-confirmação laboratorial.
- **Contribuições e uso de IA:** atividade realizada individualmente por Wesley (`wandraski`), responsável integralmente pela engenharia de MLOps, adaptação do código e dados, validação cruzada, configuração de workflows/GHCR, execução das evidências E1–E5 e documentação. Assistentes de IA (**Codex** na adaptação inicial e **Antigravity / Gemini** na auditoria, validação cruzada sem vazamento, execução de CI/CT/CD e documentação) foram utilizados como suporte técnico, com todas as decisões e verificações validadas pelo autor.
