"""API de inferência (FastAPI) servindo o modelo campeão.

Local:   uvicorn mlops_demo.api:app --reload
Docker:  docker run -p 8000:8000 mlops-demo
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlops_demo import __version__, config
from mlops_demo.data import validar_dataframe

CAMINHO_MODELO = Path(os.environ.get("MODEL_PATH", config.MODELO_CAMPEAO))
CAMINHO_METRICAS = CAMINHO_MODELO.with_name("metrics.json")

app = FastAPI(title="Triagem de diabetes (demo CI/CD/CT)", version=__version__)
_modelo = joblib.load(CAMINHO_MODELO)


class Exame(BaseModel):
    """Os atributos do exame, na mesma ordem do treino."""

    atributos: list[float] = Field(min_length=len(config.FEATURES), max_length=len(config.FEATURES))


class Predicao(BaseModel):
    classe: str
    probabilidade_diabetes: float
    versao_codigo: str


@app.get("/health")
def saude() -> dict:
    metricas = {}
    if CAMINHO_METRICAS.exists():
        metricas = json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))
    return {"status": "ok", "versao_codigo": __version__, "metricas_do_campeao": metricas}


@app.post("/predict", response_model=Predicao)
def prever(exame: Exame) -> Predicao:
    entrada = pd.DataFrame([exame.atributos], columns=config.FEATURES)
    entrada[config.ALVO] = 0
    problemas = validar_dataframe(entrada)
    if problemas:
        raise HTTPException(status_code=422, detail=problemas)
    entrada = entrada[config.FEATURES]
    previsto = int(_modelo.predict(entrada)[0])
    # A posição da classe positiva depende da ordem em classes_.
    coluna = list(_modelo.classes_).index(config.CLASSE_DIABETICA)
    p_diabetico = float(_modelo.predict_proba(entrada)[0][coluna])
    return Predicao(
        classe="diabetico" if previsto == config.CLASSE_DIABETICA else "nao_diabetico",
        probabilidade_diabetes=round(p_diabetico, 4),
        versao_codigo=__version__,
    )
