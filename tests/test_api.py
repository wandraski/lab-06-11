"""Testes de contrato da API: formato de entrada e de saída do /predict."""

import pytest
from fastapi.testclient import TestClient

from mlops_demo import config
from mlops_demo.api import app

pytestmark = pytest.mark.api

cliente = TestClient(app)


def test_health_responde_ok():
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_predict_devolve_o_formato_combinado(base_holdout):
    exame = base_holdout[config.FEATURES].iloc[0].tolist()
    resposta = cliente.post("/predict", json={"atributos": exame})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"classe", "probabilidade_diabetes", "versao_codigo"}
    assert corpo["classe"] in {"diabetico", "nao_diabetico"}
    assert 0.0 <= corpo["probabilidade_diabetes"] <= 1.0


def test_predict_rejeita_quantidade_errada_de_atributos():
    resposta = cliente.post("/predict", json={"atributos": [1.0, 2.0]})
    assert resposta.status_code == 422


def test_predict_rejeita_sete_atributos(base_holdout):
    exame = base_holdout[config.FEATURES].iloc[0].tolist()[:-1]
    assert cliente.post("/predict", json={"atributos": exame}).status_code == 422


def test_predict_rejeita_glicose_zero(base_holdout):
    exame = base_holdout[config.FEATURES].iloc[0].tolist()
    exame[config.FEATURES.index("Glucose")] = 0
    assert cliente.post("/predict", json={"atributos": exame}).status_code == 422


def test_predict_acerta_um_caso_diabetico_e_um_nao_diabetico_claros(base_holdout, campeao):
    """Escolhe os casos mais claros (maior confiança do campeão) de cada classe."""
    X = base_holdout[config.FEATURES]
    coluna = list(campeao.classes_).index(config.CLASSE_DIABETICA)
    p_diabetico = campeao.predict_proba(X)[:, coluna]
    i_diabetico = int(p_diabetico.argmax())
    i_nao_diabetico = int(p_diabetico.argmin())
    for indice, esperado in ((i_diabetico, "diabetico"), (i_nao_diabetico, "nao_diabetico")):
        resposta = cliente.post("/predict", json={"atributos": X.iloc[indice].tolist()})
        assert resposta.json()["classe"] == esperado
