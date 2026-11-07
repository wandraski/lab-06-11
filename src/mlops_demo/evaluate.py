"""Métricas de avaliação. A classe de interesse é a DIABÉTICA (rótulo 1)."""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from mlops_demo.config import CLASSE_DIABETICA
from mlops_demo.features import separar_xy


def calcular_metricas(y_real, y_previsto) -> dict:
    """Recall e precisão da classe diabetica, mais a acurácia como referência."""
    return {
        "recall_diabetico": float(recall_score(y_real, y_previsto, pos_label=CLASSE_DIABETICA)),
        "precisao_diabetica": float(
            precision_score(y_real, y_previsto, pos_label=CLASSE_DIABETICA, zero_division=0)
        ),
        "acuracia": float(accuracy_score(y_real, y_previsto)),
        "n_amostras": int(len(y_real)),
    }


def avaliar(modelo, df: pd.DataFrame) -> dict:
    """Avalia um modelo treinado em um DataFrame com as colunas do projeto."""
    X, y = separar_xy(df)
    return calcular_metricas(y, modelo.predict(X))
