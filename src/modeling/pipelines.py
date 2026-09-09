"""Pré-processamento aprendido dentro de cada dobra e persistido com o modelo."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


def pre_processador(numericas=None):
    numericas = config.NUMERICAS if numericas is None else numericas
    taxas = [c for c in numericas if c.startswith(("taxa_", "pct_"))]
    volumes = [c for c in numericas if c not in taxas]
    numerico = Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("escalar", StandardScaler()),
    ])
    assimetricos = Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("escalar", StandardScaler()),
    ])
    categorico = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="Ausente", keep_empty_features=True)),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)),
    ])
    return ColumnTransformer([
        ("taxas", numerico, taxas), ("volumes", assimetricos, volumes),
        ("categorias", categorico, config.CATEGORICAS),
    ], remainder="drop", sparse_threshold=0)


def candidatos(numericas=None):
    estimadores = {
        "baseline": DummyClassifier(strategy="prior", random_state=config.SEMENTE),
        "logistica": LogisticRegression(max_iter=1000, random_state=config.SEMENTE),
        # Sem early stopping interno: ele faria divisão aleatória por linha,
        # compartilhando escolas entre ajuste e sua validação interna.
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, min_samples_leaf=100,
            l2_regularization=10., early_stopping=False, random_state=config.SEMENTE),
    }
    return {nome: Pipeline([("preprocessar", pre_processador(numericas)), ("modelo", modelo)])
            for nome, modelo in estimadores.items()}


GRADES = {
    # O custo do falso negativo (aluno não alfabetizado classificado como alfabetizado)
    # é maior. Além do balanceamento automático, testamos pesos explícitos moderados.
    "logistica": {"modelo__C": [.1, 1.], "modelo__class_weight": [None, "balanced", {0: 2, 1: 1}]},
    "gradient_boosting": {"modelo__max_leaf_nodes": [15, 31], "modelo__min_samples_leaf": [100, 300]},
}
