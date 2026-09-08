import numpy as np
import pandas as pd
from sklearn.base import clone

from src import config
from src.evaluation.metricas import avaliar, probabilidade_risco
from src.modeling.pipelines import candidatos
from src.modeling.treinar import escolher_modelo


def test_probabilidade_e_metricas_orientadas_para_classe_zero():
    class ModeloInvertido:
        classes_ = np.array([1, 0])
        def predict_proba(self, X):
            return np.array([[.1, .9], [.8, .2]])
    p = probabilidade_risco(ModeloInvertido(), None)
    np.testing.assert_allclose(p, [.9, .2])
    m = avaliar([0, 1], p)
    assert m["recall_risco"] == 1
    assert m["roc_auc"] == 1
    assert np.isclose(m["brier"], .025)


def test_preprocessamento_nao_aprende_categoria_do_teste():
    X = pd.DataFrame({c: [1., np.nan, 3., 4.] for c in config.NUMERICAS})
    for c in config.CATEGORICAS:
        X[c] = ["A", "B", "A", "B"]
    pipeline = candidatos()["logistica"].fit(X, [0, 1, 0, 1])
    teste = X.iloc[:1].copy()
    teste["sigla_uf"] = "NOVA"
    assert pipeline.predict_proba(teste).shape == (1, 2)
    encoder = pipeline.named_steps["preprocessar"].named_transformers_["categorias"].named_steps["codificar"]
    assert "NOVA" not in encoder.categories_[1]
    assert clone(pipeline).get_params()["modelo__max_iter"] == 1000


def test_selecao_preserva_regra_de_simplicidade():
    metricas = {"baseline": {"f1_macro": .37}, "logistica": {"f1_macro": .61}, "gradient_boosting": {"f1_macro": .613}}
    assert escolher_modelo(metricas) == "logistica"
    metricas["gradient_boosting"]["f1_macro"] = .63
    assert escolher_modelo(metricas) == "gradient_boosting"


def test_boosting_nao_faz_holdout_interno_por_aluno():
    assert candidatos()["gradient_boosting"].get_params()["modelo__early_stopping"] is False
