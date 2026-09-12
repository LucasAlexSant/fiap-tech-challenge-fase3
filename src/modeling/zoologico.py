"""Catálogo amplo de algoritmos, com o mesmo pré-processamento de referência.

Cada candidato é um Pipeline completo: imputação, log1p em volumes, padronização
e one-hot são aprendidos dentro de cada ajuste, nunca antes da divisão.
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src import config
from src.modeling.pipelines import pre_processador

try:  # XGBoost é opcional; sua ausência não invalida o benchmark.
    from xgboost import XGBClassifier
    TEM_XGBOOST = True
except ImportError:  # pragma: no cover - depende do ambiente
    TEM_XGBOOST = False


def estimadores(semente=config.SEMENTE):
    """Nome -> estimador. Sementes fixas; nenhum usa holdout interno aleatório."""
    catalogo = {
        "baseline_prior": DummyClassifier(strategy="prior", random_state=semente),
        "logistica": LogisticRegression(max_iter=2000, C=1., random_state=semente),
        "logistica_balanceada": LogisticRegression(max_iter=2000, C=1., class_weight="balanced", random_state=semente),
        # Sem elasticnet: `saga` e o unico solver que a implementa e nao converge em
        # 10.000 iteracoes nas dobras da validacao cruzada. Colunas do Censo quase
        # constantes dentro de uma dobra ficam mal condicionadas depois da
        # padronizacao, e o passo do solver desaba. Um ajuste interrompido nao seria
        # comparavel aos demais; a familia linear continua representada pela
        # logistica L2, pela versao balanceada e pelo discriminante linear.
        "discriminante_linear": LinearDiscriminantAnalysis(),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features="sqrt",
                                                n_jobs=1, random_state=semente),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, max_features="sqrt",
                                            n_jobs=1, random_state=semente),
        # early_stopping=False: um holdout interno aleatório misturaria municípios
        # entre ajuste e parada, e o peso amostral tornaria a parada dependente do volume.
        "hist_gb_raso": HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
                                                       learning_rate=.08, l2_regularization=10.,
                                                       early_stopping=False, random_state=semente),
        "hist_gb_profundo": HistGradientBoostingClassifier(max_iter=500, max_leaf_nodes=63, min_samples_leaf=10,
                                                           learning_rate=.05, l2_regularization=1.,
                                                           early_stopping=False, random_state=semente),
        # early_stopping=False mantem o holdout fora do estimador; o orcamento de
        # iteracoes e generoso para que o Adam nao pare antes de estabilizar.
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=1500,
                             n_iter_no_change=15, early_stopping=False, random_state=semente),
    }
    if TEM_XGBOOST:
        catalogo["xgboost"] = XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.8,
            reg_lambda=2., min_child_weight=5, tree_method="hist", n_jobs=1,
            eval_metric="logloss", random_state=semente)
    return catalogo


def candidatos_amplos(numericas, semente=config.SEMENTE):
    return {nome: Pipeline([("preprocessar", pre_processador(numericas)), ("modelo", est)])
            for nome, est in estimadores(semente).items()}


def empilhamento(numericas, semente=config.SEMENTE):
    """Combina famílias diferentes: linear, floresta e boosting, com meta-logística."""
    base = estimadores(semente)
    membros = [(n, base[n]) for n in ["logistica", "random_forest", "hist_gb_profundo"] if n in base]
    if TEM_XGBOOST:
        membros.append(("xgboost", base["xgboost"]))
    return Pipeline([("preprocessar", pre_processador(numericas)),
                     ("modelo", StackingClassifier(membros, final_estimator=LogisticRegression(max_iter=2000),
                                                   cv=3, n_jobs=1, passthrough=False))])


GRADES_AMPLAS = {
    "logistica": {"modelo__C": [.03, .1, .3, 1., 3.]},
    "logistica_balanceada": {"modelo__C": [.03, .1, .3, 1., 3.]},
    "hist_gb_raso": {"modelo__max_leaf_nodes": [7, 15, 31], "modelo__learning_rate": [.05, .1]},
    "hist_gb_profundo": {"modelo__max_leaf_nodes": [31, 63], "modelo__min_samples_leaf": [5, 20],
                         "modelo__l2_regularization": [1., 10.]},
    "random_forest": {"modelo__min_samples_leaf": [1, 5, 20], "modelo__max_features": ["sqrt", .5]},
    "extra_trees": {"modelo__min_samples_leaf": [1, 5, 20], "modelo__max_features": ["sqrt", .5]},
    "xgboost": {"modelo__max_depth": [4, 6, 8], "modelo__learning_rate": [.03, .08],
                "modelo__min_child_weight": [1, 5]},
    "mlp": {"modelo__alpha": [1e-4, 1e-2]},
}
