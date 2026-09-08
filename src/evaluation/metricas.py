"""Risco é classe 0 (não alfabetizado); probabilidades seguem essa orientação."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)


def probabilidade_risco(modelo, X):
    classes = list(modelo.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("Modelo deve conhecer as duas classes binárias")
    return modelo.predict_proba(X)[:, classes.index(0)]


def avaliar(y, p_risco):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p_risco, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilidades inválidas")
    previsto = np.where(p >= .5, 0, 1)
    risco = (y == 0).astype(int)
    duas_classes = len(np.unique(y)) == 2
    return {
        "n": len(y), "prevalencia_risco": float(risco.mean()),
        "acuracia": accuracy_score(y, previsto),
        "acuracia_balanceada": balanced_accuracy_score(y, previsto) if duas_classes else None,
        "f1_macro": f1_score(y, previsto, labels=[0, 1], average="macro", zero_division=0),
        "recall_risco": recall_score(y, previsto, pos_label=0, zero_division=0),
        "precisao_risco": precision_score(y, previsto, pos_label=0, zero_division=0),
        "recall_alfabetizado": recall_score(y, previsto, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(risco, p) if duas_classes else None,
        "average_precision_risco": average_precision_score(risco, p) if duas_classes else None,
        "brier": brier_score_loss(risco, p),
        "log_loss": log_loss(y, np.column_stack([p, 1 - p]), labels=[0, 1]),
        "matriz_confusao_labels_0_1": confusion_matrix(y, previsto, labels=[0, 1]).tolist(),
    }


def por_grupo(base_teste, p_risco, coluna, alvo):
    registros = []
    for valor, posicoes in base_teste.reset_index(drop=True).groupby(coluna, dropna=False).indices.items():
        metricas = avaliar(base_teste.iloc[posicoes][alvo], np.asarray(p_risco)[posicoes])
        metricas.pop("matriz_confusao_labels_0_1")
        registros.append({"grupo": str(valor), "amostra_pequena": len(posicoes) < 100, **metricas})
    return registros


def bootstrap_escolas(y, p_risco, grupos, repeticoes=300, semente=42):
    """IC percentil por reamostragem de escolas, não de alunos independentes."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p_risco)
    prev = np.where(p >= .5, 0, 1)
    codigos, unicos = pd.factorize(grupos)
    n = len(unicos)
    contagens = np.column_stack([np.bincount(codigos, weights=mask.astype(float), minlength=n)
                               for mask in [(y == 0) & (prev == 0), (y == 0) & (prev == 1),
                                            (y == 1) & (prev == 0), (y == 1) & (prev == 1)]])
    rng = np.random.default_rng(semente)
    f1s, recalls = [], []
    for _ in range(repeticoes):
        a, b, c, d = contagens[rng.integers(0, n, size=n)].sum(axis=0)
        f1s.append(.5 * (2 * a / max(2*a+b+c, 1) + 2*d / max(2*d+b+c, 1)))
        recalls.append(a / max(a+b, 1))
    return {"metodo": "bootstrap de escolas, percentis 2.5/97.5; modelo fixo",
            "repeticoes": repeticoes, "escolas": n,
            "f1_macro_ic95": np.quantile(f1s, [.025, .975]).tolist(),
            "recall_risco_ic95": np.quantile(recalls, [.025, .975]).tolist()}
