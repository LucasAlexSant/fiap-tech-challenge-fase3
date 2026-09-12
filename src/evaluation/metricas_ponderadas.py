"""Métricas por avaliação calculadas sobre unidades com peso, sem expandir a base."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)


def avaliar_ponderado(y, p_risco, peso, limiar=.5):
    """Risco é a classe 0. `peso` é a contagem de avaliações de cada par (X, y)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p_risco, dtype=float)
    w = np.asarray(peso, dtype=float)
    if not (len(y) == len(p) == len(w)) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilidades ou pesos inválidos")
    previsto = np.where(p >= limiar, 0, 1)
    risco = (y == 0).astype(int)
    duas = len(np.unique(y[w > 0])) == 2
    return {
        "n": float(w.sum()), "limiar": float(limiar),
        "prevalencia_risco": float(np.average(risco, weights=w)),
        "acuracia": accuracy_score(y, previsto, sample_weight=w),
        "acuracia_balanceada": balanced_accuracy_score(y, previsto, sample_weight=w) if duas else None,
        "f1_macro": f1_score(y, previsto, labels=[0, 1], average="macro", zero_division=0, sample_weight=w),
        "recall_risco": recall_score(y, previsto, pos_label=0, zero_division=0, sample_weight=w),
        "precisao_risco": precision_score(y, previsto, pos_label=0, zero_division=0, sample_weight=w),
        "recall_alfabetizado": recall_score(y, previsto, pos_label=1, zero_division=0, sample_weight=w),
        "roc_auc": roc_auc_score(risco, p, sample_weight=w) if duas else None,
        "average_precision_risco": average_precision_score(risco, p, sample_weight=w) if duas else None,
        "brier": brier_score_loss(risco, p, sample_weight=w),
        "log_loss": log_loss(y, np.column_stack([p, 1 - p]), labels=[0, 1], sample_weight=w),
        "matriz_confusao_labels_0_1": confusion_matrix(y, previsto, labels=[0, 1], sample_weight=w).tolist(),
    }


def limiar_de_maior_acuracia(y, p_risco, peso, grade=None):
    """Escolhe o limiar apenas onde for permitido olhar (validação), nunca no teste."""
    grade = np.linspace(.05, .95, 181) if grade is None else np.asarray(grade)
    y = np.asarray(y, dtype=int)
    acuracias = [accuracy_score(y, np.where(p_risco >= t, 0, 1), sample_weight=peso) for t in grade]
    melhor = int(np.argmax(acuracias))
    return float(grade[melhor]), float(acuracias[melhor])
