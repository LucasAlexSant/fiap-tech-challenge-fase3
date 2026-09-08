import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src import config
from src.visualization.estilo import configurar, plt, salvar


def graficos_avaliacao(y, probas, escolhido):
    configurar()
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y, np.where(probas[escolhido] >= .5, 0, 1),
        labels=[0, 1], display_labels=["Não alfabetizado", "Alfabetizado"], ax=ax, cmap="Blues", colorbar=False)
    ax.set(title=f"Matriz de confusão — {escolhido}", xlabel="Classe prevista", ylabel="Classe observada")
    salvar(fig, config.IMAGENS / "teste_confusao.png")
    fig, eixos = plt.subplots(1, 2, figsize=(12, 5))
    for nome, p in probas.items():
        RocCurveDisplay.from_predictions(y == 0, p, name=nome, ax=eixos[0])
        PrecisionRecallDisplay.from_predictions(y == 0, p, name=nome, ax=eixos[1])
    eixos[0].plot([0, 1], [0, 1], "--", color="gray")
    eixos[0].set(title="ROC — risco de não alfabetização", xlabel="Taxa de falsos positivos", ylabel="Recall do risco")
    eixos[1].axhline(np.mean(y == 0), linestyle="--", color="gray")
    eixos[1].set(title="Precisão × recall — risco", xlabel="Recall do risco", ylabel="Precisão do risco")
    salvar(fig, config.IMAGENS / "teste_curvas.png")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibração ideal")
    for nome, p in probas.items():
        observada, prevista = calibration_curve(y == 0, p, n_bins=10, strategy="quantile")
        ax.plot(prevista, observada, marker="o", label=nome)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Probabilidade prevista de não alfabetização",
           ylabel="Proporção observada", title="Calibração no teste — intervalos por quantis")
    ax.legend()
    salvar(fig, config.IMAGENS / "teste_calibracao.png")
