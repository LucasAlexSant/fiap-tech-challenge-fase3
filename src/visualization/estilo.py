"""Estilo comum dos gráficos exportáveis, sem dependência de interface gráfica."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL = "#176B87"
LARANJA = "#DC6B35"


def configurar():
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#263445"})


def salvar(fig, caminho):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, bbox_inches="tight", facecolor="white")
    plt.close(fig)
