"""EDA da Gold enriquecida usada no Tech Challenge 3."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.preprocessing.enriquecida import FEATURES, NOVAS_NUMERICAS, carregar_enriquecida
from src.visualization.estilo import AZUL, LARANJA, configurar, plt, salvar


def ultima_execucao(lake: Path) -> str:
    pastas = sorted((lake / "gold" / "base_modelagem_aluno_enriquecida").glob("execution_date=*"))
    if not pastas:
        raise FileNotFoundError("Gold enriquecida não encontrada; execute a Fase 2 primeiro")
    return pastas[-1].name.split("=", 1)[1]


def executar_eda_enriquecida(lake: Path = config.LAKE, ano: int = 2024, execucao: str | None = None):
    execucao = execucao or ultima_execucao(lake)
    base, origem = carregar_enriquecida(lake, ano, execucao)
    configurar()
    treino = base.loc[base.elegivel_modelagem].copy()
    ausencias = treino[FEATURES].isna().mean().mul(100).sort_values()
    resumo = {
        "escopo": "Gold enriquecida; avaliações elegíveis da edição de treino",
        "ano": ano, "execution_date": execucao, "linhas": len(treino),
        "features": FEATURES, "novas_features": NOVAS_NUMERICAS,
        "ausencias_pct": ausencias.to_dict(),
        "classes": treino[config.ALVO].value_counts().to_dict(),
        "cobertura_gold": origem["manifesto_gold"]["cobertura"].get(str(ano), {}),
    }
    (config.RELATORIOS / "eda_enriquecida_resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(ausencias.index, ausencias, color=AZUL)
    ax.set(xlabel="Ausência (%)", title=f"Gold enriquecida — cobertura dos preditores ({ano})")
    salvar(fig, config.IMAGENS / "eda_enriquecida_ausencias.png")

    selecionadas = [c for c in ["taxa_alfabetizacao_municipio_anterior", "taxa_presenca_municipio_anterior",
                                "populacao_municipio_ibge", "idhm_municipio", "idhm_educacao_municipio",
                                "idhm_longevidade_municipio", "idhm_renda_municipio"] if c in treino]
    fig, eixos = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    for ax, coluna in zip(eixos.flat, selecionadas):
        valores = pd.to_numeric(treino[coluna], errors="coerce").dropna()
        if not valores.empty:
            ax.hist(valores[valores.le(valores.quantile(.99))], bins=35, color=AZUL)
        ax.set_title(coluna, fontsize=9)
    for ax in eixos.flat[len(selecionadas):]:
        ax.axis("off")
    fig.suptitle(f"Distribuições das variáveis enriquecidas ({ano})")
    salvar(fig, config.IMAGENS / "eda_enriquecida_distribuicoes.png")

    numericas = [c for c in FEATURES if c not in config.CATEGORICAS and c in treino]
    municipal = treino.groupby("id_municipio").agg(
        **{c: (c, "first") for c in numericas}, taxa_observada=(config.ALVO, "mean"))
    correlacao = municipal.corr(method="spearman")["taxa_observada"].drop("taxa_observada").abs().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(correlacao.index[::-1], correlacao.values[::-1], color=AZUL)
    ax.set(xlabel="Correlação de Spearman absoluta", title="Gold enriquecida — associações municipais com alfabetização")
    salvar(fig, config.IMAGENS / "eda_enriquecida_correlacoes.png")

    por_rede = treino.groupby("rede")[config.ALVO].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(por_rede.index, (1 - por_rede) * 100, color=LARANJA)
    ax.set(ylabel="Não alfabetizados (%)", title=f"Gold enriquecida — risco por rede ({ano})")
    salvar(fig, config.IMAGENS / "eda_enriquecida_rede.png")
    print(f"[EDA ENRIQUECIDA] {len(treino):,} avaliações; 4 gráficos e relatório gerados")
