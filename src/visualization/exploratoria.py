"""EDA de relações com o alvo apenas no treino; teste reservado para avaliação."""
import pandas as pd

from src import config
from src.artefatos import salvar_json
from src.preprocessing.base import carregar_base
from src.preprocessing.divisao import dividir, resumo_divisao
from src.visualization.estilo import AZUL, LARANJA, configurar, plt, salvar

ROTULOS = {
    "taxa_alfabetizacao_municipio_anterior": "Alfabetização municipal anterior (%)",
    "taxa_presenca_municipio_anterior": "Presença na avaliação anterior (%)",
    "alunos_avaliados_municipio_anterior": "Avaliações municipais anteriores",
    "total_pagamentos_bolsa_familia_anterior": "Pagamentos Bolsa Família anteriores",
    "valor_total_bolsa_familia_anterior": "Valor total Bolsa Família anterior (R$)",
    "valor_medio_pagamento_bolsa_familia_anterior": "Valor médio por pagamento (R$)",
    "rede": "Rede", "sigla_uf": "UF", "regiao_brasil": "Região",
}


def executar_eda():
    configurar()
    base, manifesto = carregar_base()
    partes = dividir(base)
    resumo_divisao(base, partes, manifesto["base_sha256"])
    treino = base.iloc[partes["treino"]]
    ausencias = treino[config.FEATURES].isna().mean().mul(100)
    resumo = {"escopo": "treino; validação e teste não orientam hipóteses",
              "linhas": len(treino), "nao_alfabetizados": int(treino[config.ALVO].eq(0).sum()),
              "ausencias_pct": ausencias.to_dict(),
              "numericas": treino[config.NUMERICAS].describe().round(3).to_dict(),
              "redes": treino.rede.value_counts().to_dict()}
    salvar_json(config.RELATORIOS / "eda_resumo.json", resumo)
    fig, ax = plt.subplots(figsize=(7, 4))
    contagem = treino[config.ALVO].value_counts().reindex([0, 1], fill_value=0)
    ax.bar(["Não alfabetizado", "Alfabetizado"], contagem, color=[LARANJA, AZUL])
    for i, v in enumerate(contagem):
        ax.text(i, v, f"{v:,}\n{v / len(treino):.1%}", ha="center", va="bottom")
    ax.set(ylim=(0, contagem.max() * 1.2), ylabel="Avaliações válidas", title="Distribuição do alvo no treino")
    salvar(fig, config.IMAGENS / "eda_classes.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    a = ausencias.sort_values()
    ax.barh([ROTULOS[c] for c in a.index], a, color=AZUL)
    ax.set(xlabel="Ausência (%)", xlim=(0, max(30, a.max() * 1.15)), title="Cobertura dos preditores no treino")
    salvar(fig, config.IMAGENS / "eda_ausencias.png")
    por_uf = treino.groupby("sigla_uf")[config.ALVO].agg(["size", "mean"]).sort_values("mean")
    salvar_json(config.RELATORIOS / "eda_uf.json", por_uf.reset_index().to_dict("records"))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(por_uf.index, 100 * (1 - por_uf["mean"]), color=LARANJA)
    ax.set(ylabel="Não alfabetizados (%)", title="Diferenças entre UFs no treino — taxa não ponderada")
    salvar(fig, config.IMAGENS / "eda_uf.png")
    fig, eixos = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, coluna in zip(eixos.flat, config.NUMERICAS):
        s = treino[coluna].dropna()
        if not s.empty:
            limite = s.quantile(.99)
            ax.hist(s[s.le(limite)], bins=35, color=AZUL)
        ax.set(title=ROTULOS[coluna], ylabel="Avaliações")
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle("Distribuições no treino — até P99 apenas para visualização", fontsize=14)
    salvar(fig, config.IMAGENS / "eda_distribuicoes.png")
    # Correlação territorial evita contar o mesmo contexto municipal milhares de vezes.
    municipios = treino.groupby("id_municipio").agg(
        **{c: (c, "first") for c in config.NUMERICAS}, taxa_observada=(config.ALVO, "mean"))
    correlacao = municipios.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7))
    m = ax.imshow(correlacao, cmap="RdBu", vmin=-1, vmax=1)
    nomes = [ROTULOS.get(c, "Alfabetização observada no treino") for c in correlacao.columns]
    ax.set_xticks(range(len(nomes)), nomes, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(nomes)), nomes, fontsize=8)
    for i in range(len(nomes)):
        for j in range(len(nomes)):
            ax.text(j, i, f"{correlacao.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(m, ax=ax, label="Spearman")
    ax.set_title("Correlação por município no treino — associação, não causalidade")
    salvar(fig, config.IMAGENS / "eda_correlacoes.png")
    print(f"[EDA] {len(treino):,} avaliações de treino; 5 gráficos e relatórios gerados")
