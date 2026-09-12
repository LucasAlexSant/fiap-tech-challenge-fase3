"""O que separa territórios com alfabetização alta e baixa.

Três leituras independentes sobre a mesma unidade (município x rede), para não
depender de um único método: contribuição SHAP ao modelo congelado, queda de
desempenho por permutação e associação bruta/parcial com a taxa observada.
A correlação parcial remove o IDHM e o porte antes de medir a associação — é o
que separa "o território é pobre" de "a escola tem tal recurso".

Nada aqui é efeito causal. São associações em dados observacionais agregados,
sujeitas a confundimento e à falácia ecológica: o padrão vale para territórios,
não para uma criança.
"""
from datetime import datetime, timezone
import importlib.metadata
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src import config
from src.artefatos import salvar_json, sha256
from src.modeling.municipal import ALVO, HISTORICO, preparar
from src.modeling.pipelines import pre_processador
from src.modeling.zoologico import TEM_XGBOOST, estimadores
from src.preprocessing.enriquecida import FEATURES, NOVAS_NUMERICAS
from src.visualization.estilo import AZUL, LARANJA, configurar, plt, salvar

NUMERICAS = config.NUMERICAS + NOVAS_NUMERICAS

PILARES = {
    "Histórico educacional": HISTORICO,
    "Transferência de renda": ["total_pagamentos_bolsa_familia_anterior", "valor_total_bolsa_familia_anterior",
                               "valor_medio_pagamento_bolsa_familia_anterior"],
    "Desenvolvimento humano": ["idhm_municipio", "idhm_educacao_municipio",
                               "idhm_longevidade_municipio", "idhm_renda_municipio"],
    "Porte e escala da rede": ["populacao_municipio_ibge", "escolas_anos_iniciais_censo",
                               "matriculas_anos_iniciais_censo", "turmas_anos_iniciais_censo",
                               "vinculos_docentes_anos_iniciais_censo", "salas_utilizadas_censo",
                               "salas_utilizadas_fora_censo", "matriculas_por_vinculo_docente_censo",
                               "matriculas_por_turma_censo"],
    "Território e rede": ["rede", "sigla_uf", "regiao_brasil", "pct_escolas_rurais_censo"],
}


def pilar_de(coluna):
    for nome, colunas in PILARES.items():
        if coluna in colunas:
            return nome
    if "acessibilidade" in coluna:
        return "Acessibilidade"
    if "material_ped" in coluna:
        return "Material pedagógico"
    if "com_prof" in coluna or "com_professores" in coluna:
        return "Recursos humanos na escola"
    if any(t in coluna for t in ("agua", "esgoto", "energia", "internet", "computador",
                                 "climatizadas", "acessiveis")):
        return "Infraestrutura básica"
    if any(t in coluna for t in ("biblioteca", "sala_leitura", "quadra", "lab_informatica")):
        return "Espaços de aprendizagem"
    return "Outros"


def modelo_para_explicar(colunas):
    """Árvore explicável por TreeExplainer; SHAP exato, sem amostragem de fundo."""
    numericas = [c for c in NUMERICAS if c in colunas]
    catalogo = estimadores()
    nome = "xgboost" if TEM_XGBOOST else "hist_gb_profundo"
    return nome, Pipeline([("preprocessar", pre_processador(numericas)), ("modelo", catalogo[nome])])


def nome_original(nome, colunas):
    for c in sorted(colunas, key=len, reverse=True):
        if c in nome:
            return c
    raise ValueError(f"Feature transformada sem mapeamento: {nome}")


def finito(valor):
    """JSON do projeto proibe NaN; medidas degeneradas viram ausencia explicita."""
    if isinstance(valor, str) or valor is None:
        return valor
    return float(valor) if np.isfinite(valor) else None


def associacoes(unidades, colunas_numericas, controles):
    """Spearman bruto e parcial (resíduos dos controles) com a taxa observada."""
    y = stats.rankdata(unidades.taxa_alfabetizacao.to_numpy())
    C = unidades[controles].to_numpy(dtype=float)
    C = np.column_stack([stats.rankdata(np.nan_to_num(c, nan=np.nanmedian(c))) for c in C.T])
    C = np.column_stack([np.ones(len(C)), (C - C.mean(0)) / C.std(0)])
    y_res = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    baixo = unidades.taxa_alfabetizacao <= unidades.taxa_alfabetizacao.quantile(.2)
    alto = unidades.taxa_alfabetizacao >= unidades.taxa_alfabetizacao.quantile(.8)
    registros = []
    for coluna in colunas_numericas:
        x = unidades[coluna].to_numpy(dtype=float)
        valido = np.isfinite(x)
        if valido.sum() < 100 or np.nanstd(x[valido]) == 0:
            continue
        bruta = stats.spearmanr(x[valido], unidades.taxa_alfabetizacao.to_numpy()[valido]).statistic
        xr = stats.rankdata(np.nan_to_num(x, nan=np.nanmedian(x)))
        x_res = xr - C @ np.linalg.lstsq(C, xr, rcond=None)[0]
        parcial = float(np.corrcoef(x_res, y_res)[0, 1]) if x_res.std() > 0 else 0.
        a, b = x[alto.to_numpy() & valido], x[baixo.to_numpy() & valido]
        dp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else np.nan
        registros.append({k: finito(v) for k, v in {
            "variavel": coluna, "pilar": pilar_de(coluna),
            "spearman_bruto": bruta, "spearman_parcial": parcial, "cobertura": valido.mean(),
            "media_quintil_superior": a.mean() if len(a) else None,
            "media_quintil_inferior": b.mean() if len(b) else None,
            "diferenca_padronizada": (a.mean() - b.mean()) / dp if np.isfinite(dp) and dp > 0 else None,
        }.items()})
    tabela = pd.DataFrame(registros)
    return tabela.sort_values("spearman_parcial", key=lambda c: c.abs().fillna(0), ascending=False)


def executar_insights(lake, execucao, ano_treino=2024, ano_teste=2025, threads=4, repeticoes=10):
    import shap
    config.preparar_diretorios()
    configurar()
    inicio = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_insights"
    desenvolvimento, prov_dev = preparar(lake, ano_treino, execucao)
    teste, prov_teste = preparar(lake, ano_teste, execucao)
    nome_modelo, pipeline = modelo_para_explicar(FEATURES)
    Xdev, ydev = desenvolvimento[FEATURES], desenvolvimento[ALVO].to_numpy()
    Xte, yte = teste[FEATURES], teste[ALVO].to_numpy()
    print(f"[INSIGHTS] {nome_modelo} em {len(Xdev)} territorios de {ano_treino}; "
          f"explicando {len(Xte)} de {ano_teste}", flush=True)
    saida = {"run_id": run_id, "modelo_explicado": nome_modelo, "ano_treino": ano_treino,
             "ano_teste": ano_teste, "unidade": "id_municipio x rede", "execucao_gold": execucao,
             "proveniencia_treino": prov_dev["fontes"], "proveniencia_teste": prov_teste["fontes"],
             "corte_desenvolvimento": prov_dev["corte_taxa"], "corte_teste": prov_teste["corte_taxa"],
             "shap_versao": importlib.metadata.version("shap"),
             "shap_escala": "log-odds do risco territorial (taxa abaixo da mediana)"}
    with threadpool_limits(limits=threads):
        modelo = clone(pipeline).fit(Xdev, ydev)
        transformador, estimador = modelo.named_steps["preprocessar"], modelo.named_steps["modelo"]
        Z = transformador.transform(Xte)
        nomes = transformador.get_feature_names_out()
        explicacao = shap.TreeExplainer(estimador, model_output="raw")(Z)
        valores = np.asarray(explicacao.values)
        if valores.ndim != 2:
            raise ValueError("Formato SHAP inesperado para classificacao binaria")
        margem = np.asarray(explicacao.base_values) + valores.sum(axis=1)
        esperada = estimador.predict_proba(Z)[:, list(estimador.classes_).index(1)]
        erro = float(np.max(np.abs(1 / (1 + np.exp(-margem)) - esperada)))
        if erro > 1e-4:
            raise ValueError(f"Contribuicoes SHAP nao reproduzem a probabilidade: erro={erro}")
        saida["erro_maximo_reconstrucao_probabilidade"] = erro
        agrupado = np.zeros((len(Z), len(FEATURES)))
        for i, nome in enumerate(nomes):
            agrupado[:, FEATURES.index(nome_original(nome, FEATURES))] += valores[:, i]
        shap_global = pd.DataFrame({
            "variavel": FEATURES, "pilar": [pilar_de(c) for c in FEATURES],
            "shap_absoluto_medio": np.abs(agrupado).mean(axis=0),
            "shap_medio_com_sinal": agrupado.mean(axis=0)}).sort_values("shap_absoluto_medio", ascending=False)
        saida["shap"] = shap_global.to_dict("records")
        por_pilar = shap_global.groupby("pilar").agg(
            variaveis=("variavel", "size"), shap_total=("shap_absoluto_medio", "sum"),
            shap_maximo=("shap_absoluto_medio", "max")).sort_values("shap_total", ascending=False)
        por_pilar["participacao_pct"] = 100 * por_pilar.shap_total / por_pilar.shap_total.sum()
        saida["shap_por_pilar"] = por_pilar.reset_index().to_dict("records")
        print(f"[INSIGHTS] SHAP validado (erro {erro:.2g}); permutacao em {repeticoes} repeticoes", flush=True)
        perm = permutation_importance(modelo, Xte, yte, scoring="accuracy", n_repeats=repeticoes,
                                      n_jobs=1, random_state=config.SEMENTE)
        permutacao = pd.DataFrame({"variavel": FEATURES, "pilar": [pilar_de(c) for c in FEATURES],
                                   "queda_acuracia": perm.importances_mean,
                                   "desvio": perm.importances_std}).sort_values("queda_acuracia", ascending=False)
        saida["permutacao"] = permutacao.to_dict("records")
    juntas = pd.concat([desenvolvimento.assign(ano=ano_treino), teste.assign(ano=ano_teste)], ignore_index=True)
    controles = ["idhm_municipio", "populacao_municipio_ibge"]
    assoc = associacoes(juntas, [c for c in NUMERICAS if c not in controles], controles)
    saida["associacoes"] = assoc.to_dict("records")
    saida["controles_da_parcial"] = controles
    saida["n_territorios_associacao"] = int(len(juntas))
    contraste_categorico = []
    for coluna in ["rede", "regiao_brasil"]:
        g = juntas.groupby(coluna, observed=True).agg(
            territorios=(ALVO, "size"), taxa_media=("taxa_alfabetizacao", "mean"),
            pct_em_risco=(ALVO, "mean")).reset_index().rename(columns={coluna: "grupo"})
        contraste_categorico += g.assign(variavel=coluna).to_dict("records")
    saida["contrastes_categoricos"] = contraste_categorico
    graficos(shap_global, por_pilar, assoc, juntas)
    saida["codigo_sha256"] = {str(p.relative_to(config.RAIZ)): sha256(p)
                              for p in sorted((config.RAIZ / "src").rglob("*.py"))}
    saida["duracao_segundos"] = round(time.monotonic() - inicio, 1)
    saida["limites"] = [
        "Associacao agregada por territorio; nao vale para individuos (falacia ecologica).",
        "SHAP e permutacao explicam o modelo, nao o mundo; variaveis correlacionadas dividem credito.",
        "A parcial controla IDHM e populacao, nao todo o confundimento.",
        "Censo e IDHM sao do ano anterior e do ultimo levantamento disponivel, respectivamente.",
    ]
    salvar_json(config.RELATORIOS / "insights_alfabetizacao.json", saida)
    salvar_json(config.RELATORIOS / "runs" / run_id / "insights.json", saida)
    return saida


def graficos(shap_global, por_pilar, assoc, juntas):
    topo = shap_global.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(topo.variavel, topo.shap_absoluto_medio, color=LARANJA)
    ax.set(xlabel="Contribuicao SHAP absoluta media (log-odds do risco territorial)",
           title="O que o modelo territorial usa")
    salvar(fig, config.IMAGENS / "insights_shap_variaveis.png")

    p = por_pilar.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(p.index, p.participacao_pct, color=AZUL)
    for i, (valor, n) in enumerate(zip(p.participacao_pct, p.variaveis)):
        ax.text(valor + .4, i, f"{valor:.1f}%  ({n} var.)", va="center", fontsize=8)
    ax.set(xlim=(0, p.participacao_pct.max() * 1.35), xlabel="Participacao na contribuicao SHAP total (%)",
           title="Peso por pilar tematico")
    salvar(fig, config.IMAGENS / "insights_shap_pilares.png")

    top = assoc.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    y = np.arange(len(top))
    ax.barh(y + .2, top.spearman_bruto, height=.38, color="#B9CBD3", label="Bruta")
    ax.barh(y - .2, top.spearman_parcial, height=.38, color=AZUL, label="Parcial (sem IDHM e porte)")
    ax.set_yticks(y, top.variavel)
    ax.axvline(0, color="gray", linewidth=.8)
    ax.set(xlabel="Correlacao de Spearman com a taxa de alfabetizacao do territorio",
           title="Quanto sobra depois de descontar desenvolvimento humano e porte")
    ax.legend(loc="lower right", fontsize=8)
    salvar(fig, config.IMAGENS / "insights_associacoes.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, coluna, rotulo in zip(axes, ["idhm_educacao_municipio", "taxa_alfabetizacao_municipio_anterior",
                                         "pct_escolas_rurais_censo"],
                                  ["IDHM educacao", "Taxa municipal anterior", "% escolas rurais"]):
        d = juntas[[coluna, "taxa_alfabetizacao"]].dropna()
        faixas = pd.qcut(d[coluna], 10, duplicates="drop")
        m = d.groupby(faixas, observed=True).agg(x=(coluna, "mean"), y=("taxa_alfabetizacao", "mean"))
        ax.plot(m.x, m.y, "o-", color=LARANJA)
        ax.set(xlabel=rotulo, ylabel="Taxa de alfabetizacao" if ax is axes[0] else None)
        ax.grid(alpha=.25)
    fig.suptitle("Relacao por decis — territorios, nao criancas", fontweight="bold")
    salvar(fig, config.IMAGENS / "insights_decis.png")
