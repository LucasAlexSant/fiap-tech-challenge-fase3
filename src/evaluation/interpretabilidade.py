"""Explica o modelo congelado; os resultados não orientam nova seleção no teste."""
import argparse
import importlib.metadata

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

from src import config
from src.artefatos import ler_json, salvar_json, sha256
from src.evaluation.metricas import probabilidade_risco
from src.preprocessing.base import carregar_base
from src.preprocessing.divisao import dividir
from src.visualization.estilo import AZUL, LARANJA, configurar, plt, salvar
from src.visualization.exploratoria import ROTULOS


def validar_artefatos(bundle, resultado, manifesto):
    if bundle["run_id"] != resultado["run_id"]:
        raise ValueError("Modelo e métricas são de execuções diferentes")
    if bundle["base_sha256"] != manifesto["base_sha256"]:
        raise ValueError("Modelo treinado em outra base; execute treinar novamente")
    if bundle["features"] != config.FEATURES:
        raise ValueError("Contrato de preditores mudou")
    if sha256(config.MODELOS / "modelo_alfabetizacao.joblib") != resultado["modelo_sha256"]:
        raise ValueError("Arquivo do modelo diverge do hash registrado")


def nome_original(nome):
    for c in sorted(config.FEATURES, key=len, reverse=True):
        if c in nome:
            return c
    raise ValueError(f"Feature transformada sem mapeamento: {nome}")


def diagnostico_territorial(teste, p):
    """Somente escolas do teste; não estima todo o município ou um ano futuro."""
    df = teste.copy()
    df["probabilidade_risco"] = p
    df["risco_observado"] = df[config.ALVO].eq(0).astype(float)
    tabela = df.groupby(["ano", "sigla_uf", "id_municipio", "id_municipio_nome", "rede"], dropna=False).agg(
        avaliacoes_teste=(config.ALVO, "size"), escolas_teste=("id_escola", "nunique"),
        score_risco_medio=("probabilidade_risco", "mean"), risco_observado=("risco_observado", "mean"),
        meta_municipal_contextual=("meta_alfabetizacao_municipio", "first"),
    ).reset_index()
    # Meta municipal tem abrangência Municipal: não comparar outras redes.
    tabela.loc[~tabela.rede.eq("Municipal"), "meta_municipal_contextual"] = np.nan
    # O class_weight pode deslocar a calibração. Não transformar a média
    # dos scores em taxa municipal nem calcular distância prevista da meta.
    tabela["amostra_minima"] = tabela.avaliacoes_teste.ge(100) & tabela.escolas_teste.ge(3)
    tabela.to_parquet(config.BASE.parent / "diagnostico_territorial_teste.parquet", index=False)
    ranking = tabela.loc[tabela.amostra_minima].sort_values("score_risco_medio", ascending=False).head(20)
    # JSON só com agregados de amostras mínimas, sem registros de alunos.
    registros = ranking.astype(object).where(ranking.notna(), None).to_dict("records")
    salvar_json(config.RELATORIOS / "priorizacao_amostra_teste.json", {
        "escopo": "amostra de escolas do teste; scores sem recalibração e sem ponderação; não estima taxa municipal ou resultado futuro",
        "filtro": "pelo menos 100 avaliações e 3 escolas no teste por município/rede",
        "grupos_com_amostra_minima": int(tabela.amostra_minima.sum()), "top20": registros,
    })
    if not ranking.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        r = ranking.iloc[::-1]
        nomes = r.id_municipio_nome + " / " + r.sigla_uf + " / " + r.rede
        ax.barh(nomes, r.score_risco_medio, color=LARANJA)
        ax.set(xlabel="Score médio de risco (0–1; sem recalibração)",
               title="Prioridades na amostra de teste — não representa toda a rede municipal")
        salvar(fig, config.IMAGENS / "prioridades_teste.png")


def interpretar(amostra=5000, repeticoes=5, threads=4):
    if amostra < 2 or repeticoes < 1 or threads < 1:
        raise ValueError("Amostra >= 2, repetições e threads positivas")
    configurar()
    base, manifesto = carregar_base()
    resultado = ler_json(config.RELATORIOS / "resultados.json")
    bundle = joblib.load(config.MODELOS / "modelo_alfabetizacao.joblib")
    validar_artefatos(bundle, resultado, manifesto)
    pipeline = bundle["pipeline"]
    partes = dividir(base)
    rng = np.random.default_rng(config.SEMENTE)
    idx = rng.choice(partes["teste"], size=min(amostra, len(partes["teste"])), replace=False)
    X = base.iloc[idx][config.FEATURES]
    y = base.iloc[idx][config.ALVO]
    print(f"[INTERPRETACAO] {len(X)} avaliações de teste; permutação em {repeticoes} repetições")
    with threadpool_limits(limits=threads):
        perm = permutation_importance(pipeline, X, y, scoring="f1_macro", n_repeats=repeticoes,
                                      n_jobs=1, random_state=config.SEMENTE)
        importancia = pd.DataFrame({"variavel": config.FEATURES, "queda_f1_macro": perm.importances_mean,
                                   "desvio_permutacoes": perm.importances_std}).sort_values("queda_f1_macro", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        r = importancia.iloc[::-1]
        ax.barh([ROTULOS[c] for c in r.variavel], r.queda_f1_macro, xerr=r.desvio_permutacoes, color=AZUL)
        ax.axvline(0, color="gray", linewidth=.8)
        ax.set(xlabel="Queda de F1 macro após permutação", title="Sensibilidade do modelo — variáveis correlacionadas compartilham sinal")
        salvar(fig, config.IMAGENS / "interpretacao_permutacao.png")
        import shap
        transformador = pipeline.named_steps["preprocessar"]
        modelo = pipeline.named_steps["modelo"]
        transformado = transformador.transform(X.iloc[:2000])
        nomes = transformador.get_feature_names_out()
        if bundle["selecao"]["modelo"] == "logistica":
            fundo_idx = rng.choice(partes["treino"], size=min(512, len(partes["treino"])), replace=False)
            fundo = transformador.transform(base.iloc[fundo_idx][config.FEATURES])
            explicacao = shap.LinearExplainer(modelo, fundo)(transformado)
        elif bundle["selecao"]["modelo"] == "gradient_boosting":
            explicacao = shap.TreeExplainer(modelo, model_output="raw")(transformado)
        else:
            raise ValueError("Baseline venceu: não atribuir explicações de features a um modelo constante")
        # Modelos sklearn binários explicam a margem da classe 1 (alfabetizado).
        # Negar as contribuições e a base explica o log-odds do risco (classe 0).
        valores = -np.asarray(explicacao.values)
        base_shap = -np.asarray(explicacao.base_values)
        if valores.ndim != 2:
            raise ValueError("Formato SHAP inesperado para classificação binária")
        reconstruida = expit(base_shap + valores.sum(axis=1))
        p = probabilidade_risco(pipeline, X.iloc[:len(transformado)])
        erro = float(np.max(np.abs(reconstruida - p)))
        if erro > 1e-5:
            raise ValueError(f"Contribuições SHAP não reproduzem a probabilidade: erro={erro}")
        agrupado = np.zeros((len(transformado), len(config.FEATURES)))
        for i, nome in enumerate(nomes):
            agrupado[:, config.FEATURES.index(nome_original(nome))] += valores[:, i]
        global_shap = pd.DataFrame({"variavel": config.FEATURES,
                                   "shap_absoluto_medio_logodds": np.abs(agrupado).mean(axis=0)}).sort_values(
                                       "shap_absoluto_medio_logodds", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        r = global_shap.iloc[::-1]
        ax.barh([ROTULOS[c] for c in r.variavel], r.shap_absoluto_medio_logodds, color=LARANJA)
        ax.set(xlabel="Contribuição SHAP absoluta média (log-odds do risco)",
               title="SHAP global — categorias e indicadores somados à variável original")
        salvar(fig, config.IMAGENS / "interpretacao_shap.png")
        teste = base.iloc[partes["teste"]]
        diagnostico_territorial(teste, probabilidade_risco(pipeline, teste[config.FEATURES]))
    saida = {"run_id": bundle["run_id"], "modelo": bundle["selecao"]["modelo"],
             "amostra_permutacao": len(X), "amostra_shap": len(transformado), "semente": config.SEMENTE,
             "repeticoes": repeticoes, "shap_versao": importlib.metadata.version("shap"),
             "shap_escala": "log-odds da não alfabetização", "erro_maximo_reconstrucao_probabilidade": erro,
             "permutacao": importancia.to_dict("records"), "shap": global_shap.to_dict("records"),
             "limites": "Sensibilidade e contribuição ao modelo; não causalidade. Correlação divide atribuições; desvios de permutação não são IC populacional."}
    salvar_json(config.RELATORIOS / "interpretacao.json", saida)
    print(f"[INTERPRETACAO] SHAP validado: erro máximo {erro:.3g}; relatórios e gráficos salvos")
    return saida


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amostra", type=int, default=5000)
    parser.add_argument("--repeticoes", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    interpretar(args.amostra, args.repeticoes, args.threads)
