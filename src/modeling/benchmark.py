"""Benchmark amplo no nível da avaliação: desenvolve em 2024, testa em 2025.

A base colapsa em município x rede sem perda: cada unidade entra com o peso
(n0, n1) das suas avaliações, preservando a proporção da base inteira, e as
métricas continuam sendo por avaliação. O que era um ajuste em 1,85 milhão de
linhas vira um ajuste em 13 mil, o que permite comparar doze famílias de
algoritmos em minutos em vez de horas.
"""
from datetime import datetime, timezone
from itertools import product
import importlib.metadata
import platform
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

from src import config
from src.artefatos import salvar_json, sha256
from src.evaluation.metricas_ponderadas import avaliar_ponderado, limiar_de_maior_acuracia
from src.modeling.amostragem import ajustar
from src.modeling.zoologico import GRADES_AMPLAS, candidatos_amplos, empilhamento
from src.preprocessing.enriquecida import FEATURES, NOVAS_NUMERICAS, carregar_enriquecida
from src.preprocessing.unidades import carregar_unidades
from src.visualization.estilo import configurar, plt, salvar

NUMERICAS = config.NUMERICAS + NOVAS_NUMERICAS


def pares_com_peso(unidades):
    """Duas linhas por unidade (y=0 e y=1) com o número de avaliações como peso."""
    X = pd.concat([unidades[FEATURES]] * 2, ignore_index=True)
    y = np.concatenate([np.zeros(len(unidades), int), np.ones(len(unidades), int)])
    peso = np.concatenate([unidades.n_nao_alfabetizados.to_numpy(float),
                           unidades.n_alfabetizados.to_numpy(float)])
    grupos = np.concatenate([unidades.id_municipio.to_numpy()] * 2)
    positivos = peso > 0
    return X[positivos].reset_index(drop=True), y[positivos], peso[positivos], grupos[positivos]


def tetos_de_informacao(lake, ano, execucao):
    """Acurácia máxima de qualquer preditor constante por município ou por escola."""
    base, _ = carregar_enriquecida(lake, ano, execucao)
    y = base[config.ALVO].to_numpy()
    tetos = {"baseline_maioria": float(max(y.mean(), 1 - y.mean()))}
    for chaves, nome in [(["id_municipio"], "municipio"), (["id_municipio", "rede"], "municipio_rede"),
                         (["id_municipio", "id_escola"], "escola")]:
        taxa = base.groupby(chaves, observed=True)[config.ALVO].transform("mean").to_numpy()
        acuracias = [float((np.where(taxa >= t, 1, 0) == y).mean()) for t in np.linspace(.05, .95, 91)]
        tetos[f"oraculo_{nome}"] = {"acuracia_max": max(acuracias),
                                    "grupos": int(base.groupby(chaves, observed=True).ngroups)}
    del base
    return tetos


def buscar(pipeline, grade, X, y, peso, cv):
    """Grade exaustiva com peso amostral, avaliada por F1 macro ponderado.

"""
    nomes = list(grade)
    melhor, registros = None, []
    for combinacao in product(*(grade[n] for n in nomes)):
        parametros = dict(zip(nomes, combinacao))
        pontuacoes = []
        for treino, val in cv:
            modelo, _ = ajustar(clone(pipeline).set_params(**parametros), X.iloc[treino],
                                y[treino], peso[treino], semente=config.SEMENTE)
            pontuacoes.append(f1_score(y[val], modelo.predict(X.iloc[val]), labels=[0, 1],
                                       average="macro", zero_division=0, sample_weight=peso[val]))
        media = float(np.mean(pontuacoes))
        registros.append({"parametros": {k: str(v) for k, v in parametros.items()},
                          "f1_macro_cv": media, "desvio": float(np.std(pontuacoes))})
        if melhor is None or media > melhor[0]:
            melhor = (media, parametros)
    return clone(pipeline).set_params(**melhor[1]), {
        "melhores_parametros": {k: str(v) for k, v in melhor[1].items()},
        "f1_macro_cv": melhor[0], "dobras": len(cv), "candidatos": registros}


def executar_benchmark(lake, execucao, ano_treino=2024, ano_teste=2025, threads=4, com_empilhamento=True):
    config.preparar_diretorios()
    configurar()
    inicio = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_benchmark"
    pasta = config.RELATORIOS / "runs" / run_id
    unidades, proveniencia = carregar_unidades(lake, ano_treino, execucao)
    print(f"[BENCH] {ano_treino}: {proveniencia['avaliacoes']:,} avaliacoes colapsadas em "
          f"{len(unidades):,} unidades municipio x rede", flush=True)
    municipios = unidades.id_municipio.to_numpy()
    idx_treino, idx_val = next(GroupShuffleSplit(n_splits=1, test_size=.25,
                                                 random_state=config.SEMENTE).split(unidades, groups=municipios))
    if set(municipios[idx_treino]) & set(municipios[idx_val]):
        raise ValueError("Municipio compartilhado entre treino e validacao")
    treino, validacao = unidades.iloc[idx_treino], unidades.iloc[idx_val]
    Xtr, ytr, wtr, gtr = pares_com_peso(treino)
    Xvl, yvl, wvl, _ = pares_com_peso(validacao)
    cv = list(GroupKFold(n_splits=3).split(Xtr, ytr, gtr))
    modelos = candidatos_amplos(NUMERICAS)
    if com_empilhamento:
        modelos["empilhamento"] = empilhamento(NUMERICAS)
    resultados, ajustados, buscas = {}, {}, {}
    with threadpool_limits(limits=threads):
        for nome, pipeline in modelos.items():
            t = time.monotonic()
            if nome in GRADES_AMPLAS:
                pipeline, buscas[nome] = buscar(pipeline, GRADES_AMPLAS[nome], Xtr, ytr, wtr, cv)
            modelo, modo = ajustar(clone(pipeline), Xtr, ytr, wtr, semente=config.SEMENTE)
            p = modelo.predict_proba(Xvl)[:, list(modelo.classes_).index(0)]
            limiar, acuracia = limiar_de_maior_acuracia(yvl, p, wvl)
            resultados[nome] = {**avaliar_ponderado(yvl, p, wvl), "ajuste": modo,
                                "limiar_de_maior_acuracia": limiar, "acuracia_nesse_limiar": acuracia,
                                "segundos": round(time.monotonic() - t, 1)}
            ajustados[nome] = clone(pipeline)
            print(f"[VALIDACAO] {nome:22s} acc={resultados[nome]['acuracia']:.4f} "
                  f"acc*={acuracia:.4f}(th={limiar:.2f}) f1m={resultados[nome]['f1_macro']:.4f} "
                  f"auc={resultados[nome]['roc_auc']:.4f} [{resultados[nome]['segundos']}s]", flush=True)
        aprendidos = [n for n in resultados if n != "baseline_prior"]
        escolhido = max(aprendidos, key=lambda n: resultados[n]["f1_macro"])
        por_acuracia = max(aprendidos, key=lambda n: resultados[n]["acuracia_nesse_limiar"])
        selecao = {"modelo_por_f1_macro": escolhido, "modelo_por_acuracia": por_acuracia,
                   "limiar_congelado": resultados[por_acuracia]["limiar_de_maior_acuracia"],
                   "criterio": "F1 macro e acuracia na validacao de municipios disjuntos de 2024",
                   "teste_usado_na_selecao": False, "run_id": run_id}
        salvar_json(pasta / "selecao_antes_teste.json", selecao)
        print(f"[SELECAO] F1 macro: {escolhido}; acuracia: {por_acuracia} "
              f"(limiar {selecao['limiar_congelado']:.2f})", flush=True)
        Xdev, ydev, wdev, _ = pares_com_peso(unidades)
        unidades_teste, prov_teste = carregar_unidades(lake, ano_teste, execucao)
        Xte, yte, wte, _ = pares_com_peso(unidades_teste)
        teste = {}
        for nome, pipeline in ajustados.items():
            modelo, _ = ajustar(clone(pipeline), Xdev, ydev, wdev, semente=config.SEMENTE)
            p = modelo.predict_proba(Xte)[:, list(modelo.classes_).index(0)]
            teste[nome] = {"limiar_meio": avaliar_ponderado(yte, p, wte),
                           "limiar_congelado": avaliar_ponderado(yte, p, wte, selecao["limiar_congelado"])}
            print(f"[TESTE {ano_teste}] {nome:22s} acc={teste[nome]['limiar_meio']['acuracia']:.4f} "
                  f"acc(th*)={teste[nome]['limiar_congelado']['acuracia']:.4f} "
                  f"f1m={teste[nome]['limiar_meio']['f1_macro']:.4f} "
                  f"auc={teste[nome]['limiar_meio']['roc_auc']:.4f}", flush=True)
    tetos = tetos_de_informacao(lake, ano_teste, execucao)
    resultado = {"run_id": run_id, "ano_treino": ano_treino, "ano_teste": ano_teste,
                 "execucao_gold": execucao, "features": FEATURES, "selecao": selecao,
                 "proveniencia_treino": proveniencia["fontes"], "proveniencia_teste": prov_teste["fontes"],
                 "colapso": {"avaliacoes_treino": proveniencia["avaliacoes"],
                             "unidades_treino": len(unidades), "unidades_validacao": len(validacao),
                             "avaliacoes_teste": prov_teste["avaliacoes"], "unidades_teste": len(unidades_teste),
                             "nota": "todos os preditores sao constantes dentro de municipio x rede"},
                 "busca": buscas, "validacao": resultados, "teste": teste,
                 "tetos_de_informacao_no_teste": tetos,
                 "ambiente": {"python": platform.python_version(), "threads": threads,
                              "versoes": {p: importlib.metadata.version(p)
                                          for p in ["numpy", "pandas", "scikit-learn", "scipy"]}},
                 "codigo_sha256": {str(p.relative_to(config.RAIZ)): sha256(p)
                                   for p in sorted((config.RAIZ / "src").rglob("*.py"))},
                 "duracao_segundos": round(time.monotonic() - inicio, 1)}
    salvar_json(pasta / "resultados.json", resultado)
    salvar_json(config.RELATORIOS / "benchmark_modelos.json", resultado)
    grafico_benchmark(resultado)
    return resultado


def grafico_benchmark(resultado):
    nomes = [n for n in resultado["teste"] if n != "baseline_prior"]
    acc = [resultado["teste"][n]["limiar_congelado"]["acuracia"] for n in nomes]
    ordem = np.argsort(acc)
    fig, ax = plt.subplots(figsize=(9, .42 * len(nomes) + 2.4))
    ax.barh([nomes[i] for i in ordem], [acc[i] for i in ordem], color="#176B87")
    tetos = resultado["tetos_de_informacao_no_teste"]
    for valor, rotulo, cor in [(tetos["baseline_maioria"], "maioria", "gray"),
                               (tetos["oraculo_municipio_rede"]["acuracia_max"], "teto municipio x rede", "#DC6B35"),
                               (tetos["oraculo_escola"]["acuracia_max"], "teto escola", "#8B1E3F")]:
        ax.axvline(valor, linestyle="--", color=cor, linewidth=1.2, label=f"{rotulo} = {valor:.3f}")
    ax.set(xlim=(.5, .78), xlabel=f"Acuracia no teste {resultado['ano_teste']}",
           title="Nenhum algoritmo ultrapassa o teto de informacao da base")
    ax.legend(loc="lower right", fontsize=8)
    salvar(fig, config.IMAGENS / "benchmark_acuracia.png")
