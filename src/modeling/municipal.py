"""Classificação territorial: município x rede abaixo da mediana de alfabetização.

Motivação medida, não estilística. Os 79 preditores do contrato enriquecido são
constantes dentro de município x rede; no nível da avaliação individual eles não
distinguem duas crianças da mesma rede, e a acurácia fica presa ao teto do
oráculo municipal. Ao adotar o território como unidade, o ruído binomial
individual sai da métrica e o mesmo conjunto de variáveis passa a separar
territórios com folga.

Alvo `risco_territorial`: 1 = taxa de alfabetização do território abaixo da
mediana nacional do ano; 0 = igual ou acima. Classes equilibradas por
construção, então a acurácia é comparável a um acaso de 50%.
"""
from datetime import datetime, timezone
import importlib.metadata
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

from src import config
from src.artefatos import salvar_json, sha256
from src.modeling.benchmark import buscar
from src.modeling.zoologico import GRADES_AMPLAS, candidatos_amplos, empilhamento
from src.preprocessing.enriquecida import FEATURES, NOVAS_NUMERICAS
from src.preprocessing.unidades import carregar_unidades

MINIMO_AVALIACOES = 30
MINIMO_ESCOLAS = 2
ALVO = "risco_territorial"
# Histórico da própria alfabetização: separá-lo permite medir quanto os
# determinantes estruturais explicam sem o atalho da persistência.
HISTORICO = ["taxa_alfabetizacao_municipio_anterior", "taxa_presenca_municipio_anterior",
             "alunos_avaliados_municipio_anterior"]
ESTRUTURAIS = [c for c in FEATURES if c not in HISTORICO]
BLOCOS = {"completo": FEATURES, "estrutural": ESTRUTURAIS}


def preparar(lake, ano, execucao, corte=None):
    """Unidades confiáveis do ano, com o alvo relativo à mediana nacional."""
    unidades, proveniencia = carregar_unidades(lake, ano, execucao)
    antes = len(unidades)
    unidades = unidades.loc[unidades.avaliacoes.ge(MINIMO_AVALIACOES)
                            & unidades.escolas.ge(MINIMO_ESCOLAS)].reset_index(drop=True)
    corte = float(unidades.taxa_alfabetizacao.median()) if corte is None else float(corte)
    unidades[ALVO] = unidades.taxa_alfabetizacao.lt(corte).astype(int)
    proveniencia.update({"corte_taxa": corte, "unidades_antes_do_filtro": antes,
                         "unidades_usadas": len(unidades),
                         "avaliacoes_cobertas": int(unidades.avaliacoes.sum()),
                         "prevalencia_risco": float(unidades[ALVO].mean()),
                         "filtro": f"avaliacoes >= {MINIMO_AVALIACOES} e escolas >= {MINIMO_ESCOLAS}"})
    return unidades, proveniencia


def avaliar_territorio(y, p, limiar=.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    previsto = (p >= limiar).astype(int)
    duas = len(np.unique(y)) == 2
    return {"n": int(len(y)), "limiar": float(limiar), "prevalencia_risco": float(y.mean()),
            "acuracia": accuracy_score(y, previsto),
            "acuracia_balanceada": balanced_accuracy_score(y, previsto) if duas else None,
            "f1_macro": f1_score(y, previsto, labels=[0, 1], average="macro", zero_division=0),
            "f1_risco": f1_score(y, previsto, pos_label=1, zero_division=0),
            "recall_risco": recall_score(y, previsto, pos_label=1, zero_division=0),
            "precisao_risco": precision_score(y, previsto, pos_label=1, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if duas else None,
            "average_precision": average_precision_score(y, p) if duas else None,
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p, labels=[0, 1]),
            "matriz_confusao_labels_0_1": confusion_matrix(y, previsto, labels=[0, 1]).tolist()}


def numericas_do_bloco(colunas):
    return [c for c in config.NUMERICAS + NOVAS_NUMERICAS if c in colunas]


def avaliar_nas_duas_regras(y, p):
    """Limiar 0,5 e regra de posto, declarada antes de ver o teste.

    O alvo é definido pela mediana do ano, então metade dos territórios está em
    risco por construção. Marcar a metade de menor score usa apenas a ordenação
    prevista e a definição do alvo, nunca os rótulos do teste. Isso corrige o
    deslocamento do nível entre edições (corte 0,619 em 2024 e 0,708 em 2025),
    que penaliza modelos lineares calibrados na escala do ano de treino.
    """
    mediana = float(np.median(p))
    return {"limiar_meio": avaliar_territorio(y, p),
            "regra_de_posto": {**avaliar_territorio(y, p, mediana), "limiar_usado": mediana}}


def executar_municipal(lake, execucao, ano_treino=2024, ano_teste=2025, threads=4, com_empilhamento=True):
    config.preparar_diretorios()
    inicio = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_municipal"
    pasta = config.RELATORIOS / "runs" / run_id
    desenvolvimento, prov_dev = preparar(lake, ano_treino, execucao)
    teste, prov_teste = preparar(lake, ano_teste, execucao)
    print(f"[MUNICIPAL] desenvolvimento {ano_treino}: {len(desenvolvimento)} territorios "
          f"(corte {prov_dev['corte_taxa']:.4f}); teste {ano_teste}: {len(teste)} territorios "
          f"(corte {prov_teste['corte_taxa']:.4f})", flush=True)
    grupos = desenvolvimento.id_municipio.to_numpy()
    idx_tr, idx_vl = next(GroupShuffleSplit(n_splits=1, test_size=.25,
                                            random_state=config.SEMENTE).split(desenvolvimento, groups=grupos))
    if set(grupos[idx_tr]) & set(grupos[idx_vl]):
        raise ValueError("Municipio compartilhado entre treino e validacao")
    teste = teste.assign(municipio_novo=~teste.id_municipio.isin(set(desenvolvimento.id_municipio)))
    resultado = {"run_id": run_id, "ano_treino": ano_treino, "ano_teste": ano_teste,
                 "execucao_gold": execucao, "alvo": ALVO,
                 "definicao_alvo": "1 = taxa de alfabetizacao do territorio abaixo da mediana nacional do ano",
                 "unidade": "id_municipio x rede", "filtro": prov_dev["filtro"],
                 "desenvolvimento": prov_dev, "teste": prov_teste, "blocos": {}}
    modelo_final = None
    with threadpool_limits(limits=threads):
        for bloco, colunas in BLOCOS.items():
            print(f"\n[BLOCO {bloco}] {len(colunas)} preditores", flush=True)
            saida, vencedor = rodar_bloco(desenvolvimento, teste, colunas, idx_tr, idx_vl,
                                          com_empilhamento, run_id)
            resultado["blocos"][bloco] = saida
            if bloco == "completo":
                modelo_final = vencedor
    caminho = config.MODELOS / "modelo_municipal.joblib"
    joblib.dump({"pipeline": modelo_final, "features": FEATURES, "alvo": ALVO, "run_id": run_id,
                 "corte_desenvolvimento": prov_dev["corte_taxa"]}, caminho, compress=3)
    resultado["modelo_sha256"] = sha256(caminho)
    resultado["ambiente"] = {"python": platform.python_version(), "threads": threads,
                             "versoes": {p: importlib.metadata.version(p)
                                         for p in ["numpy", "pandas", "scikit-learn", "scipy"]}}
    resultado["codigo_sha256"] = {str(p.relative_to(config.RAIZ)): sha256(p)
                                  for p in sorted((config.RAIZ / "src").rglob("*.py"))}
    resultado["duracao_segundos"] = round(time.monotonic() - inicio, 1)
    resultado["limites"] = [
        "O alvo e relativo a mediana do proprio ano: mede posicao, nao nivel absoluto.",
        "Territorios com menos de 30 avaliacoes ou 2 escolas ficam fora; taxas seriam instaveis.",
        "Associacao entre contexto e alfabetizacao; nenhum coeficiente e efeito causal.",
        "Municipios aparecem em 2024 e 2025; o recorte 'municipio_novo' isola os ineditos.",
    ]
    salvar_json(pasta / "resultados.json", resultado)
    salvar_json(config.RELATORIOS / "modelo_municipal.json", resultado)
    return resultado


def rodar_bloco(desenvolvimento, teste, colunas, idx_tr, idx_vl, com_empilhamento, run_id):
    numericas = numericas_do_bloco(colunas)
    modelos = candidatos_amplos(numericas)
    if com_empilhamento:
        modelos["empilhamento"] = empilhamento(numericas)
    Xtr, ytr = desenvolvimento.iloc[idx_tr][colunas], desenvolvimento.iloc[idx_tr][ALVO].to_numpy()
    Xvl, yvl = desenvolvimento.iloc[idx_vl][colunas], desenvolvimento.iloc[idx_vl][ALVO].to_numpy()
    gtr = desenvolvimento.iloc[idx_tr].id_municipio.to_numpy()
    cv = list(GroupKFold(n_splits=4).split(Xtr, ytr, gtr))
    pesos = np.ones(len(ytr))
    validacao, ajustados, buscas = {}, {}, {}
    for nome, pipeline in modelos.items():
        t = time.monotonic()
        if nome in GRADES_AMPLAS:
            pipeline, buscas[nome] = buscar(pipeline, GRADES_AMPLAS[nome], Xtr, ytr, pesos, cv)
        modelo = clone(pipeline).fit(Xtr, ytr)
        p = modelo.predict_proba(Xvl)[:, list(modelo.classes_).index(1)]
        validacao[nome] = {**avaliar_nas_duas_regras(yvl, p), "segundos": round(time.monotonic() - t, 1)}
        ajustados[nome] = clone(pipeline)
        print(f"  [VAL] {nome:22s} acc={validacao[nome]['limiar_meio']['acuracia']:.4f} "
              f"acc_posto={validacao[nome]['regra_de_posto']['acuracia']:.4f} "
              f"auc={validacao[nome]['limiar_meio']['roc_auc']:.4f} "
              f"[{validacao[nome]['segundos']}s]", flush=True)
    aprendidos = [n for n in validacao if n != "baseline_prior"]
    escolhido = max(aprendidos, key=lambda n: validacao[n]["regra_de_posto"]["acuracia"])
    print(f"  [SELECAO] {escolhido}; congelado antes do teste", flush=True)
    Xdev, ydev = desenvolvimento[colunas], desenvolvimento[ALVO].to_numpy()
    Xte, yte = teste[colunas], teste[ALVO].to_numpy()
    metricas, vencedor = {}, None
    for nome, pipeline in ajustados.items():
        modelo = clone(pipeline).fit(Xdev, ydev)
        p = modelo.predict_proba(Xte)[:, list(modelo.classes_).index(1)]
        metricas[nome] = avaliar_nas_duas_regras(yte, p)
        print(f"  [TESTE] {nome:22s} acc={metricas[nome]['limiar_meio']['acuracia']:.4f} "
              f"acc_posto={metricas[nome]['regra_de_posto']['acuracia']:.4f} "
              f"auc={metricas[nome]['limiar_meio']['roc_auc']:.4f}", flush=True)
        if nome == escolhido:
            vencedor, p_vencedor = modelo, p
    limiar_posto = float(np.median(p_vencedor))
    recortes = {}
    for coluna in ["rede", "regiao_brasil", "municipio_novo"]:
        recortes[coluna] = [{"grupo": str(valor), "n": int(len(pos)),
                             **{k: v for k, v in avaliar_territorio(yte[pos], p_vencedor[pos], limiar_posto).items()
                                if k in ("acuracia", "f1_macro", "roc_auc", "prevalencia_risco")}}
                            for valor, pos in teste.reset_index(drop=True).groupby(coluna, dropna=False).indices.items()
                            if len(np.unique(yte[pos])) == 2]
    saida = {"preditores": colunas, "n_preditores": len(colunas),
             "selecao": {"modelo": escolhido,
                         "criterio": "maior acuracia sob a regra de posto na validacao de municipios disjuntos",
                         "teste_usado_na_selecao": False, "run_id": run_id},
             "busca": buscas, "validacao": validacao, "teste": metricas, "recortes_teste": recortes}
    return saida, vencedor
