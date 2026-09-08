"""Busca no treino, escolha na validação e avaliação final com teste isolado."""
from datetime import datetime, timezone
import importlib.metadata
import platform
import time

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from src import config
from src.artefatos import salvar_json, sha256
from src.evaluation.metricas import avaliar, bootstrap_escolas, por_grupo, probabilidade_risco
from src.modeling.pipelines import GRADES, candidatos
from src.preprocessing.base import carregar_base
from src.preprocessing.divisao import amostrar_escolas, dividir, grupos_escola, resumo_divisao


def escolher_modelo(validacao):
    # Regra pré-declarada: em diferença <= 0.005 de F1 macro entre modelos
    # aprendidos, prefere-se a logística pela simplicidade. Baseline pode vencer.
    melhor = max(validacao, key=lambda nome: validacao[nome]["f1_macro"])
    if melhor == "gradient_boosting" and validacao[melhor]["f1_macro"] - validacao["logistica"]["f1_macro"] <= .005:
        melhor = "logistica"
    return melhor


def treinar(max_busca=150_000, dobras=3, sem_busca=False, threads=4):
    if dobras < 2 or threads < 1 or max_busca < 1:
        raise ValueError("Dobras >= 2, threads e max_busca positivos são obrigatórios")
    config.preparar_diretorios()
    base, manifesto = carregar_base()
    partes = dividir(base)
    split = resumo_divisao(base, partes, manifesto["base_sha256"])
    idx_busca = amostrar_escolas(base, partes["treino"], max_busca)
    X = base[config.FEATURES]
    y = base[config.ALVO]
    grupos = grupos_escola(base)
    if grupos.iloc[idx_busca].nunique() < dobras or y.iloc[idx_busca].nunique() != 2:
        raise ValueError("Amostra de busca insuficiente para as dobras e duas classes")
    cv = list(StratifiedGroupKFold(n_splits=dobras, shuffle=True, random_state=config.SEMENTE).split(
        X.iloc[idx_busca], y.iloc[idx_busca], grupos.iloc[idx_busca]))
    for a, b in cv:
        if set(grupos.iloc[idx_busca[a]]) & set(grupos.iloc[idx_busca[b]]):
            raise ValueError("Sobreposição de escola na validação cruzada")
        if y.iloc[idx_busca[a]].nunique() != 2 or y.iloc[idx_busca[b]].nunique() != 2:
            raise ValueError("Dobra sem duas classes; aumente a amostra")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    pasta_run = config.RELATORIOS / "runs" / run_id
    validacao, buscas, parametrizados = {}, {}, {}
    inicio = time.monotonic()
    print(f"[TREINO] {run_id}: busca com {len(idx_busca):,} linhas em {dobras} dobras por escola")
    with threadpool_limits(limits=threads):
        for nome, pipeline in candidatos().items():
            print(f"[TREINO] Ajustando {nome}...", flush=True)
            t = time.monotonic()
            if nome in GRADES:
                grade = {} if sem_busca else GRADES[nome]
                busca = GridSearchCV(pipeline, grade, cv=cv, scoring={"f1_macro": "f1_macro", "roc_auc": "roc_auc"},
                                     refit="f1_macro", n_jobs=1, error_score="raise", return_train_score=True)
                busca.fit(X.iloc[idx_busca], y.iloc[idx_busca])
                pipeline = clone(busca.best_estimator_)
                res = busca.cv_results_
                buscas[nome] = {"melhores_parametros": busca.best_params_, "linhas": len(idx_busca), "dobras": dobras,
                    "candidatos": [{"parametros": res["params"][i],
                                    "f1_treino": float(res["mean_train_f1_macro"][i]),
                                    "f1_validacao_media": float(res["mean_test_f1_macro"][i]),
                                    "f1_validacao_desvio": float(res["std_test_f1_macro"][i]),
                                    "roc_auc_media": float(res["mean_test_roc_auc"][i])}
                                   for i in range(len(res["params"]))]}
            pipeline.fit(X.iloc[partes["treino"]], y.iloc[partes["treino"]])
            validacao[nome] = avaliar(y.iloc[partes["validacao"]], probabilidade_risco(pipeline, X.iloc[partes["validacao"]]))
            validacao[nome]["tempo_busca_e_ajuste_segundos"] = round(time.monotonic() - t, 2)
            parametrizados[nome] = clone(pipeline)
            print(f"[VALIDACAO] {nome}: F1 macro={validacao[nome]['f1_macro']:.4f}", flush=True)
        escolhido = escolher_modelo(validacao)
        selecao = {"modelo": escolhido, "criterio": "F1 macro de validação; tolerância 0.005 favorece logística",
                   "limiar_risco": .5, "teste_usado_na_selecao": False, "run_id": run_id}
        salvar_json(pasta_run / "selecao_antes_teste.json", selecao)
        print(f"[SELECAO] {escolhido}; escolha congelada antes do teste", flush=True)
        desenvolvimento = np.concatenate([partes["treino"], partes["validacao"]])
        teste, probas, vencedor = {}, {}, None
        for nome, pipeline in parametrizados.items():
            print(f"[FINAL] Reajuste {nome} em treino + validação...", flush=True)
            pipeline.fit(X.iloc[desenvolvimento], y.iloc[desenvolvimento])
            p = probabilidade_risco(pipeline, X.iloc[partes["teste"]])
            teste[nome] = avaliar(y.iloc[partes["teste"]], p)
            probas[nome] = p
            if nome == escolhido:
                vencedor = pipeline
        base_teste = base.iloc[partes["teste"]].copy()
        p = probas[escolhido]
        incerteza = bootstrap_escolas(base_teste[config.ALVO], p, grupos.iloc[partes["teste"]])
        recortes = {c: por_grupo(base_teste, p, c, config.ALVO) for c in ["sigla_uf", "rede", "regiao_brasil", "tem_historico"]}
    versoes = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "pyarrow", "scikit-learn", "joblib", "threadpoolctl"]}
    resultado = {"run_id": run_id, "ano": manifesto["ano"], "execution_date_gold": manifesto["execution_date_gold"],
                 "base_sha256": manifesto["base_sha256"], "features": config.FEATURES, "selecao": selecao,
                 "divisao": split, "busca": buscas, "validacao": validacao, "teste": teste,
                 "incerteza_escolas": incerteza, "recortes_teste": recortes,
                 "ambiente": {"python": platform.python_version(), "versoes": versoes, "threads": threads},
                 "fontes_codigo_sha256": {str(p.relative_to(config.RAIZ)): sha256(p) for p in sorted((config.RAIZ / "src").rglob("*.py"))},
                 "duracao_segundos": round(time.monotonic() - inicio, 2)}
    caminho_modelo = config.MODELOS / "modelo_alfabetizacao.joblib"
    joblib.dump({"pipeline": vencedor, "run_id": run_id, "features": config.FEATURES,
                 "base_sha256": manifesto["base_sha256"], "selecao": selecao}, caminho_modelo, compress=3)
    resultado["modelo_sha256"] = sha256(caminho_modelo)
    # Predições individuais são artefatos locais ignorados pelo Git.
    base_teste["probabilidade_nao_alfabetizado"] = p
    base_teste.to_parquet(config.BASE.parent / "predicoes_teste.parquet", index=False)
    salvar_json(pasta_run / "resultados.json", resultado)
    salvar_json(config.RELATORIOS / "resultados.json", resultado)
    from src.visualization.avaliacao import graficos_avaliacao
    graficos_avaliacao(base_teste[config.ALVO].to_numpy(), probas, escolhido)
    print(f"[CONCLUIDO] {escolhido}: teste F1={teste[escolhido]['f1_macro']:.4f}, AUC={teste[escolhido]['roc_auc']:.4f}")
    return resultado
