"""Desenvolve em 2024 e avalia em 2025; teste não participa da seleção/calibração."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from src import config
from src.artefatos import salvar_json, sha256
from src.evaluation.metricas import avaliar, bootstrap_escolas, por_grupo, probabilidade_risco
from src.modeling.pipelines import candidatos, GRADES
from src.preprocessing.divisao import dividir, amostrar_escolas, grupos_escola
from src.preprocessing.enriquecida import carregar_enriquecida, FEATURES, NOVAS_NUMERICAS
from src.visualization.estilo import configurar, salvar, plt


# Perfil experimental solicitado: retirar histórico de alfabetização e recortes
# territoriais, mantendo rede e os complementos municipais/escolares.
EXCLUIDAS_TESTE = {"taxa_alfabetizacao_municipio_anterior", "sigla_uf", "regiao_brasil"}


def verificar_ordem(ano_treino, ano_teste):
    if ano_teste <= ano_treino:
        raise ValueError("Teste temporal deve ser posterior ao treino")


def calibrar_congelado(modelo, X, y):
    """Ajusta apenas o mapa sigmoide em escolas que não ajustaram o estimador."""
    # FrozenEstimator ignora fit nas dobras: todas as saídas vêm do mesmo
    # estimador já ajustado. Só o mapa sigmoide aprende com esta amostra.
    calibrado = CalibratedClassifierCV(FrozenEstimator(modelo), method="sigmoid", cv=2, n_jobs=1)
    calibrado.fit(X, y)
    return calibrado


def executar_temporal(lake, execucao, ano_treino=2024, ano_teste=2025, max_busca=150000, dobras=3, threads=4):
    verificar_ordem(ano_treino, ano_teste)
    if dobras < 2 or max_busca < 1 or threads < 1:
        raise ValueError("Dobras >= 2; amostra e threads positivas")
    config.preparar_diretorios()
    configurar()
    inicio = time.monotonic()
    base, proveniencia = carregar_enriquecida(lake, ano_treino, execucao)
    features_modelo = [c for c in FEATURES if c not in EXCLUIDAS_TESTE]
    numericas_modelo = [c for c in config.NUMERICAS + NOVAS_NUMERICAS if c not in EXCLUIDAS_TESTE]
    categoricas_modelo = [c for c in config.CATEGORICAS if c not in EXCLUIDAS_TESTE]
    partes = dividir(base)
    partes["calibracao"] = partes.pop("teste")
    grupos = grupos_escola(base)
    resumo = {nome: {"linhas": len(idx), "escolas": int(grupos.iloc[idx].nunique()),
                     "indices_sha256": hashlib.sha256(np.asarray(idx, dtype="int64").tobytes()).hexdigest()}
               for nome, idx in partes.items()}
    idx = amostrar_escolas(base, partes["treino"], max_busca)
    X, y = base[features_modelo], base[config.ALVO]
    cv = list(StratifiedGroupKFold(n_splits=dobras, shuffle=True, random_state=config.SEMENTE).split(
        X.iloc[idx], y.iloc[idx], grupos.iloc[idx]))
    for a, b in cv:
        if set(grupos.iloc[idx[a]]) & set(grupos.iloc[idx[b]]) or y.iloc[idx[a]].nunique() != 2 or y.iloc[idx[b]].nunique() != 2:
            raise ValueError("CV temporal: escola compartilhada ou dobra sem duas classes")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_temporal"
    pasta = config.RELATORIOS / "runs" / run_id
    # EDA somente no treino, com medidas adicionais. Não altera a EDA inicial.
    treino = base.iloc[partes["treino"]]
    eda = {"ano": ano_treino, "escopo": "treino", "n": len(treino),
           "ausencias_pct": treino[FEATURES].isna().mean().mul(100).to_dict(),
           "novas_numericas": treino[NOVAS_NUMERICAS].describe().to_dict()}
    salvar_json(config.RELATORIOS / "eda_enriquecida.json", eda)
    del treino
    originais = candidatos(numericas_modelo, categoricas_modelo)
    novos = candidatos(numericas_modelo, categoricas_modelo)
    modelos = {"baseline": originais["baseline"], "referencia_logistica": originais["logistica"],
               "enriquecida_logistica": novos["logistica"], "enriquecida_boosting": novos["gradient_boosting"]}
    validacao, buscas, ajustados = {}, {}, {}
    print(f"[TEMPORAL] Desenvolvimento {ano_treino}; busca {len(idx):,} avaliações; {dobras} dobras", flush=True)
    with threadpool_limits(limits=threads):
        for nome, modelo in modelos.items():
            print(f"[TEMPORAL] Ajustando {nome}", flush=True)
            if nome != "baseline":
                grade = GRADES["gradient_boosting" if nome.endswith("boosting") else "logistica"]
                busca = GridSearchCV(modelo, grade, cv=cv, scoring="f1_macro", n_jobs=1,
                                     error_score="raise", return_train_score=True)
                busca.fit(X.iloc[idx], y.iloc[idx])
                modelo = clone(busca.best_estimator_)
                res = busca.cv_results_
                buscas[nome] = {"melhores_parametros": busca.best_params_, "linhas": len(idx), "dobras": dobras,
                                "candidatos": [{"parametros": res["params"][i],
                                                "f1_treino": float(res["mean_train_score"][i]),
                                                "f1_validacao_media": float(res["mean_test_score"][i]),
                                                "f1_validacao_desvio": float(res["std_test_score"][i])}
                                               for i in range(len(res["params"]))]}
            modelo.fit(X.iloc[partes["treino"]], y.iloc[partes["treino"]])
            validacao[nome] = avaliar(y.iloc[partes["validacao"]], probabilidade_risco(modelo, X.iloc[partes["validacao"]]))
            ajustados[nome] = clone(modelo)
            print(f"[VALIDACAO {ano_treino}] {nome}: F1={validacao[nome]['f1_macro']:.4f}", flush=True)
        escolhido = max(validacao, key=lambda nome: validacao[nome]["f1_macro"])
        selecao = {"modelo": escolhido, "criterio": "maior F1 macro de validação antes de calibrar; empates exatos pela ordem dos candidatos",
                   "ano_treino": ano_treino, "ano_teste": ano_teste, "limiar": .5,
                   "calibracao": "sigmoid fixa, em 20% de escolas de desenvolvimento reservadas",
                   "teste_usado_na_selecao": False, "run_id": run_id}
        salvar_json(pasta / "selecao_antes_teste.json", selecao)
        desenvolvimento = np.concatenate([partes["treino"], partes["validacao"]])
        for nome, modelo in ajustados.items():
            modelo.fit(X.iloc[desenvolvimento], y.iloc[desenvolvimento])
        vencedor = calibrar_congelado(ajustados[escolhido], X.iloc[partes["calibracao"]], y.iloc[partes["calibracao"]])
        caminho_modelo = config.MODELOS / "modelo_temporal.joblib"
        joblib.dump({"pipeline": vencedor, "pipeline_sem_calibracao": ajustados[escolhido], "features": FEATURES,
                     "run_id": run_id, "selecao": selecao, "origem_treino": proveniencia["fontes"]}, caminho_modelo, compress=3)
        # Salvo antes de carregar a edição de teste: seleção, ajuste e calibração terminam aqui.
        congelamento = {**selecao, "modelo_sha256": sha256(caminho_modelo),
                        "congelado_em_utc": datetime.now(timezone.utc).isoformat()}
        salvar_json(config.RELATORIOS / "selecao_temporal_antes_teste.json", congelamento)
        salvar_json(pasta / "modelo_congelado.json", congelamento)
        ufs_treino = set(base.sigla_uf)
        municipios_treino = set(base.id_municipio)
        del X, y, base, grupos
        print(f"[TEMPORAL] Modelo congelado. Abrindo teste {ano_teste}...", flush=True)
        teste, origem_teste = carregar_enriquecida(lake, ano_teste, execucao)
        X_teste, y_teste = teste[features_modelo], teste[config.ALVO]
        metricas, probas = {}, {}
        for nome, modelo in ajustados.items():
            p = probabilidade_risco(modelo, X_teste)
            metricas[nome] = avaliar(y_teste, p)
            probas[nome] = p
        p = probabilidade_risco(vencedor, X_teste)
        metricas["selecionado_calibrado"] = avaliar(y_teste, p)
        teste["uf_nova"] = ~teste.sigla_uf.isin(ufs_treino)
        teste["municipio_novo"] = ~teste.id_municipio.isin(municipios_treino)
        recortes = {c: por_grupo(teste, p, c, config.ALVO)
                    for c in ["sigla_uf", "regiao_brasil", "rede", "uf_nova", "municipio_novo", "tem_historico"]}
        incerteza = bootstrap_escolas(y_teste, p, grupos_escola(teste))
        amostra = np.random.default_rng(config.SEMENTE).choice(len(teste), min(5000, len(teste)), replace=False)
        perm = permutation_importance(vencedor, X_teste.iloc[amostra], y_teste.iloc[amostra],
                                      scoring="f1_macro", n_repeats=5, n_jobs=1, random_state=config.SEMENTE)
        importancia = pd.DataFrame({"variavel": features_modelo, "queda_f1_macro": perm.importances_mean,
                                   "desvio": perm.importances_std}).sort_values("queda_f1_macro", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 8))
        r = importancia.iloc[::-1]
        ax.barh(r.variavel, r.queda_f1_macro, xerr=r.desvio)
        ax.axvline(0, color="gray", linewidth=.7)
        ax.set(title=f"Permutação no teste {ano_teste} — modelo congelado e calibrado", xlabel="Queda de F1 macro")
        salvar(fig, config.IMAGENS / "temporal_permutacao.png")
        fig, ax = plt.subplots(figsize=(8, 5))
        risco = y_teste.eq(0).astype(int)
        curvas = {}
        for nome, valores in [("Sem calibração", probas[escolhido]), ("Sigmoide aprendida em 2024", p)]:
            obs, prev = calibration_curve(risco, valores, n_bins=10, strategy="quantile")
            ax.plot(prev, obs, "o-", label=nome)
            curvas[nome] = {"score_medio": prev.tolist(), "frequencia_observada": obs.tolist()}
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(title=f"Calibração fora do tempo — teste {ano_teste}", xlabel="Probabilidade média de risco", ylabel="Fração observada de não alfabetizados")
        ax.legend()
        salvar(fig, config.IMAGENS / "temporal_calibracao.png")
        # Apenas agregados e amostras com pelo menos 100 avaliações/3 escolas.
        territorio = teste.assign(score_risco=p, observado=risco).groupby(
            ["sigla_uf", "id_municipio", "id_municipio_nome", "rede"]).agg(
                avaliacoes=(config.ALVO, "size"), escolas=("id_escola", "nunique"),
                score_medio=("score_risco", "mean"), fracao_nao_alfabetizada=("observado", "mean")).reset_index()
        territorio = territorio.loc[territorio.avaliacoes.ge(100) & territorio.escolas.ge(3)]
        territorio.to_parquet(config.BASE.parent / "diagnostico_territorial_2025.parquet", index=False)
    resultado = {"run_id": run_id, "execucao_gold": execucao, "selecao": congelamento,
                 "features": features_modelo, "novas_features": NOVAS_NUMERICAS, "divisao_desenvolvimento": resumo,
                 "proveniencia_treino": proveniencia["fontes"], "proveniencia_teste": origem_teste["fontes"],
                 "cobertura_gold": origem_teste["manifesto_gold"]["cobertura"],
                 "busca": buscas, "validacao": validacao, "teste": metricas,
                 "recortes_teste": recortes, "incerteza_escolas": incerteza,
                 "permutacao": importancia.to_dict("records"), "amostra_permutacao": len(amostra), "repeticoes_permutacao": 5,
                 "curvas_calibracao": curvas,
                 "ranking_diagnostico_2025": territorio.sort_values("score_medio", ascending=False).head(20).to_dict("records"),
                 "ambiente": {"python": platform.python_version(), "threads": threads,
                              "versoes": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "pandas", "scikit-learn", "pyarrow", "joblib"]}},
                 "codigo_sha256": {str(p.relative_to(config.RAIZ)): sha256(p) for p in sorted((config.RAIZ / "src").rglob("*.py"))},
                 "duracao_segundos": time.monotonic() - inicio,
                 "metas_futuras_habilitadas": False,
                 "limites": ["Teste de outra edição; identidades escolares não são longitudinais.",
                             "Datas e versões históricas de publicação não estão integralmente comprovadas.",
                             "Calibração aprendida em 2024 pode se degradar com mudanças em 2025.",
                             "População de 2022 é censo; 2024 é estimativa; não inferir crescimento diretamente.",
                             "Resultados não ponderados por desenho amostral; não estimam indicadores oficiais.",
                             "Não prever metas futuras até validar disponibilidade temporal, agregação e incerteza."]}
    salvar_json(pasta / "resultados.json", resultado)
    salvar_json(config.RELATORIOS / "resultados_temporais.json", resultado)
    salvar_json(config.RELATORIOS / "proveniencia_enriquecimento.json", origem_teste["manifesto_gold"])
    from src.evaluation.relatorio_temporal import gerar_relatorio_temporal
    gerar_relatorio_temporal(resultado)
    print(f"[TEMPORAL CONCLUIDO] {escolhido}; teste {ano_teste}: F1 calibrado={metricas['selecionado_calibrado']['f1_macro']:.4f}", flush=True)
    return resultado
