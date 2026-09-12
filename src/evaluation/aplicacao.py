"""Respostas de negócio a partir do modelo territorial congelado.

Prioriza territórios, agrupa perfis semelhantes e mede a distância até a meta
municipal. Nada aqui refaz seleção de modelo: o `joblib` gravado pelo comando
`municipal` é carregado como está e apenas aplicado à edição de teste.

A meta municipal entra como contexto, nunca como preditor: as datas de
publicação e revisão das metas não foram comprovadas, então usá-la para treinar
seria vazamento de informação futura. Ela serve para traduzir o score em uma
pergunta que o gestor faz — quem está longe de onde deveria estar.
"""
from datetime import datetime, timezone
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config
from src.artefatos import salvar_json, sha256
from src.modeling.municipal import ALVO, ESTRUTURAIS, MINIMO_AVALIACOES, MINIMO_ESCOLAS
from src.preprocessing.enriquecida import FEATURES
from src.preprocessing.unidades import carregar_unidades
from src.visualization.estilo import AZUL, LARANJA, configurar, plt, salvar

METAS = ["meta_alfabetizacao_municipio", "rede_meta_municipio", "ano_base_meta_municipio",
         "meta_alfabetizacao_uf", "meta_alfabetizacao_brasil"]
TOPO = 25
# As metas e as taxas municipais do contrato Gold vêm em percentual (0 a 100);
# a taxa calculada no colapso é fração (0 a 1). Comparar sem converter marcaria
# todo território como abaixo da meta.
ESCALA_PERCENTUAL = 100.
CORTES_DO_TOPO = (10, 25, 50, 100)


def carregar_modelo():
    caminho = config.MODELOS / "modelo_municipal.joblib"
    if not caminho.exists():
        raise FileNotFoundError("Rode `python main.py municipal` antes de `aplicacao`")
    pacote = joblib.load(caminho)
    if pacote["features"] != FEATURES:
        raise ValueError("Contrato de preditores mudou desde o treino do modelo territorial")
    return pacote, sha256(caminho)


def territorios_do_teste(lake, ano, execucao):
    unidades, proveniencia = carregar_unidades(lake, ano, execucao, extras=METAS)
    unidades = unidades.loc[unidades.avaliacoes.ge(MINIMO_AVALIACOES)
                            & unidades.escolas.ge(MINIMO_ESCOLAS)].reset_index(drop=True)
    unidades[ALVO] = unidades.taxa_alfabetizacao.lt(unidades.taxa_alfabetizacao.median()).astype(int)
    return unidades, proveniencia


def priorizar(unidades, score):
    """Ordena por risco previsto; mostra o observado ao lado, sem escondê-lo."""
    tabela = unidades.assign(
        score_risco=score,
        meta_municipal=unidades.meta_alfabetizacao_municipio / ESCALA_PERCENTUAL
    ).sort_values("score_risco", ascending=False)
    colunas = ["id_municipio_nome", "sigla_uf", "rede", "avaliacoes", "escolas",
               "score_risco", "taxa_alfabetizacao", "meta_municipal", ALVO]
    return tabela[colunas].head(TOPO).to_dict("records")


def precisao_no_topo(unidades, score):
    """Quanto do topo da lista está de fato abaixo da mediana do ano.

    Uma lista de prioridade só é útil se acertar onde o gestor vai olhar
    primeiro. A precisão em K expõe isso melhor que a acurácia global, que
    dilui o erro do topo entre milhares de territórios medianos.
    """
    ordem = np.argsort(-np.asarray(score))
    y = unidades[ALVO].to_numpy()
    base = float(y.mean())
    return {"prevalencia_de_referencia": base,
            "cortes": [{"k": k, "precisao": float(y[ordem[:k]].mean()),
                        "ganho_sobre_o_acaso": float(y[ordem[:k]].mean() - base)}
                       for k in CORTES_DO_TOPO if k <= len(y)]}


def transformador_numerico(numericas):
    """Mesmo tratamento do pipeline de modelagem, sem o one-hot das categóricas.

    `pre_processador` sempre inclui rede, UF e região; aqui elas ficariam de
    fora do agrupamento, então o transformador é montado com as numéricas
    apenas — mediana para ausentes, log1p em volumes, padronização.
    """
    taxas = [c for c in numericas if c.startswith(("taxa_", "pct_"))]
    volumes = [c for c in numericas if c not in taxas]
    simples = Pipeline([("imputar", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("escalar", StandardScaler())])
    assimetricos = Pipeline([("imputar", SimpleImputer(strategy="median", keep_empty_features=True)),
                             ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                             ("escalar", StandardScaler())])
    return ColumnTransformer([("taxas", simples, taxas), ("volumes", assimetricos, volumes)],
                             remainder="drop", sparse_threshold=0)


def perfis_territoriais(unidades, semente=config.SEMENTE):
    """Agrupa territórios por contexto estrutural, sem usar resultado nem geografia.

    UF, região e rede ficam fora do agrupamento de propósito: incluí-las faria
    o KMeans redesenhar o mapa e a resposta seria circular. Agrupando só por
    infraestrutura, recursos e porte, a distribuição das regiões entre os
    perfis vira a resposta — quais regiões compartilham contexto de fato.

    PCA antes do KMeans porque a padronização deixa caudas longas em volumes do
    Censo, e uma única cidade extrema captura um cluster inteiro. k sai da maior
    silhueta entre 2 e 6, ignorando soluções com grupo de menos de 1% dos casos.
    """
    numericas = [c for c in ESTRUTURAIS if c in unidades and unidades[c].dtype != object]
    Z = transformador_numerico(numericas).fit_transform(unidades[numericas])
    Z = PCA(n_components=10, random_state=semente).fit_transform(Z)
    minimo = max(2, int(.01 * len(unidades)))
    melhor = None
    for k in range(2, 7):
        rotulos = KMeans(n_clusters=k, n_init=10, random_state=semente).fit_predict(Z)
        tamanhos = np.bincount(rotulos, minlength=k)
        if len(np.unique(rotulos)) < k or tamanhos.min() < minimo:
            continue
        s = float(silhouette_score(Z, rotulos))
        if melhor is None or s > melhor[0]:
            melhor = (s, k, rotulos)
    if melhor is None:
        raise ValueError("Nenhum agrupamento com grupos de tamanho utilizável")
    silhueta, k, rotulos = melhor
    descricao = unidades.assign(perfil=rotulos).groupby("perfil").agg(
        territorios=(ALVO, "size"), taxa_media=("taxa_alfabetizacao", "mean"),
        pct_em_risco=(ALVO, "mean"), idhm=("idhm_municipio", "mean"),
        pct_rural=("pct_escolas_rurais_censo", "mean"),
        pct_internet=("pct_escolas_internet_censo", "mean"),
        pct_biblioteca=("pct_escolas_biblioteca_sala_leitura_censo", "mean"),
        populacao_mediana=("populacao_municipio_ibge", "median")).reset_index()
    regiao = pd.crosstab(rotulos, unidades.regiao_brasil, normalize="index").mul(100).round(1)
    descricao["regiao_predominante"] = regiao.idxmax(axis=1).to_numpy()
    descricao["pct_da_regiao_predominante"] = regiao.max(axis=1).to_numpy()
    return {"k": k, "silhueta": silhueta,
            "criterio": "maior silhueta entre k de 2 a 6, exigindo grupo minimo de 1%",
            "variaveis": ("somente contexto estrutural numerico, sobre 10 componentes principais; "
                          "UF, regiao, rede e o resultado observado ficam fora do agrupamento"),
            "grupos": descricao.to_dict("records")}


def distancia_da_meta(unidades, score):
    """Meta municipal tem abrangência Municipal; outras redes ficam de fora."""
    alvo = unidades.rede.eq("Municipal") & unidades.meta_alfabetizacao_municipio.notna()
    m = unidades.loc[alvo].assign(score_risco=score[alvo.to_numpy()])
    if m.empty:
        return {"territorios_com_meta": 0}
    meta = m.meta_alfabetizacao_municipio / ESCALA_PERCENTUAL
    if not meta.between(0, 1).all():
        raise ValueError("Meta fora de 0 a 100: escala do contrato Gold mudou")
    m["abaixo_da_meta"] = m.taxa_alfabetizacao.lt(meta)
    m["distancia"] = m.taxa_alfabetizacao - meta
    faixas = pd.qcut(m.score_risco, 5, labels=[f"Q{i}" for i in range(1, 6)])
    por_faixa = m.groupby(faixas, observed=True).agg(
        territorios=("abaixo_da_meta", "size"), pct_abaixo_da_meta=("abaixo_da_meta", "mean"),
        distancia_media=("distancia", "mean"), taxa_media=("taxa_alfabetizacao", "mean")).reset_index()
    por_faixa.columns = ["faixa_de_score"] + list(por_faixa.columns[1:])
    auc = (float(roc_auc_score(m.abaixo_da_meta, m.score_risco))
           if m.abaixo_da_meta.nunique() == 2 else None)
    return {"territorios_com_meta": int(len(m)),
            "pct_abaixo_da_meta": float(m.abaixo_da_meta.mean()),
            "distancia_mediana": float(m.distancia.median()),
            "auc_do_score_para_abaixo_da_meta": auc,
            "por_faixa_de_score": por_faixa.to_dict("records"),
            "leitura": ("O score ordena quem fica abaixo da meta da própria edição. "
                        "Não é previsão de meta futura: exigiria comprovar a data de "
                        "publicação de cada meta e validar o erro fora do tempo.")}


def executar_aplicacao(lake, execucao, ano_teste=2025):
    config.preparar_diretorios()
    configurar()
    inicio = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_aplicacao"
    pacote, hash_modelo = carregar_modelo()
    unidades, proveniencia = territorios_do_teste(lake, ano_teste, execucao)
    modelo = pacote["pipeline"]
    score = modelo.predict_proba(unidades[FEATURES])[:, list(modelo.classes_).index(1)]
    print(f"[APLICACAO] {len(unidades)} territorios de {ano_teste} pontuados pelo modelo "
          f"congelado em {pacote['run_id']}", flush=True)
    saida = {"run_id": run_id, "ano_teste": ano_teste, "execucao_gold": execucao,
             "modelo": {"run_id": pacote["run_id"], "sha256": hash_modelo,
                        "corte_desenvolvimento": pacote["corte_desenvolvimento"]},
             "proveniencia": proveniencia["fontes"],
             "filtro": f"avaliacoes >= {MINIMO_AVALIACOES} e escolas >= {MINIMO_ESCOLAS}",
             "territorios": int(len(unidades)),
             "avaliacoes_cobertas": int(unidades.avaliacoes.sum()),
             "prioridades": priorizar(unidades, score),
             "precisao_no_topo": precisao_no_topo(unidades, score),
             "perfis": perfis_territoriais(unidades),
             "metas": distancia_da_meta(unidades, score),
             "limites": [
                 "Score ordena risco; não é estimativa calibrada da taxa do território.",
                 "Prioridade é ponto de partida para diagnóstico local, não corte de recurso.",
                 "Meta municipal vale para a rede Municipal; outras redes ficam fora.",
                 "Agrupamento descreve contexto; não implica política comum.",
                 "Nenhum resultado autoriza classificar uma criança."]}
    saida["duracao_segundos"] = round(time.monotonic() - inicio, 1)
    salvar_json(config.RELATORIOS / "aplicacao_territorial.json", saida)
    salvar_json(config.RELATORIOS / "runs" / run_id / "aplicacao.json", saida)
    grafico_prioridades(saida)
    return saida


def grafico_prioridades(saida):
    p = pd.DataFrame(saida["prioridades"]).head(15).iloc[::-1]
    nomes = p.id_municipio_nome + "/" + p.sigla_uf + " · " + p.rede
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(nomes, p.score_risco, color=LARANJA, label="Score de risco previsto")
    ax.plot(1 - p.taxa_alfabetizacao, nomes, "o", color=AZUL, label="Não alfabetizados observados")
    ax.set(xlabel="Score de risco (0-1) e fração observada", xlim=(0, 1),
           title=f"Territorios prioritarios em {saida['ano_teste']} — diagnostico, nao corte de recurso")
    ax.legend(loc="lower right", fontsize=8)
    salvar(fig, config.IMAGENS / "aplicacao_prioridades.png")
