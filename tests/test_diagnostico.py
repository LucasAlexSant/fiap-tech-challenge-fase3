import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src import config
from src.evaluation.insights import associacoes, pilar_de
from src.evaluation.metricas_ponderadas import avaliar_ponderado, limiar_de_maior_acuracia
from src.modeling.amostragem import aceita_peso, ajustar, normalizar
from src.modeling.municipal import ALVO, ESTRUTURAIS, HISTORICO, avaliar_nas_duas_regras
from src.preprocessing.enriquecida import FEATURES
from src.preprocessing.unidades import CHAVE_UNIDADE, colapsar, expandir_para_avaliacoes, verificar_colapso


def base_sintetica():
    """Duas unidades município x rede; preditores constantes dentro de cada uma."""
    linhas = []
    for municipio, rede, n, alfabetizados, taxa in [("1", "Municipal", 5, 3, .4), ("2", "Estadual", 4, 1, .9)]:
        for i in range(n):
            linhas.append({"id_municipio": municipio, "rede": rede, "id_escola": f"{municipio}-{i % 2}",
                           "id_municipio_nome": f"Cidade {municipio}", "sigla_uf": "SP",
                           "regiao_brasil": "Sudeste", "tem_historico": True,
                           config.ALVO: 1 if i < alfabetizados else 0,
                           "taxa_alfabetizacao_municipio_anterior": taxa,
                           **{c: 1. for c in FEATURES if c not in
                              ("rede", "sigla_uf", "regiao_brasil", "taxa_alfabetizacao_municipio_anterior")}})
    return pd.DataFrame(linhas)


def test_colapso_preserva_todas_as_avaliacoes_e_o_alvo():
    base = base_sintetica()
    assert verificar_colapso(base) == 2
    unidades = colapsar(base)
    assert len(unidades) == 2
    assert unidades.avaliacoes.sum() == len(base)
    assert unidades.n_alfabetizados.sum() == base[config.ALVO].sum()
    assert list(unidades.columns).count("sigla_uf") == 1
    np.testing.assert_allclose(sorted(unidades.taxa_alfabetizacao), [.25, .6])


def test_colapso_rejeita_preditor_que_varia_dentro_da_unidade():
    base = base_sintetica()
    base.loc[0, "taxa_alfabetizacao_municipio_anterior"] = .99
    with pytest.raises(ValueError, match="variam dentro"):
        verificar_colapso(base)


def test_metricas_ponderadas_reproduzem_as_metricas_por_avaliacao():
    base = base_sintetica()
    unidades = colapsar(base)
    y, peso, indice = expandir_para_avaliacoes(unidades)
    assert peso.sum() == len(base)
    # Score constante por unidade, como qualquer modelo sobre estes preditores.
    score_unidade = np.array([.8, .2])
    ponderada = avaliar_ponderado(y, score_unidade[indice], peso)
    ordenada = base.sort_values(CHAVE_UNIDADE)
    expandida = np.repeat(score_unidade, unidades.avaliacoes.to_numpy())
    bruta = avaliar_ponderado(ordenada[config.ALVO].to_numpy(), expandida, np.ones(len(base)))
    assert np.isclose(ponderada["acuracia"], bruta["acuracia"])
    assert np.isclose(ponderada["roc_auc"], bruta["roc_auc"])
    assert np.isclose(ponderada["f1_macro"], bruta["f1_macro"])


def test_limiar_de_maior_acuracia_nao_piora_o_limiar_padrao():
    y = np.array([0, 0, 1, 1])
    p = np.array([.9, .6, .55, .1])
    limiar, acuracia = limiar_de_maior_acuracia(y, p, np.ones(4))
    assert acuracia >= avaliar_ponderado(y, p, np.ones(4))["acuracia"]
    assert .05 <= limiar <= .95


def test_replica_substitui_peso_apenas_quando_o_estimador_nao_o_aceita():
    assert aceita_peso(LogisticRegression())
    assert not aceita_peso(LinearDiscriminantAnalysis())
    X = pd.DataFrame({"x": [0., .2, .8, 1.]})
    y = [0, 0, 1, 1]
    _, modo = ajustar(LogisticRegression(), X, y, [10., 10., 90., 90.])
    assert modo["ajuste"] == "sample_weight_normalizado"
    # Réplica proporcional 1:9: a classe 1 domina o ajuste apesar de ter as mesmas duas linhas.
    # O orçamento nunca excede o peso total, então 400 pedidos viram as 200 avaliações existentes.
    modelo, modo = ajustar(LinearDiscriminantAnalysis(), X, y, [10., 10., 90., 90.], max_replicas=400)
    assert modo == {"ajuste": "replica_proporcional", "linhas_replicadas": 200}
    assert modelo.predict_proba(pd.DataFrame({"x": [.5]}))[0, 1] > .5


def test_regra_de_posto_marca_metade_dos_territorios():
    y = np.array([1, 1, 0, 0])
    p = np.array([.9, .8, .7, .6])  # todos acima de 0,5: o limiar fixo marca todos
    saida = avaliar_nas_duas_regras(y, p)
    assert saida["limiar_meio"]["acuracia"] == .5
    assert saida["regra_de_posto"]["acuracia"] == 1.
    assert np.isclose(saida["regra_de_posto"]["limiar_usado"], .75)


def test_bloco_estrutural_remove_apenas_o_historico_de_alfabetizacao():
    assert set(HISTORICO) <= set(FEATURES)
    assert set(ESTRUTURAIS) == set(FEATURES) - set(HISTORICO)
    assert "idhm_educacao_municipio" in ESTRUTURAIS
    assert ALVO not in FEATURES


def test_pilares_cobrem_todos_os_preditores_do_contrato():
    pilares = {c: pilar_de(c) for c in FEATURES}
    assert "Outros" not in set(pilares.values()), [c for c, p in pilares.items() if p == "Outros"]
    assert pilares["idhm_renda_municipio"] == "Desenvolvimento humano"
    assert pilares["pct_escolas_agua_potavel_censo"] == "Infraestrutura básica"
    assert pilares["pct_escolas_com_prof_psicologo_censo"] == "Recursos humanos na escola"


def test_correlacao_parcial_zera_associacao_que_e_so_o_controle():
    rng = np.random.default_rng(0)
    idhm = rng.normal(size=400)
    unidades = pd.DataFrame({
        "idhm_municipio": idhm, "populacao_municipio_ibge": rng.normal(size=400),
        "taxa_alfabetizacao": idhm + .01 * rng.normal(size=400),
        # Espelha o controle: a associação bruta é forte e a parcial some.
        "taxa_presenca_municipio_anterior": idhm + .01 * rng.normal(size=400),
        "alunos_avaliados_municipio_anterior": rng.normal(size=400)})
    saida = associacoes(unidades, ["taxa_presenca_municipio_anterior", "alunos_avaliados_municipio_anterior"],
                        ["idhm_municipio", "populacao_municipio_ibge"]).set_index("variavel")
    assert saida.loc["taxa_presenca_municipio_anterior", "spearman_bruto"] > .9
    assert abs(saida.loc["taxa_presenca_municipio_anterior", "spearman_parcial"]) < .3
    assert abs(saida.loc["alunos_avaliados_municipio_anterior", "spearman_bruto"]) < .2


def test_normalizacao_preserva_a_proporcao_entre_unidades():
    peso = np.array([10., 90., 400.])
    normalizado = normalizar(peso)
    assert np.isclose(normalizado.mean(), 1.)
    np.testing.assert_allclose(normalizado / normalizado.sum(), peso / peso.sum())
    with pytest.raises(ValueError, match="positivo"):
        normalizar(np.zeros(3))
