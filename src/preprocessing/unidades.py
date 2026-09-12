"""Colapso exato da base aluno em unidades município x rede.

Todos os 79 preditores do contrato enriquecido são constantes dentro de
município x rede: 1.851.852 avaliações de 2024 assumem apenas 6.536 valores
distintos de X. Treinar nas linhas únicas com pesos (n0, n1) reproduz a
importância relativa de cada unidade exatamente como na base inteira, e as
métricas ponderadas reproduzem exatamente as métricas por avaliação.
"""
import gc

import numpy as np
import pandas as pd

from src import config
from src.preprocessing.enriquecida import FEATURES, carregar_enriquecida

CHAVE_UNIDADE = ["id_municipio", "rede"]


def verificar_colapso(base, features=FEATURES):
    """Falha se algum preditor variar dentro de município x rede."""
    features = [c for c in features if c not in CHAVE_UNIDADE]
    variaveis = base.groupby(CHAVE_UNIDADE, observed=True)[features].nunique(dropna=False).max()
    if (instaveis := variaveis[variaveis > 1].index.tolist()):
        raise ValueError(f"Preditores variam dentro de município x rede: {instaveis}")
    return int(base.groupby(CHAVE_UNIDADE, observed=True).ngroups)


def colapsar(base, features=FEATURES, extras=()):
    """Uma linha por unidade, com contagens do alvo e agregados descritivos.

    Nenhuma operação copia a base inteira: `assign` e `drop_duplicates` sobre
    1,85 milhão de linhas e 115 colunas consolidam um bloco float64 de 1,1 GiB e
    estouram a memória. O alvo é 0/1, então a soma já conta os alfabetizados, e
    `head(1)` recupera a linha representativa de cada unidade antes de qualquer
    seleção de colunas.
    """
    agrupada = base.groupby(CHAVE_UNIDADE, observed=True)
    contagens = agrupada.agg(avaliacoes=(config.ALVO, "size"), escolas=("id_escola", "nunique"),
                             n_alfabetizados=(config.ALVO, "sum"),
                             id_municipio_nome=("id_municipio_nome", "first")).reset_index()
    contagens["n_nao_alfabetizados"] = contagens.avaliacoes - contagens.n_alfabetizados
    # `rede` é chave e preditor ao mesmo tempo: mantém-se uma única cópia.
    descritivas = list(dict.fromkeys(
        c for c in features + ["sigla_uf", "regiao_brasil", "tem_historico"] + list(extras)
        if c not in CHAVE_UNIDADE))
    unidades = agrupada.head(1)[CHAVE_UNIDADE + descritivas]
    unidades = unidades.merge(contagens, on=CHAVE_UNIDADE, validate="one_to_one")
    unidades["taxa_alfabetizacao"] = unidades.n_alfabetizados / unidades.avaliacoes
    if unidades.avaliacoes.sum() != len(base):
        raise ValueError("Colapso perdeu avaliações")
    return unidades


def expandir_para_avaliacoes(unidades):
    """Par (y, peso) por unidade: métricas ponderadas == métricas por avaliação."""
    y = np.tile([0, 1], len(unidades))
    peso = np.empty(2 * len(unidades), dtype=float)
    peso[0::2] = unidades.n_nao_alfabetizados.to_numpy()
    peso[1::2] = unidades.n_alfabetizados.to_numpy()
    indice = np.repeat(np.arange(len(unidades)), 2)
    return y, peso, indice


def carregar_unidades(lake, ano, execucao, features=FEATURES, extras=()):
    base, proveniencia = carregar_enriquecida(lake, ano, execucao)
    # Extras entram na mesma verificação: um valor que variasse dentro da
    # unidade seria silenciosamente reduzido ao primeiro pelo colapso.
    grupos = verificar_colapso(base, list(features) + [c for c in extras if c in base])
    unidades = colapsar(base, features, extras)
    proveniencia["avaliacoes"] = len(base)
    proveniencia["unidades"] = grupos
    proveniencia["ano"] = ano
    # A base bruta ocupa alguns GB e cada etapa carrega duas edicoes. Sem a
    # liberacao explicita o heap acumula ao longo do comando `diagnostico` e a
    # etapa seguinte fica sem memoria.
    del base
    gc.collect()
    return unidades, proveniencia
