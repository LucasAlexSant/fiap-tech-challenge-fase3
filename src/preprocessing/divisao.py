"""Holdout por escola dentro da edição; teste fica isolado da seleção."""
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src import config
from src.artefatos import salvar_json


def grupos_escola(base: pd.DataFrame) -> pd.Series:
    return base[["ano", "id_municipio", "id_escola"]].astype(str).agg("/".join, axis=1)


def dividir(base: pd.DataFrame, semente: int = config.SEMENTE):
    if base.ano.nunique() != 1:
        raise ValueError("Códigos de escola não podem identificar grupos entre edições")
    grupos = grupos_escola(base)
    if grupos.nunique() < 15:
        raise ValueError("São necessárias pelo menos 15 escolas para a divisão")
    todos = np.arange(len(base))
    desenvolvimento, teste = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=semente).split(todos, groups=grupos))
    treino_local, validacao_local = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=semente + 1).split(
        desenvolvimento, groups=grupos.iloc[desenvolvimento]))
    partes = {"treino": desenvolvimento[treino_local], "validacao": desenvolvimento[validacao_local], "teste": teste}
    conjuntos = {nome: set(grupos.iloc[idx]) for nome, idx in partes.items()}
    if any(conjuntos[a] & conjuntos[b] for a, b in [("treino", "validacao"), ("treino", "teste"), ("validacao", "teste")]):
        raise ValueError("Escolas compartilhadas entre partições")
    if sorted(np.concatenate(list(partes.values())).tolist()) != todos.tolist():
        raise ValueError("Partição não cobre cada linha exatamente uma vez")
    for nome, idx in partes.items():
        if base.iloc[idx][config.ALVO].nunique() != 2:
            raise ValueError(f"As duas classes precisam estar presentes em {nome}")
    return partes


def resumo_divisao(base, partes, base_sha256):
    resumo = {"semente": config.SEMENTE, "base_sha256": base_sha256,
              "unidade": "ano + municipio + escola", "proporcoes_alvo_escolas": [0.6, 0.2, 0.2],
              "sobreposicao_escolas": 0, "particoes": {}}
    grupos = grupos_escola(base)
    for nome, idx in partes.items():
        g = base.iloc[idx]
        resumo["particoes"][nome] = {
            "linhas": len(g), "escolas": grupos.iloc[idx].nunique(),
            "municipios": g.id_municipio.nunique(), "ufs": g.sigla_uf.nunique(),
            "proporcao_nao_alfabetizados": float(g[config.ALVO].eq(0).mean()),
            "indices_sha256": hashlib.sha256(np.asarray(idx, dtype="int64").tobytes()).hexdigest(),
        }
    salvar_json(config.RELATORIOS / "divisao.json", resumo)
    return resumo


def amostrar_escolas(base, indices, max_linhas, semente=config.SEMENTE):
    """Amostra somente no treino, conservando todas as linhas de cada escola."""
    if max_linhas is None or len(indices) <= max_linhas:
        return np.asarray(indices)
    if max_linhas <= 0:
        raise ValueError("Limite da busca deve ser positivo")
    grupos = grupos_escola(base.iloc[indices])
    tamanhos = grupos.value_counts(sort=False).sort_index()
    ordem = np.random.default_rng(semente).permutation(tamanhos.index.to_numpy())
    acumulado = tamanhos.loc[ordem].cumsum().to_numpy()
    n = max(1, int(np.searchsorted(acumulado, max_linhas, side="right")))
    escolhidos = set(ordem[:n])
    return np.asarray(indices)[grupos.isin(escolhidos).to_numpy()]
