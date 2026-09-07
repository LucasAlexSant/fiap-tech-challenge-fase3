import numpy as np
import pandas as pd
import pytest
from src import config
from src.preprocessing.divisao import amostrar_escolas, dividir, grupos_escola


def base_grupos():
    return pd.DataFrame({"ano": [2024] * 400, "id_municipio": ["1100015"] * 400,
                         "id_escola": np.repeat(np.arange(40).astype(str), 10),
                         config.ALVO: np.tile([0, 1], 200)})


def test_escolas_isoladas_e_reprodutibilidade():
    base = base_grupos()
    partes = dividir(base)
    repeticao = dividir(base)
    grupos = grupos_escola(base)
    for nome in partes:
        np.testing.assert_array_equal(partes[nome], repeticao[nome])
    assert len(set(np.concatenate(list(partes.values())))) == len(base)
    assert not set(grupos.iloc[partes['treino']]) & set(grupos.iloc[partes['teste']])
    assert not set(grupos.iloc[partes['validacao']]) & set(grupos.iloc[partes['teste']])


def test_amostra_preserva_escolas_inteiras_e_exclui_teste():
    base = base_grupos()
    partes = dividir(base)
    amostra = amostrar_escolas(base, partes["treino"], 105)
    assert len(amostra) == 100
    assert set(amostra) <= set(partes["treino"])
    assert base.iloc[amostra].groupby("id_escola").size().eq(10).all()


def test_nao_mistura_identificadores_entre_edicoes():
    base = base_grupos()
    base.loc[0, "ano"] = 2023
    with pytest.raises(ValueError, match="entre edições"):
        dividir(base)
