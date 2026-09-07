import pandas as pd
import pytest

from src import config
from src.preprocessing.base import ler_gold, validar_gold


def exemplo():
    df = pd.DataFrame({
        "ano": [2024, 2024], "id_municipio": ["1100015"] * 2,
        "id_escola": ["1", "2"], "id_aluno": ["1", "2"], "serie": ["2 ano"] * 2,
        "rede": ["Municipal"] * 2, "sigla_uf": ["RO"] * 2, "regiao_brasil": ["Norte"] * 2,
        "elegivel_modelagem": [True, False], "motivo_exclusao_modelagem": ["elegivel", "ausente"],
        config.ALVO: pd.Series([1, None], dtype="Int8"),
    })
    for c in config.NUMERICAS:
        df[c] = 50.0
    for c in config.REFERENCIAS:
        df[c] = 2023
    return df


def test_rejeita_referencia_da_mesma_safra():
    df = exemplo()
    df[config.REFERENCIAS[0]] = 2024
    with pytest.raises(ValueError, match="ano anterior"):
        validar_gold(df)


def test_ausente_nao_vira_classe_negativa():
    df = exemplo()
    validar_gold(df)
    df.loc[1, config.ALVO] = 0
    with pytest.raises(ValueError, match="inelegível"):
        validar_gold(df)


def test_nao_regride_snapshot_quando_falta_ano(tmp_path):
    tabela = tmp_path / "gold" / config.TABELA_GOLD
    antigo = tabela / "execution_date=2026-09-01/ano=2024"
    antigo.mkdir(parents=True)
    exemplo().to_parquet(antigo / "a.parquet", index=False)
    (tabela / "execution_date=2026-09-07").mkdir()
    with pytest.raises(FileNotFoundError, match="Safra"):
        ler_gold(tmp_path, 2024)
    df, execucao, fontes = ler_gold(tmp_path, 2024, "2026-09-01")
    assert len(df) == 2 and execucao == "2026-09-01"
    assert len(fontes[0]["sha256"]) == 64


def test_colunas_extras_nao_entram_automaticamente_no_modelo():
    df = exemplo()
    df["proficiencia"] = [900, None]
    validar_gold(df)
    assert "proficiencia" not in config.FEATURES
    assert not set(config.FEATURES).intersection(config.PROIBIDAS)


def test_presenca_anterior_sem_avaliacao_valida_e_legitima():
    df = exemplo()
    df[config.REFERENCIAS[0]] = float("nan")
    df[config.NUMERICAS[0]] = float("nan")
    df[config.NUMERICAS[2]] = float("nan")
    df["taxa_presenca_municipio_anterior"] = 0.0
    validar_gold(df)
