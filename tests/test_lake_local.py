"""A Gold versionada precisa bater com o manifesto publicado, não só existir."""
import os

import pytest

from src import config
from src.artefatos import ler_json, sha256
from src.preprocessing.enriquecida import NOVAS_NUMERICAS, TABELA

LAKE_LOCAL = config.RAIZ / "data/lake"
EXECUCAO_ENRIQUECIDA = "2026-09-08"
EXECUCAO_BASE = "2026-09-07"


def test_lake_padrao_fica_dentro_do_repositorio():
    """Sem FASE2_LAKE_PATH, nada deve depender da Fase 2 ao lado."""
    if os.getenv("FASE2_LAKE_PATH"):
        pytest.skip("FASE2_LAKE_PATH definido: o padrão do repositório não está em uso")
    assert config.LAKE == LAKE_LOCAL
    assert config.RAIZ in config.LAKE.parents


def test_tabelas_lidas_pela_fase3_estao_versionadas():
    assert (LAKE_LOCAL / "gold" / config.TABELA_GOLD / f"execution_date={EXECUCAO_BASE}").is_dir()
    enriquecida = LAKE_LOCAL / "gold" / TABELA / f"execution_date={EXECUCAO_ENRIQUECIDA}"
    assert enriquecida.is_dir()
    for ano in (2023, 2024, 2025):
        assert list((enriquecida / f"ano={ano}").glob("*.parquet")), f"edição {ano} ausente"


def test_copia_local_bate_com_o_manifesto_publicado():
    """Mesma verificação que `carregar_enriquecida` faz antes de ler."""
    pasta = LAKE_LOCAL / "gold" / TABELA / f"execution_date={EXECUCAO_ENRIQUECIDA}"
    manifesto = ler_json(pasta / "manifesto_enriquecimento.json")
    assert manifesto["execution_date"] == EXECUCAO_ENRIQUECIDA
    assert manifesto["novas_features"] == NOVAS_NUMERICAS
    registrados = {p["arquivo"].replace("\\", "/"): p["sha256"] for p in manifesto["arquivos_gold"]}
    conferidos = 0
    for arquivo in sorted(pasta.rglob("*.parquet")):
        relativo = arquivo.relative_to(LAKE_LOCAL).as_posix()
        assert relativo in registrados, f"arquivo fora do manifesto: {relativo}"
        assert sha256(arquivo) == registrados[relativo], f"cópia diverge da origem: {relativo}"
        conferidos += 1
    assert conferidos == 3
