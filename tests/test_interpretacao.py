import numpy as np
import pandas as pd

from src import config
from src.artefatos import ler_json
from src.evaluation.interpretabilidade import diagnostico_territorial, nome_original


def test_mapeia_indicador_de_ausencia_para_variavel_original():
    assert nome_original("taxas__missingindicator_taxa_alfabetizacao_municipio_anterior") == config.NUMERICAS[0]
    assert nome_original("categorias__sigla_uf_RO") == "sigla_uf"


def test_ranking_respeita_rede_e_nao_estima_taxa_com_score(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "BASE", tmp_path / "base.parquet")
    monkeypatch.setattr(config, "RELATORIOS", tmp_path)
    monkeypatch.setattr(config, "IMAGENS", tmp_path)
    df = pd.DataFrame({"ano": [2024] * 240, "sigla_uf": ["RO"] * 240,
                       "id_municipio": ["1100015"] * 240, "id_municipio_nome": ["Cidade"] * 240,
                       "rede": ["Municipal"] * 120 + ["Estadual"] * 120,
                       "id_escola": np.tile(np.repeat(["1", "2", "3"], 40), 2),
                       config.ALVO: np.tile([0, 1], 120), "meta_alfabetizacao_municipio": [70.] * 240})
    diagnostico_territorial(df, np.full(240, .6))
    r = ler_json(tmp_path / "priorizacao_amostra_teste.json")
    assert len(r["top20"]) == 2
    estadual = next(linha for linha in r["top20"] if linha["rede"] == "Estadual")
    assert estadual["meta_municipal_contextual"] is None
    assert "distancia_prevista_meta_pp_amostra" not in estadual
