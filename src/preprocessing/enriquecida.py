"""Contrato aditivo para Gold IBGE/Censo, independente do código da Fase 2."""
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import config
from src.artefatos import ler_json, sha256
from src.preprocessing.base import validar_gold


TABELA = "base_modelagem_aluno_enriquecida"
NOVAS_NUMERICAS = [
    "populacao_municipio_ibge", "escolas_anos_iniciais_censo", "matriculas_anos_iniciais_censo",
    "vinculos_docentes_anos_iniciais_censo", "turmas_anos_iniciais_censo",
    "pct_escolas_rurais_censo", "pct_escolas_internet_censo", "pct_escolas_agua_potavel_censo",
    "pct_escolas_biblioteca_censo", "pct_escolas_lab_informatica_censo",
    "pct_escolas_sala_leitura_censo", "pct_escolas_quadra_esportes_censo",
    "pct_escolas_energia_rede_publica_censo", "pct_escolas_computador_censo",
    "matriculas_por_vinculo_docente_censo", "matriculas_por_turma_censo",
]
FEATURES = config.FEATURES + NOVAS_NUMERICAS


def validar_contextos(df):
    obrigatorias = NOVAS_NUMERICAS + ["ano_referencia_ibge", "ano_referencia_censo", "tem_populacao_ibge", "tem_censo_escolar"]
    if faltantes := set(obrigatorias) - set(df):
        raise ValueError(f"Gold sem enriquecimento: {sorted(faltantes)}")
    for ref, colunas in [("ano_referencia_ibge", [NOVAS_NUMERICAS[0]]), ("ano_referencia_censo", NOVAS_NUMERICAS[1:])]:
        referencia = pd.to_numeric(df[ref], errors="raise")
        if (referencia.notna() & (referencia.ge(df.ano) | referencia.le(0))).any():
            raise ValueError(f"Referência contemporânea/futura ou inválida: {ref}")
        if ref.endswith("censo") and (referencia.notna() & referencia.ne(df.ano - 1)).any():
            raise ValueError("Censo deve corresponder ao ano anterior")
        if (referencia.isna() & df[colunas].notna().any(axis=1)).any():
            raise ValueError("Medida sem referência temporal")
    for c in NOVAS_NUMERICAS:
        v = pd.to_numeric(df[c], errors="raise").to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(v).any() or (v < 0).any() or (c.startswith("pct_") and (v > 100).any()):
            raise ValueError(f"Medida de contexto inválida: {c}")
    for marcador, medida in [("tem_populacao_ibge", "populacao_municipio_ibge"), ("tem_censo_escolar", "escolas_anos_iniciais_censo")]:
        if df[marcador].isna().any() or not df[marcador].eq(df[medida].notna()).all():
            raise ValueError("Marcador de cobertura inconsistente")


def carregar_enriquecida(lake, ano, execucao):
    date.fromisoformat(execucao)
    pasta = lake / "gold" / TABELA / f"execution_date={execucao}"
    manifesto = ler_json(pasta / "manifesto_enriquecimento.json")
    if manifesto["execution_date"] != execucao or manifesto["novas_features"] != NOVAS_NUMERICAS:
        raise ValueError("Manifesto/contrato da Gold enriquecida diverge")
    arquivos = sorted((pasta / f"ano={ano}").glob("*.parquet"))
    if not arquivos:
        raise FileNotFoundError(f"Edição {ano} ausente da Gold enriquecida")
    hashes = {p["arquivo"].replace("\\", "/"): p["sha256"] for p in manifesto["arquivos_gold"]}
    quadros, fontes = [], []
    for arquivo in arquivos:
        relativo = arquivo.relative_to(lake).as_posix()
        digest = sha256(arquivo)
        if hashes.get(relativo) != digest:
            raise ValueError("Gold diverge do hash publicado")
        df = pq.ParquetFile(arquivo).read().to_pandas()
        if "ano" in df and not df.ano.eq(ano).all():
            raise ValueError("Ano interno diverge da partição")
        df["ano"] = ano
        quadros.append(df)
        fontes.append({"arquivo": relativo, "sha256": digest})
    base = pd.concat(quadros, ignore_index=True)
    validar_gold(base)
    validar_contextos(base)
    base = base.loc[base.elegivel_modelagem].sort_values(config.CHAVE).reset_index(drop=True)
    base[config.ALVO] = base[config.ALVO].astype("int8")
    for c in config.CATEGORICAS:
        base[c] = base[c].astype(object).where(base[c].notna(), np.nan)
    for c in config.NUMERICAS + NOVAS_NUMERICAS:
        base[c] = base[c].astype(float)
    return base, {"fontes": fontes, "manifesto_gold": manifesto}
