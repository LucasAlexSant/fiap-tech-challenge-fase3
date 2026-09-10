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
    "taxa_alfabetizacao_municipio_lag2", "taxa_presenca_municipio_lag2",
    "delta_taxa_alfabetizacao_municipio", "delta_taxa_presenca_municipio",
    "populacao_municipio_ibge", "escolas_anos_iniciais_censo", "matriculas_anos_iniciais_censo",
    "vinculos_docentes_anos_iniciais_censo", "turmas_anos_iniciais_censo",
    "pct_escolas_rurais_censo", "pct_escolas_internet_censo", "pct_escolas_agua_potavel_censo",
    "pct_escolas_biblioteca_censo", "pct_escolas_lab_informatica_censo",
    "pct_escolas_sala_leitura_censo", "pct_escolas_quadra_esportes_censo",
    "pct_escolas_energia_rede_publica_censo", "pct_escolas_computador_censo",
    "pct_escolas_agua_rede_publica_censo", "pct_escolas_agua_poco_artesiano_censo",
    "pct_escolas_agua_cacimba_censo", "pct_escolas_agua_fonte_rio_censo", "pct_escolas_agua_inexistente_censo",
    "pct_escolas_energia_gerador_fossil_censo", "pct_escolas_energia_renovavel_censo", "pct_escolas_energia_inexistente_censo",
    "pct_escolas_esgoto_rede_publica_censo", "pct_escolas_esgoto_fossa_septica_censo",
    "pct_escolas_esgoto_fossa_comum_censo", "pct_escolas_esgoto_fossa_censo", "pct_escolas_esgoto_inexistente_censo",
    "pct_escolas_biblioteca_sala_leitura_censo", "pct_escolas_quadra_esportes_coberta_censo",
    "pct_escolas_quadra_esportes_descoberta_censo", "pct_escolas_acessibilidade_corrimao_censo",
    "pct_escolas_acessibilidade_elevador_censo", "pct_escolas_acessibilidade_pisos_tateis_censo",
    "pct_escolas_acessibilidade_vao_livre_censo", "pct_escolas_acessibilidade_rampas_censo",
    "pct_escolas_acessibilidade_sinal_sonoro_censo", "pct_escolas_acessibilidade_sinal_tatil_censo",
    "pct_escolas_acessibilidade_sinal_visual_censo", "pct_escolas_internet_alunos_censo",
    "pct_escolas_internet_administrativo_censo", "pct_escolas_internet_aprendizagem_censo",
    "pct_escolas_internet_comunidade_censo", "pct_escolas_acesso_internet_computador_censo",
    "pct_escolas_com_professores_censo", "pct_escolas_com_prof_administrativos_censo",
    "pct_escolas_com_prof_alimentacao_censo", "pct_escolas_com_prof_assist_social_censo",
    "pct_escolas_com_prof_bibliotecario_censo", "pct_escolas_com_prof_coordenador_censo",
    "pct_escolas_com_prof_fonoaudiologo_censo", "pct_escolas_com_prof_gestao_censo",
    "pct_escolas_com_prof_monitores_censo", "pct_escolas_com_prof_nutricionista_censo",
    "pct_escolas_com_prof_pedagogia_censo", "pct_escolas_com_prof_psicologo_censo",
    "pct_escolas_com_prof_saude_censo", "pct_escolas_material_ped_multimidia_censo",
    "pct_escolas_material_ped_infantil_censo", "pct_escolas_material_ped_cientifico_censo",
    "pct_escolas_material_ped_jogos_censo",
    "salas_utilizadas_censo", "salas_climatizadas_censo", "salas_acessiveis_censo", "salas_utilizadas_fora_censo",
    "matriculas_por_vinculo_docente_censo", "matriculas_por_turma_censo",
    "idhm_municipio", "idhm_educacao_municipio", "idhm_longevidade_municipio", "idhm_renda_municipio",
]
FEATURES = config.FEATURES + NOVAS_NUMERICAS
IDHM_NUMERICAS = ["idhm_municipio", "idhm_educacao_municipio", "idhm_longevidade_municipio", "idhm_renda_municipio"]
HISTORICAS_LAG2 = ["taxa_alfabetizacao_municipio_lag2", "taxa_presenca_municipio_lag2", "delta_taxa_alfabetizacao_municipio", "delta_taxa_presenca_municipio"]


def validar_contextos(df):
    obrigatorias = NOVAS_NUMERICAS + ["ano_referencia_ibge", "ano_referencia_censo", "tem_populacao_ibge", "tem_censo_escolar"]
    if faltantes := set(obrigatorias) - set(df):
        raise ValueError(f"Gold sem enriquecimento: {sorted(faltantes)}")
    colunas_censo = [c for c in NOVAS_NUMERICAS[1:] if c not in IDHM_NUMERICAS + HISTORICAS_LAG2]
    for ref, colunas in [("ano_referencia_ibge", [NOVAS_NUMERICAS[0]]), ("ano_referencia_censo", colunas_censo)]:
        referencia = pd.to_numeric(df[ref], errors="raise")
        if (referencia.notna() & (referencia.ge(df.ano) | referencia.le(0))).any():
            raise ValueError(f"Referência contemporânea/futura ou inválida: {ref}")
        if ref.endswith("censo") and (referencia.notna() & referencia.ne(df.ano - 1)).any():
            raise ValueError("Censo deve corresponder ao ano anterior")
        if (referencia.isna() & df[colunas].notna().any(axis=1)).any():
            raise ValueError("Medida sem referência temporal")
    if "ano_referencia_idhm" in df:
        referencia_idhm = pd.to_numeric(df["ano_referencia_idhm"], errors="raise")
        if (referencia_idhm.notna() & (referencia_idhm.ge(df.ano) | referencia_idhm.le(0))).any():
            raise ValueError("ReferÃªncia IDHM contemporÃ¢nea/futura ou invÃ¡lida")
        if (referencia_idhm.isna() & df[IDHM_NUMERICAS].notna().any(axis=1)).any():
            raise ValueError("Medida IDHM sem referÃªncia temporal")
    if "ano_referencia_historico_municipio_lag2" in df:
        ref_lag2 = pd.to_numeric(df["ano_referencia_historico_municipio_lag2"], errors="raise")
        if (ref_lag2.notna() & (ref_lag2.ge(df.ano) | ref_lag2.le(0) | ref_lag2.ne(df.ano - 2))).any():
            raise ValueError("ReferÃªncia histÃ³rica de segundo atraso invÃ¡lida")
        if (ref_lag2.isna() & df[HISTORICAS_LAG2].notna().any(axis=1)).any():
            raise ValueError("Feature histórica de segundo atraso sem referência")
    for c in NOVAS_NUMERICAS:
        v = pd.to_numeric(df[c], errors="raise").to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(v).any() or (not c.startswith("delta_") and (v < 0).any()) or (c.startswith("pct_") and (v > 100).any()):
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
