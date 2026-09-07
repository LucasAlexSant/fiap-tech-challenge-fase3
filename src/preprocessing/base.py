"""Leitura exclusiva da Gold, com validação de contrato e proveniência."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import config
from src.artefatos import ler_json, salvar_json, sha256


def localizar_execucao(lake: Path, execucao: str | None = None) -> Path:
    tabela = lake / "gold" / config.TABELA_GOLD
    if execucao:
        # Impede que um argumento de CLI escape da tabela configurada.
        datetime.strptime(execucao, "%Y-%m-%d")
        destino = tabela / f"execution_date={execucao}"
    else:
        pastas = sorted(p for p in tabela.glob("execution_date=*") if p.is_dir())
        if not pastas:
            raise FileNotFoundError(f"Gold ausente em {tabela}. Execute a Fase 2 ou configure FASE2_LAKE_PATH.")
        destino = pastas[-1]
    if not destino.is_dir():
        raise FileNotFoundError(destino)
    return destino


def ler_gold(lake: Path, ano: int, execucao: str | None = None):
    pasta = localizar_execucao(lake, execucao)
    # Nunca recua silenciosamente a um snapshot antigo se faltar a safra.
    arquivos = sorted((pasta / f"ano={ano}").glob("*.parquet"))
    if not arquivos:
        raise FileNotFoundError(f"Safra {ano} ausente em {pasta}")
    quadros, fontes = [], []
    for arquivo in arquivos:
        df = pq.ParquetFile(arquivo).read().to_pandas()
        if "ano" in df and not df.ano.eq(ano).all():
            raise ValueError("Ano interno diverge da partição")
        df["ano"] = ano
        quadros.append(df)
        fontes.append({"arquivo": str(arquivo.relative_to(lake)), "sha256": sha256(arquivo),
                       "bytes": arquivo.stat().st_size, "linhas": len(df)})
    return pd.concat(quadros, ignore_index=True), pasta.name.split("=", 1)[1], fontes


def validar_gold(df: pd.DataFrame):
    obrigatorias = set(config.CHAVE + config.FEATURES + config.REFERENCIAS +
                       [config.ALVO, "elegivel_modelagem", "motivo_exclusao_modelagem"])
    faltantes = obrigatorias - set(df)
    if faltantes:
        raise ValueError(f"Contrato Gold incompleto: {sorted(faltantes)}")
    if df.empty or df[config.CHAVE].isna().any().any() or df.duplicated(config.CHAVE).any():
        raise ValueError("Gold vazia, chave nula ou duplicada")
    if not pd.api.types.is_bool_dtype(df.elegivel_modelagem.dtype) or df.elegivel_modelagem.isna().any():
        raise ValueError("Elegibilidade deve ser booleana, preenchida")
    validos = df.loc[df.elegivel_modelagem]
    if validos.empty or not validos[config.ALVO].isin([0, 1]).all():
        raise ValueError("Alvo inválido em avaliação elegível")
    if df.loc[~df.elegivel_modelagem, config.ALVO].notna().any():
        raise ValueError("Avaliação inelegível não pode fornecer alvo")
    if not validos.motivo_exclusao_modelagem.eq("elegivel").all():
        raise ValueError("Motivo de elegibilidade inconsistente")
    if df["ano"].nunique() != 1:
        raise ValueError("Experimento por escola deve usar uma única edição")
    if not df.id_municipio.astype(str).str.fullmatch(r"\d{7}").all():
        raise ValueError("Código IBGE municipal inválido")
    for coluna in config.REFERENCIAS:
        ref = pd.to_numeric(df[coluna], errors="raise")
        if (ref.notna() & ref.ne(df.ano - 1)).any():
            raise ValueError(f"Referência deve corresponder ao ano anterior: {coluna}")
    pares = {
        # A Gold pode ter presença anterior (inclusive 0%) sem qualquer
        # avaliação válida no município. O marcador histórico acompanha
        # o agregado de desempenho, não o de presença.
        "ano_referencia_historico_municipio": [config.NUMERICAS[0], config.NUMERICAS[2]],
        "ano_referencia_bolsa_familia": config.NUMERICAS[3:],
    }
    for ref, colunas in pares.items():
        if (df[ref].isna() & df[colunas].notna().any(axis=1)).any():
            raise ValueError(f"Feature preenchida sem ano de referência: {ref}")
    for coluna in config.NUMERICAS:
        valores = pd.to_numeric(df[coluna], errors="raise").to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(valores).any() or (valores < 0).any():
            raise ValueError(f"Medida numérica inválida: {coluna}")
        if coluna.startswith("taxa_") and (valores > 100).any():
            raise ValueError(f"Taxa acima de 100: {coluna}")


def preparar_base(lake: Path = config.LAKE, ano: int = config.ANO_PADRAO,
                  execucao: str | None = None):
    config.preparar_diretorios()
    df, data_execucao, fontes = ler_gold(lake, ano, execucao)
    validar_gold(df)
    exclusoes = df.motivo_exclusao_modelagem.value_counts().to_dict()
    base = df.loc[df.elegivel_modelagem].copy()
    base[config.ALVO] = base[config.ALVO].astype("int8")
    for coluna in config.CATEGORICAS:
        base[coluna] = base[coluna].astype(object).where(base[coluna].notna(), np.nan)
    for coluna in config.NUMERICAS:
        base[coluna] = base[coluna].astype("float64")
    base = base.sort_values(config.CHAVE).reset_index(drop=True)
    base.to_parquet(config.BASE, index=False)
    manifesto = {
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(), "ano": ano,
        "execution_date_gold": data_execucao, "fonte_camada": "gold", "fontes": fontes,
        "linhas_gold": len(df), "linhas_elegiveis": len(base), "elegibilidade": exclusoes,
        "alvo": config.ALVO, "classe_risco": 0, "features": config.FEATURES,
        "features_proibidas": config.PROIBIDAS, "base_sha256": sha256(config.BASE),
    }
    salvar_json(config.MANIFESTO, manifesto)
    salvar_json(config.RELATORIOS / "proveniencia.json", manifesto)
    print(f"[BASE] Gold {data_execucao}, ano {ano}: {len(base):,} avaliações válidas de {len(df):,}")
    return base


def carregar_base():
    if not config.BASE.exists() or not config.MANIFESTO.exists():
        raise FileNotFoundError("Execute primeiro: python main.py base")
    manifesto = ler_json(config.MANIFESTO)
    if manifesto["features"] != config.FEATURES or sha256(config.BASE) != manifesto["base_sha256"]:
        raise ValueError("Base/contrato mudou: reconstrua com python main.py base")
    return pd.read_parquet(config.BASE), manifesto
