"""Contrato explícito de preditores: nenhuma descoberta automática de colunas."""
from pathlib import Path
import os

RAIZ = Path(__file__).resolve().parents[1]
# Cópia local das duas tabelas Gold que a Fase 3 lê, byte a byte idêntica à
# origem: o repositório roda sozinho, sem exigir a Fase 3 lado a lado com a
# Fase 2. Os hashes do manifesto são relativos à raiz do lake, então a árvore
# `gold/<tabela>/execution_date=.../ano=...` precisa ser preservada.
# FASE2_LAKE_PATH continua tendo precedência para apontar ao lake original.
LAKE = Path(os.getenv("FASE2_LAKE_PATH", RAIZ / "data/lake"))
TABELA_GOLD = "base_modelagem_aluno"
SEMENTE = 42
ANO_PADRAO = 2024
ALVO = "alfabetizado_binario"  # 1 alfabetizado, 0 não alfabetizado
CHAVE = ["ano", "id_municipio", "id_escola", "id_aluno", "serie", "rede"]
CATEGORICAS = ["rede", "sigla_uf", "regiao_brasil"]
NUMERICAS = [
    "taxa_alfabetizacao_municipio_anterior",
    "taxa_presenca_municipio_anterior",
    "alunos_avaliados_municipio_anterior",
    "total_pagamentos_bolsa_familia_anterior",
    "valor_total_bolsa_familia_anterior",
    "valor_medio_pagamento_bolsa_familia_anterior",
]
FEATURES = CATEGORICAS + NUMERICAS
REFERENCIAS = ["ano_referencia_historico_municipio", "ano_referencia_bolsa_familia"]
PROIBIDAS = {
    "proficiencia": "define o próprio alvo pelo corte 743",
    "presenca": "é critério de elegibilidade, conhecido após a prova",
    "elegivel_modelagem": "deriva da existência e validade da avaliação",
    "motivo_exclusao_modelagem": "deriva da avaliação e do rótulo",
    "peso_aluno": "não foi validado como disponível antes da avaliação",
    "meta_alfabetizacao_municipio": "publicação/revisões anteriores à previsão não comprovadas",
    "meta_alfabetizacao_uf": "publicação/revisões anteriores à previsão não comprovadas",
    "meta_alfabetizacao_brasil": "publicação/revisões anteriores à previsão não comprovadas",
    "taxa_alfabetizacao_escola_anterior": "códigos escolares não são estáveis entre anos",
    "status_meta": "deriva do resultado contemporâneo",
    "distancia_meta": "deriva do resultado contemporâneo",
}
BASE = RAIZ / "data/processed/base_analitica.parquet"
MANIFESTO = RAIZ / "data/processed/manifesto.json"
RELATORIOS = RAIZ / "reports"
IMAGENS = RAIZ / "images"
MODELOS = RAIZ / "modelos"


def preparar_diretorios():
    for pasta in [BASE.parent, RELATORIOS, IMAGENS, MODELOS]:
        pasta.mkdir(parents=True, exist_ok=True)
