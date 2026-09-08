"""Comandos reproduzíveis da Fase 3."""
import argparse
import os
from pathlib import Path

from src import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etapa", choices=["base", "eda", "treinar", "interpretar", "relatorio", "tudo"])
    parser.add_argument("--lake", type=Path, default=config.LAKE)
    parser.add_argument("--ano", type=int, default=config.ANO_PADRAO)
    parser.add_argument("--execution-date")
    parser.add_argument("--max-busca", type=int, default=150000)
    parser.add_argument("--dobras", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sem-busca", action="store_true")
    parser.add_argument("--amostra-interpretacao", type=int, default=5000)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads deve ser positivo")
    # Evita a consulta de núcleos físicos via subprocesso no Windows,
    # cuja saída localizada pode não ser UTF-8. Não altera os dados/modelos.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(args.threads))
    if args.etapa in ["base", "tudo"]:
        from src.preprocessing.base import preparar_base
        preparar_base(args.lake, args.ano, args.execution_date)
    if args.etapa in ["eda", "tudo"]:
        from src.visualization.exploratoria import executar_eda
        executar_eda()
    if args.etapa in ["treinar", "tudo"]:
        from src.modeling.treinar import treinar
        treinar(args.max_busca, args.dobras, args.sem_busca, args.threads)
    if args.etapa in ["interpretar", "tudo"]:
        from src.evaluation.interpretabilidade import interpretar
        interpretar(amostra=args.amostra_interpretacao, threads=args.threads)
    if args.etapa in ["relatorio", "tudo"]:
        from src.evaluation.relatorio import gerar_relatorio
        gerar_relatorio()


if __name__ == "__main__":
    main()
