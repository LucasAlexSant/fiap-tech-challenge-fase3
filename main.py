"""Comandos reproduzíveis da Fase 3."""
import argparse
from pathlib import Path

from src import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etapa", choices=["base", "eda"])
    parser.add_argument("--lake", type=Path, default=config.LAKE)
    parser.add_argument("--ano", type=int, default=config.ANO_PADRAO)
    parser.add_argument("--execution-date")
    args = parser.parse_args()
    if args.etapa == "base":
        from src.preprocessing.base import preparar_base
        preparar_base(args.lake, args.ano, args.execution_date)
    elif args.etapa == "eda":
        from src.visualization.exploratoria import executar_eda
        executar_eda()


if __name__ == "__main__":
    main()
