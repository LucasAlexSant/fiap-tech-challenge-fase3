"""Comandos reproduzíveis da Fase 3."""
import argparse
from pathlib import Path

from src import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etapa", choices=["base"])
    parser.add_argument("--lake", type=Path, default=config.LAKE)
    parser.add_argument("--ano", type=int, default=config.ANO_PADRAO)
    parser.add_argument("--execution-date")
    args = parser.parse_args()
    from src.preprocessing.base import preparar_base
    preparar_base(args.lake, args.ano, args.execution_date)


if __name__ == "__main__":
    main()
