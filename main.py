"""Comandos reproduzíveis da Fase 3."""
import argparse
from pathlib import Path

from src import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etapa", choices=["base", "eda", "treinar"])
    parser.add_argument("--lake", type=Path, default=config.LAKE)
    parser.add_argument("--ano", type=int, default=config.ANO_PADRAO)
    parser.add_argument("--execution-date")
    parser.add_argument("--max-busca", type=int, default=150000)
    parser.add_argument("--dobras", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sem-busca", action="store_true")
    args = parser.parse_args()
    if args.etapa == "base":
        from src.preprocessing.base import preparar_base
        preparar_base(args.lake, args.ano, args.execution_date)
    elif args.etapa == "eda":
        from src.visualization.exploratoria import executar_eda
        executar_eda()
    elif args.etapa == "treinar":
        from src.modeling.treinar import treinar
        treinar(args.max_busca, args.dobras, args.sem_busca, args.threads)


if __name__ == "__main__":
    main()
