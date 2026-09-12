"""Comandos reproduzíveis da Fase 3."""
import argparse
import os
from pathlib import Path

from src import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etapa", choices=["base", "eda", "treinar", "interpretar", "relatorio", "tudo",
                                          "temporal", "benchmark", "municipal", "insights",
                                          "relatorio-insights", "aplicacao", "diagnostico"])
    parser.add_argument("--lake", type=Path, default=config.LAKE)
    parser.add_argument("--ano", type=int, default=config.ANO_PADRAO)
    parser.add_argument("--ano-treino", type=int, default=2024)
    parser.add_argument("--ano-teste", type=int, default=2025)
    parser.add_argument("--execution-date")
    parser.add_argument("--max-busca", type=int, default=150000)
    parser.add_argument("--dobras", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sem-busca", action="store_true")
    parser.add_argument("--amostra-interpretacao", type=int, default=5000)
    parser.add_argument("--sem-empilhamento", action="store_true",
                        help="pula o StackingClassifier no benchmark e no modelo territorial")
    parser.add_argument("--repeticoes-permutacao", type=int, default=10)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads deve ser positivo")
    # Evita a consulta de núcleos físicos via subprocesso no Windows,
    # cuja saída localizada pode não ser UTF-8. Não altera os dados/modelos.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(args.threads))
    if args.etapa == "temporal":
        if not args.execution_date:
            parser.error("temporal exige --execution-date da Gold enriquecida")
        from src.modeling.temporal import executar_temporal
        executar_temporal(args.lake, args.execution_date, args.ano_treino, args.ano_teste,
                         args.max_busca, args.dobras, args.threads)
    # Trilha de diagnóstico: mede o teto de informação da base, muda a unidade
    # de análise para o território e explica o que separa os territórios.
    if args.etapa in ["benchmark", "municipal", "insights", "aplicacao", "diagnostico"] and not args.execution_date:
        parser.error(f"{args.etapa} exige --execution-date da Gold enriquecida")
    if args.etapa in ["benchmark", "diagnostico"]:
        from src.modeling.benchmark import executar_benchmark
        executar_benchmark(args.lake, args.execution_date, args.ano_treino, args.ano_teste,
                           args.threads, not args.sem_empilhamento)
    if args.etapa in ["municipal", "diagnostico"]:
        from src.modeling.municipal import executar_municipal
        executar_municipal(args.lake, args.execution_date, args.ano_treino, args.ano_teste,
                           args.threads, not args.sem_empilhamento)
    if args.etapa in ["insights", "diagnostico"]:
        from src.evaluation.insights import executar_insights
        executar_insights(args.lake, args.execution_date, args.ano_treino, args.ano_teste,
                          args.threads, args.repeticoes_permutacao)
    if args.etapa in ["aplicacao", "diagnostico"]:
        from src.evaluation.aplicacao import executar_aplicacao
        executar_aplicacao(args.lake, args.execution_date, args.ano_teste)
    if args.etapa in ["relatorio-insights", "diagnostico"]:
        from src.evaluation.relatorio_insights import gerar_relatorio_insights
        gerar_relatorio_insights()
    if args.etapa in ["base", "tudo"]:
        from src.preprocessing.base import preparar_base
        preparar_base(args.lake, args.ano, args.execution_date)
    if args.etapa in ["eda", "tudo"]:
        from src.visualization.exploratoria import executar_eda
        executar_eda()
        from src.visualization.exploratoria_enriquecida import executar_eda_enriquecida
        executar_eda_enriquecida(args.lake, args.ano_treino, args.execution_date)
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
