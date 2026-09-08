"""Resumo em Markdown sempre derivado das métricas materializadas."""
from src import config
from src.artefatos import ler_json


def gerar_relatorio():
    r = ler_json(config.RELATORIOS / "resultados.json")
    i = ler_json(config.RELATORIOS / "interpretacao.json")
    p = ler_json(config.RELATORIOS / "proveniencia.json")
    if r["run_id"] != i["run_id"] or r["base_sha256"] != p["base_sha256"]:
        raise ValueError("Relatórios de execuções diferentes; execute a pipeline completa")
    escolhido = r["selecao"]["modelo"]
    m = r["teste"][escolhido]
    linhas = ["# Resultados da execução", "", f"Execução: `{r['run_id']}`. Gold: `{r['execution_date_gold']}`, ano {r['ano']}.", "",
        f"Modelo escolhido **antes do teste**: **{escolhido}**. {r['selecao']['criterio']}. Limiar fixo 0,5.", "",
        "| Modelo | F1 validação | F1 teste | Recall risco | Precisão risco | ROC AUC | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for nome, resultado in r["teste"].items():
        linhas.append(f"| {nome} | {r['validacao'][nome]['f1_macro']:.4f} | {resultado['f1_macro']:.4f} | {resultado['recall_risco']:.4f} | {resultado['precisao_risco']:.4f} | {resultado['roc_auc']:.4f} | {resultado['brier']:.4f} |")
    ic = r["incerteza_escolas"]
    linhas += ["", f"Teste: **{m['n']:,} avaliações**. F1 macro do escolhido: IC 95% por bootstrap de escolas "
               f"**[{ic['f1_macro_ic95'][0]:.4f}; {ic['f1_macro_ic95'][1]:.4f}]** ({ic['repeticoes']} repetições).",
               "", "## Interpretação", "",
               f"No limiar escolhido, identifica {m['recall_risco']:.1%} dos não alfabetizados; "
               f"{m['precisao_risco']:.1%} dos classificados em risco pertencem a essa classe.", "",
               "O critério de escolha foi F1 macro. AUC, Brier e recall podem favorecer candidatos diferentes; "
               "não se troca o vencedor após observar o teste. Class weighting pode deslocar as probabilidades: "
               "a curva de calibração deve acompanhar qualquer uso dos scores.", "",
               "![Calibração](../images/teste_calibracao.png)", "",
               "## Variáveis e explicações", "",
               "| Variável | Queda de F1 na permutação | Desvio das permutações |", "|---|---:|---:|"]
    for v in i["permutacao"]:
        linhas.append(f"| `{v['variavel']}` | {v['queda_f1_macro']:.5f} | {v['desvio_permutacoes']:.5f} |")
    linhas += ["", f"Permutação: {i['amostra_permutacao']} avaliações, {i['repeticoes']} repetições. "
               f"SHAP: {i['amostra_shap']} avaliações; erro máximo ao reconstruir a probabilidade "
               f"{i['erro_maximo_reconstrucao_probabilidade']:.3g}.", "",
               "![SHAP](../images/interpretacao_shap.png)", "",
               "A importância descreve o modelo, não causas. Correlações entre pagamentos, valores "
               "financeiros e porte territorial compartilham sinal e limitam atribuições individuais.", "",
               "## Uso e limitações", "",
               "O ranking territorial usa somente escolas reservadas para teste, com pelo menos 100 "
               "avaliações e 3 escolas por município/rede. É diagnóstico da amostra; não representa "
               "todos os estudantes, não estima a taxa oficial e não prevê o ano seguinte.", "",
               "A divisão mede generalização para escolas não vistas dentro da edição, em territórios "
               "que podem aparecer no treino. Não é validação temporal nem teste de municípios inteiramente novos.", "",
               "As fontes não comprovam disponibilidade numa data de decisão passada. Ausentes não fazem "
               "parte do treino e os resultados não se estendem automaticamente a eles. Não há atributos "
               "individuais familiares, população ou infraestrutura escolar nesta versão.", "",
               "Próxima comparação: enriquecer Censo Escolar/IBGE na Gold e avaliar sob protocolo "
               "pré-definido com nova validação; calibrar e escolher limiar fora do teste. Não reutilizar "
               "o mesmo teste repetidamente para escolher melhorias.", ""]
    destino = config.RELATORIOS / "resultados.md"
    destino.write_text("\n".join(linhas), encoding="utf-8")
    print(f"[RELATORIO] {destino}")
