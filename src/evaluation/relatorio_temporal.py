"""Síntese gerada a partir da avaliação temporal e de sua proveniência."""
from src import config


def gerar_relatorio_temporal(r):
    s = r["selecao"]
    linhas = ["# Gold enriquecida e validação em outra edição", "",
              f"Execução `{r['run_id']}`. Desenvolvimento: **{s['ano_treino']}**. Teste: **{s['ano_teste']}**.", "",
              f"**12 novos atributos**, totalizando {len(r['features'])} preditores. Modelo escolhido na validação: **{s['modelo']}**.", "",
              "Ajuste e escolha usam somente 2024. A sigmoide é aprendida em escolas de 2024 reservadas para calibração. "
              "O modelo é salvo e seu hash registrado antes de carregar o teste de 2025.", "",
              "## Cobertura da Gold", "",
              "| Ano | Avaliações | Elegíveis | UFs | IBGE (%) | Censo (%) | Histórico (%) | Bolsa (%) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for ano, c in r["cobertura_gold"].items():
        linhas.append(f"| {ano} | {c['linhas_gold']} | {c['elegiveis']} | {len(c['ufs'])} | {c['cobertura_populacao_pct']:.2f} | {c['cobertura_censo_pct']:.2f} | {c['cobertura_historico_pct']:.2f} | {c['cobertura_bolsa_pct']:.2f} |")
    linhas += ["", "IBGE é integrado por município; Censo por município e rede. Escolas ativas com matrículas "
               "nos anos iniciais compõem o contexto. Docentes são vínculos por escola, não pessoas únicas. "
               "Percentuais usam respostas válidas; ausência não vira zero.", "", "## Comparação", "",
               "| Modelo | F1 validação 2024 | F1 teste 2025 | Recall risco | Precisão risco | AUC | Brier |", "|---|---:|---:|---:|---:|---:|---:|"]
    for nome, m in r["teste"].items():
        f1v = f"{r['validacao'][nome]['f1_macro']:.4f}" if nome in r["validacao"] else "—"
        linhas.append(f"| {nome} | {f1v} | {m['f1_macro']:.4f} | {m['recall_risco']:.4f} | {m['precisao_risco']:.4f} | {m['roc_auc']:.4f} | {m['brier']:.4f} |")
    delta = r["teste"]["enriquecida_logistica"]["f1_macro"] - r["teste"]["referencia_logistica"]["f1_macro"]
    linhas += ["", f"Comparando logísticas sob o mesmo protocolo, a diferença de F1 no teste com enriquecimento foi **{delta:+.4f}**. "
               "O sinal pode favorecer ou desfavorecer a inclusão; o teste não é usado para nova escolha.", "",
               "A calibração é uma transformação predefinida, não um novo candidato escolhido pelo teste. "
               "Pode melhorar Brier e reduzir recall/F1 no limiar 0,5; ambos os resultados são apresentados.", "",
               "![Calibração temporal](../images/temporal_calibracao.png)", "",
               "## Influência dos atributos", "",
               "Permutação de 5.000 avaliações do teste, cinco repetições, F1 macro. Diagnóstico final, sem seleção posterior de atributos.", "",
               "![Permutação](../images/temporal_permutacao.png)", "",
               "## Limites e metas futuras", ""]
    linhas.extend(f"- {limite}" for limite in r["limites"])
    linhas += ["", "O ranking agregado é diagnóstico das avaliações elegíveis de 2025, sem converter scores "
               "em taxas oficiais ou em distância prevista de metas futuras. "
               "As métricas por UF, região, rede e municípios novos constam de [resultados_temporais.json](resultados_temporais.json).", ""]
    destino = config.RELATORIOS / "resultados_temporais.md"
    destino.write_text("\n".join(linhas), encoding="utf-8")
