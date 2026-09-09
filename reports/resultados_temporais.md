# Gold enriquecida e validação em outra edição

Execução `20260909T004124029778Z_temporal`. Desenvolvimento: **2024**. Teste: **2025**.

**12 novos atributos**, totalizando 75 preditores. Modelo escolhido na validação: **enriquecida_logistica**.

Ajuste e escolha usam somente 2024. A sigmoide é aprendida em escolas de 2024 reservadas para calibração. O modelo é salvo e seu hash registrado antes de carregar o teste de 2025.

## Cobertura da Gold

| Ano | Avaliações | Elegíveis | UFs | IBGE (%) | Censo (%) | Histórico (%) | Bolsa (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 1747439 | 1502809 | 23 | 100.00 | 100.00 | 0.00 | 0.00 |
| 2024 | 2120560 | 1851852 | 26 | 100.00 | 100.00 | 76.88 | 100.00 |
| 2025 | 2222164 | 1966095 | 27 | 99.99 | 99.99 | 99.26 | 99.99 |

IBGE é integrado por município; Censo por município e rede. Escolas ativas com matrículas nos anos iniciais compõem o contexto. Docentes são vínculos por escola, não pessoas únicas. Percentuais usam respostas válidas; ausência não vira zero.

## Comparação

| Modelo | F1 validação 2024 | F1 teste 2025 | Recall risco | Precisão risco | AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.3733 | 0.3985 | 0.0000 | 0.0000 | 0.5000 | 0.2278 |
| referencia_logistica | 0.5945 | 0.5839 | 0.4950 | 0.4402 | 0.6311 | 0.2333 |
| enriquecida_logistica | 0.6011 | 0.5825 | 0.4918 | 0.4388 | 0.6282 | 0.2331 |
| enriquecida_boosting | 0.5870 | 0.5667 | 0.3075 | 0.4768 | 0.6375 | 0.2154 |
| selecionado_calibrado | — | 0.5679 | 0.3214 | 0.4691 | 0.6282 | 0.2182 |

Comparando logísticas sob o mesmo protocolo, a diferença de F1 no teste com enriquecimento foi **-0.0013**. O sinal pode favorecer ou desfavorecer a inclusão; o teste não é usado para nova escolha.

A calibração é uma transformação predefinida, não um novo candidato escolhido pelo teste. Pode melhorar Brier e reduzir recall/F1 no limiar 0,5; ambos os resultados são apresentados.

![Calibração temporal](../images/temporal_calibracao.png)

## Influência dos atributos

Permutação de 5.000 avaliações do teste, cinco repetições, F1 macro. Diagnóstico final, sem seleção posterior de atributos.

![Permutação](../images/temporal_permutacao.png)

## Limites e metas futuras

- Teste de outra edição; identidades escolares não são longitudinais.
- Datas e versões históricas de publicação não estão integralmente comprovadas.
- Calibração aprendida em 2024 pode se degradar com mudanças em 2025.
- População de 2022 é censo; 2024 é estimativa; não inferir crescimento diretamente.
- Resultados não ponderados por desenho amostral; não estimam indicadores oficiais.
- Não prever metas futuras até validar disponibilidade temporal, agregação e incerteza.

O ranking agregado é diagnóstico das avaliações elegíveis de 2025, sem converter scores em taxas oficiais ou em distância prevista de metas futuras. As métricas por UF, região, rede e municípios novos constam de [resultados_temporais.json](resultados_temporais.json).
