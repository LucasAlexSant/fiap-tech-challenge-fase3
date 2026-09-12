# Resultados da execução

Execução: `20260910T030117850091Z`. Gold: `2026-09-07`, ano 2024.

Modelo escolhido **antes do teste**: **logistica**. F1 macro de validação; tolerância 0.005 favorece logística. Limiar fixo 0,5.

| Modelo | F1 validação | F1 teste | Recall risco | Precisão risco | ROC AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.3733 | 0.3746 | 0.0000 | 0.0000 | 0.5000 | 0.2402 |
| logistica | 0.5945 | 0.5980 | 0.6356 | 0.5023 | 0.6567 | 0.2304 |
| gradient_boosting | 0.5880 | 0.5884 | 0.3549 | 0.5853 | 0.6661 | 0.2204 |

Teste: **369,206 avaliações**. F1 macro do escolhido: IC 95% por bootstrap de escolas **[0.5934; 0.6023]** (300 repetições).

## Interpretação

No limiar escolhido, identifica 63.6% dos não alfabetizados; 50.2% dos classificados em risco pertencem a essa classe.

O critério de escolha foi F1 macro. AUC, Brier e recall podem favorecer candidatos diferentes; não se troca o vencedor após observar o teste. Class weighting pode deslocar as probabilidades: a curva de calibração deve acompanhar qualquer uso dos scores.

![Calibração](../images/teste_calibracao.png)

## Variáveis e explicações

| Variável | Queda de F1 na permutação | Desvio das permutações |
|---|---:|---:|
| `taxa_alfabetizacao_municipio_anterior` | 0.03743 | 0.00475 |
| `rede` | 0.00967 | 0.00191 |
| `regiao_brasil` | 0.00750 | 0.00351 |
| `sigla_uf` | -0.00005 | 0.00176 |
| `valor_medio_pagamento_bolsa_familia_anterior` | -0.00226 | 0.00041 |
| `taxa_presenca_municipio_anterior` | -0.00235 | 0.00235 |
| `valor_total_bolsa_familia_anterior` | -0.00433 | 0.00157 |
| `total_pagamentos_bolsa_familia_anterior` | -0.00436 | 0.00157 |
| `alunos_avaliados_municipio_anterior` | -0.00651 | 0.00288 |

Permutação: 5000 avaliações, 5 repetições. SHAP: 2000 avaliações; erro máximo ao reconstruir a probabilidade 2.22e-16.

![SHAP](../images/interpretacao_shap.png)

A importância descreve o modelo, não causas. Correlações entre pagamentos, valores financeiros e porte territorial compartilham sinal e limitam atribuições individuais.

## Uso e limitações

O ranking territorial usa somente escolas reservadas para teste, com pelo menos 100 avaliações e 3 escolas por município/rede. É diagnóstico da amostra; não representa todos os estudantes, não estima a taxa oficial e não prevê o ano seguinte.

A divisão mede generalização para escolas não vistas dentro da edição, em territórios que podem aparecer no treino. Não é validação temporal nem teste de municípios inteiramente novos.

As fontes não comprovam disponibilidade numa data de decisão passada. Ausentes não fazem parte do treino e os resultados não se estendem automaticamente a eles. Não há atributos individuais familiares, população ou infraestrutura escolar nesta versão.

Próxima comparação: enriquecer Censo Escolar/IBGE na Gold e avaliar sob protocolo pré-definido com nova validação; calibrar e escolher limiar fora do teste. Não reutilizar o mesmo teste repetidamente para escolher melhorias.
