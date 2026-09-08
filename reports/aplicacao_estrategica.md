# Aplicação estratégica

Resultados da execução `20260908T003806585513Z`, sobre avaliações elegíveis
de 2024. Este documento responde às perguntas do enunciado com o alcance
permitido pelos dados disponíveis.

## Quais fatores mais impactam a alfabetização?

O experimento identifica associações preditivas. O histórico municipal
anterior foi o atributo mais relevante: queda média de F1 de 0,0374 na
permutação; rede (0,0097) e região (0,0075) vieram depois.
Isso justifica investigar persistência territorial e diferenças de rede.

Não se pode concluir que mudar um atributo causará melhora. Pagamentos
Bolsa Família descrevem contexto municipal, sem ligação demonstrada com
a família do aluno. Valores totais também refletem porte populacional,
que ainda não foi incorporado como denominador.

## Quais municípios apresentam maior risco educacional?

O ranking é um diagnóstico da amostra de escolas do teste, por município
e rede. Exige pelo menos 100 avaliações e três escolas; 642 grupos atendem
ao filtro. Os primeiros grupos municipais são:

| Município/rede Municipal | Avaliações de teste | Escolas | Score médio |
|---|---:|---:|---:|
| Araci/BA | 112 | 3 | 0,8161 |
| Paulo Afonso/BA | 343 | 7 | 0,8115 |
| Porto Alegre/RS | 399 | 7 | 0,8101 |
| Feira de Santana/BA | 711 | 19 | 0,7959 |

Fonte: [ranking agregado](priorizacao_amostra_teste.json).
Scores são de 0 a 1, sem recalibração; não são percentuais oficiais nem
estimativas de toda a população. A diferença entre scores próximos não
foi submetida a teste de significância.

A utilidade imediata é selecionar territórios para diagnóstico local,
conferindo cobertura, resultados observados e informações da secretaria
antes de definir qualquer apoio. A posição não deve determinar corte de
recursos ou classificação individual de crianças.

## Quais regiões possuem padrões semelhantes?

Na amostra de teste, Centro-Oeste e Sudeste têm prevalências observadas
próximas (36,17% e 37,04%). Sul tem 39,19%, Nordeste 42,90% e Norte 49,43%.
É uma comparação descritiva; não foi ajustado um modelo de agrupamento.

| Região | Avaliações de teste | F1 macro | Recall do risco |
|---|---:|---:|---:|
| Centro-Oeste | 35.594 | 0,5618 | 0,3555 |
| Nordeste | 94.115 | 0,6321 | 0,7450 |
| Norte | 39.551 | 0,4664 | 0,9008 |
| Sudeste | 145.705 | 0,5694 | 0,5635 |
| Sul | 54.241 | 0,6225 | 0,5370 |

Fonte: `recortes_teste.regiao_brasil` em [resultados.json](resultados.json).
No Norte, o alto recall do risco acompanha recall de apenas 17,57% dos
alfabetizados: o modelo marca risco com frequência excessiva nesse recorte.
Roraima não está na base. Os números não são estimativas oficiais regionais.

Similaridade de prevalência não implica igualdade de condições. Comparar
infraestrutura, urbanização, porte e rede permitiria testar perfis
territoriais em outra rodada.

## Como prever municípios que podem não atingir metas futuras?

Esta versão ainda não responde com uma previsão validada. Foi avaliada
uma edição, e os scores não estão calibrados. Subtrair o score médio de
uma meta produziria uma comparação sem sustentação.

Para tornar essa análise possível:

1. Registrar datas de publicação de atributos e versões das metas.
2. Incorporar outras edições, treinando no passado e reservando uma edição
   futura para teste temporal.
3. Ajustar calibração e limiar apenas nos conjuntos de desenvolvimento.
4. Definir população e rede de referência e avaliar a agregação municipal.
5. Estimar incerteza e comparar alfabetização prevista e meta da mesma
   rede/ano, antes de afirmar risco de descumprimento.

Metas municipais atuais aparecem somente como contexto na rede Municipal.
Valores ausentes permanecem ausentes.

## Quais variáveis possuem maior influência nos modelos?

A permutação mede sensibilidade do F1; SHAP mede contribuição às saídas.
Ambas colocaram o histórico municipal em primeiro lugar. UF aparece em
segundo no SHAP, apesar de permutação próxima de zero, o que é compatível
com redundância entre UF, região e contexto municipal.

Não se deve tratar rankings diferentes como contradição automática.
Os métodos respondem perguntas diferentes, e variáveis correlacionadas
compartilham informação. Remover atributos com importância negativa será
hipótese de ablação em outra rodada, sem selecionar pelo teste atual.

## Proposta de uso na gestão

A equipe técnica apresentaria um painel agregado com cobertura, resultado
observado, score, limitações e histórico. A secretaria verificaria a
situação com as escolas e planejaria apoio pedagógico conforme diagnóstico,
não apenas conforme score.

Um piloto acompanhado poderia medir cobertura do atendimento, capacidade
de identificar necessidade e evolução educacional. Melhorar F1 não prova
que uma intervenção melhora a alfabetização: esse efeito exige avaliação
própria.

## Enriquecimentos prioritários da Gold

| Prioridade | Fonte e atributos propostos | Integração e verificação |
|---|---|---|
| 1 | IBGE: população, população em idade escolar, urbanização quando disponível | Município/ano; publicação anterior à decisão; denominadores compatíveis |
| 2 | Censo Escolar: matrículas, docentes, infraestrutura, rural/urbano | Agregar por município/rede/ano; não ligar escolas pelo código instável |
| 3 | Outra edição de microdados e cobertura faltante | Mesma definição do alvo; auditar RR e mudanças de cobertura |
| 4 | FUNDEB ou outros indicadores socioeconômicos | Unidade monetária, correção temporal, cobertura e disponibilidade |

Toda nova fonte deve passar pela engenharia da Fase 2 e integrar uma nova
versão Gold antes de entrar na Fase 3. Este ciclo ainda não baixou nem
incorporou esses enriquecimentos.
