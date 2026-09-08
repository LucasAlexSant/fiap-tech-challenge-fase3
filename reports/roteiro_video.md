# Roteiro executivo — até cinco minutos

Versão atualizada para IBGE, Censo Escolar e teste em 2025.
Duração planejada: 4min45s. Público: gestores públicos e lideranças educacionais.
A gravação e o envio ainda precisam ser realizados pelo grupo.

| Tempo | Mensagem sugerida | Apoio visual |
|---|---|---|
| 0:00–0:40 | “Queremos apoiar o diagnóstico de dificuldades de alfabetização. Usamos a Gold da Fase 2, agora com população do IBGE e infraestrutura, matrículas, docentes e turmas do Censo Escolar.” | Notebook 02: contexto e novos atributos |
| 0:40–1:20 | “Desenvolvemos o modelo em 1,85 milhão de avaliações válidas de 2024. Reservamos outra edição, com 1,97 milhão de avaliações de 2025, para testar sua generalização. Roraima passou a estar na base.” | Cobertura e divisão temporal |
| 1:20–2:10 | “O enriquecimento melhorou a validação de 2024, mas esse ganho não se confirmou em 2025: F1 de 0,5833 contra 0,5839 da referência. Esse resultado mostra por que precisamos testar em outro ano, e não confiar apenas no desempenho do desenvolvimento.” | reports/resultados_temporais.md: tabela de modelos |
| 2:10–2:55 | “A calibração melhorou a qualidade das probabilidades, mas reduziu a identificação de casos de risco no limiar de 0,5. Não ajustamos esse limiar usando o teste. Em Roraima, que não estava no treino, a generalização foi insuficiente.” | temporal_calibracao.png; recorte de UF nova |
| 2:55–3:35 | “O histórico municipal segue como a informação mais relevante. Internet aparece entre as contribuições positivas do Censo. São associações do modelo, não evidência de causalidade.” | temporal_permutacao.png |
| 3:35–4:15 | “A aplicação imediata é apoiar investigação e planejamento de apoio, cruzando o diagnóstico com informações das secretarias e escolas. O ranking de 2025 não é uma taxa oficial e ainda não prevê metas futuras.” | Notebook 02: diagnóstico territorial |
| 4:15–4:45 | “Entregamos dados rastreáveis, testes e uma avaliação temporal que expõe os limites da solução. Os próximos passos são verificar disponibilidade histórica das fontes, avaliar agregação municipal e escolher um limiar de atendimento em nova validação.” | README e protocolo_temporal.md |

## Preparação da gravação

- Abrir o notebook 02 já executado e ensaiar para ficar abaixo de cinco minutos.
- Dividir os blocos entre os integrantes.
- Explicar F1 como equilíbrio da classificação das duas classes.
- Diferenciar resultado sem calibração e resultado calibrado: F1 0,5833 e
  0,5666, respectivamente; a escolha da calibração não veio do teste.
- Mostrar somente agregados, sem abrir microdados ou quarentena individual.
- Não afirmar que mais atributos melhoraram a previsão em 2025.
- Explicar que esse teste é retrospectivo e ainda não libera previsão operacional de metas.

## Pendências de entrega

Gravar o vídeo, disponibilizá-lo conforme orientação da turma e realizar o
PR e a revisão colaborativa combinados. Não há link de vídeo ou PR fictício.
