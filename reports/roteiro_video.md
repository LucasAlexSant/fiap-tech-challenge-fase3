# Roteiro executivo — até cinco minutos

Duração planejada: 4min45s. Público: gestores públicos e lideranças
educacionais. Usar os gráficos exportados ou o notebook para apoiar a fala.
A gravação e o envio ainda precisam ser realizados pelo grupo.

| Tempo | Mensagem sugerida | Apoio visual |
|---|---|---|
| 0:00–0:40 | “Queremos apoiar o diagnóstico de dificuldades de alfabetização. Integramos a Gold da Fase 2 e estudamos 1,85 milhão de avaliações válidas de 2024. O modelo usa contexto territorial, rede e dados do ano anterior.” | README: problema e cobertura |
| 0:40–1:15 | “Parte dos municípios não tem histórico anterior disponível, e Roraima está ausente. Tratamos faltantes dentro do modelo. Separamos escolas inteiras para que a avaliação não misture alunos da mesma escola entre treino e teste.” | eda_ausencias.png; tabela da divisão |
| 1:15–2:05 | “Comparamos uma referência simples, regressão logística e boosting. Escolhemos a logística na validação, antes de abrir o teste. No teste, o F1 macro foi 0,598 e identificamos 63,6% dos casos de não alfabetização. Cerca de metade dos alertas correspondeu à classe de risco.” | teste_confusao.png; tabela de resultados |
| 2:05–2:50 | “O histórico municipal foi a informação com maior influência. Rede e região também contribuem. São associações, não causas. Há diferenças regionais relevantes: no Norte, o modelo produz alertas em excesso. O score ainda precisa de calibração para ser usado como probabilidade.” | interpretacao_permutacao.png; teste_calibracao.png |
| 2:50–3:45 | “O ranking ajuda a escolher onde aprofundar o diagnóstico. Ele considera apenas as escolas do teste e exige uma amostra mínima. Araci, Paulo Afonso e Porto Alegre aparecem entre os primeiros grupos municipais; isso não é um ranking oficial nem previsão do próximo ano.” | prioridades_teste.png |
| 3:45–4:25 | “Para apoiar políticas públicas, propomos cruzar o diagnóstico com informações locais e planejar apoio pedagógico. A próxima etapa incorpora população e infraestrutura escolar, calibra o modelo e testa outra edição. Só então poderemos avaliar previsões de metas futuras.” | aplicacao_estrategica.md: próximos passos |
| 4:25–4:45 | “Entregamos uma análise reproduzível e rastreável, com código, testes, explicações e limites claros. O valor é orientar investigação e planejamento de apoio, com revisão dos gestores e avaliação dos resultados das intervenções.” | README e comando python main.py tudo |

## Preparação da gravação

- Abrir o notebook já executado, com resultados e figuras prontos.
- Dividir os blocos entre integrantes e ensaiar para ficar abaixo de cinco minutos.
- Explicar “F1” como equilíbrio da classificação entre as duas classes,
  sem apresentar acurácia como evidência suficiente.
- Manter visível que o ranking usa a amostra do teste e scores sem calibração.
- Mostrar resultados agregados, sem abrir microdados ou predições individuais.
- Confirmar que as métricas apresentadas pertencem à mesma execução.

## Antes da entrega acadêmica

O repositório local e o roteiro estão preparados. Ainda é necessário gravar
o vídeo, disponibilizá-lo conforme orientação da turma e realizar o PR e a
revisão colaborativa combinados. Não há link de vídeo ou PR fictício.
