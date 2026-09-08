# Protocolo da evolução com IBGE/Censo

Esta rodada acrescentou 12 atributos e uma edição de teste nova, 2025.
O experimento inicial de nove atributos permanece disponível no notebook 01.
O notebook 02 apresenta a evolução.

## Dados e hipóteses

População ajuda a representar o porte municipal. Infraestrutura, localização
rural, matrículas, docentes e turmas representam recursos e organização das
redes. As hipóteses são de associação, sem atribuir efeito causal.

A Gold enriquecida é uma versão adicional da Fase 2, com 52 colunas físicas
e partições de ano e execução. São usados os nove preditores anteriores e
12 novos. As listas são explícitas; proficiência, metas, identificadores,
marcadores e datas não entram automaticamente no modelo.

O Censo é agregado por município/rede nas escolas ativas com anos iniciais.
Docentes são vínculos contados nas escolas, não pessoas únicas. Percentuais
de infraestrutura usam respostas válidas, com denominadores documentados.
Razões com denominador zero ficam nulas. IBGE usa população de 2022 no
desenvolvimento e estimativa de 2024 no teste, sem calcular crescimento
entre métodos diferentes.

## Separação e escolha

| Papel | Edição | Avaliações elegíveis | Escolas |
|---|---|---:|---:|
| Treino e CV | 2024 | 1.112.203 | 25.396 |
| Escolha do candidato | 2024 | 370.443 | 8.466 |
| Calibração sigmoide | 2024 | 369.206 | 8.466 |
| Teste temporal | 2025 | 1.966.095 | 43.538 |

A divisão de 2024 reaproveita os grupos determinísticos da referência inicial.
O antigo teste de 2024 passa a ser desenvolvimento/calibração nesta rodada;
portanto, não é tratado como teste novo. A avaliação final passa a ser 2025.
IDs de escolas são máscaras; não se tenta estabelecer correspondência
longitudinal nem afirmar que todas as escolas de 2025 são novas.

A busca usa 149.967 avaliações de escolas inteiras do treino e três dobras
StratifiedGroupKFold. Medianas, indicadores de ausência, padronização,
log1p e one-hot são aprendidos dentro das dobras. Taxas/percentuais não
recebem log1p. Quantidades e razões positivas recebem essa transformação.

Candidatos: baseline prior, logística com nove atributos, logística com
21 atributos e boosting com 21 atributos. Logísticas buscam C em 0,1/1 e
pesos None/balanced; boosting busca 15/31 folhas e mínimo de 100/300 registros
por folha, com 150 iterações, L2=10 e sem holdout interno de early stopping.

O maior F1 macro de validação escolhe o candidato; empates exatos seguem a
ordem declarada dos candidatos. Limiar fixo 0,5. Não há troca de candidato
depois de observar o teste.

## Calibração e congelamento

Os candidatos escolhidos na CV são reajustados no treino+validação de 2024.
No vencedor, uma sigmoide é aprendida nos 20% de escolas de 2024 reservados
para calibração. FrozenEstimator impede que essa etapa reajuste o estimador
base. O teste automatizado confere que seus coeficientes não mudam.

Modelo, seleção e hash são salvos antes de carregar a edição de teste.
Esse registro está em [selecao_temporal_antes_teste.json](selecao_temporal_antes_teste.json).
O ajuste da sigmoide é fixado pelo protocolo, sem escolher sua inclusão
pela métrica do teste. Resultados com e sem calibração são apresentados
separadamente porque o mapeamento altera o limiar de classificação efetivo.

Não foi otimizado um limiar usando 2025. Um futuro limiar orientado à
capacidade de atendimento precisa de outra validação de desenvolvimento.

## Resultado e interpretação

A logística enriquecida venceu na validação: F1 0,5994 contra 0,5945 da
referência. No teste temporal, teve F1 0,5833 contra 0,5839 da referência:
o ganho da validação **não se confirmou em 2025**. A diferença é pequena e
não foi testada como efeito estatisticamente significativo.

A calibração reduziu o Brier do vencedor de 0,2340 para 0,2182, mas o F1
no limiar 0,5 caiu para 0,5666 e o recall do risco para 0,3195. A AUC ficou
em 0,6301. A melhoria de calibração não equivale a melhora da detecção.
O IC 95% por bootstrap de escolas do F1 calibrado é [0,5643; 0,5688],
com 300 reamostragens e modelo fixo.

O histórico de alfabetização municipal permanece em primeiro lugar na
permutação. Internet aparece entre as contribuições positivas, mas as
importâncias do conjunto adicional são pequenas ou negativas nesta amostra.
Não se fez nova seleção de atributos com esse resultado.

Roraima, ausente no desenvolvimento, tem 9.310 avaliações elegíveis em 2025.
Nesse recorte, o modelo calibrado teve AUC 0,4995 e F1 0,3324: generalização
insuficiente. A presença de dados no teste não significa que o modelo esteja
adequado para operar em uma UF não vista no treino.

## Alcance e próximas decisões

Foi validada uma edição posterior. Permanecem em aberto a disponibilidade
histórica das versões de todas as fontes e a validade da agregação para
taxas municipais oficiais. O manifesto não declara vintage histórico
comprovado. As métricas não usam ponderação amostral.

Previsões de metas futuras continuam desabilitadas. O diagnóstico de 2025
é agregado por município/rede, com pelo menos 100 avaliações e três escolas,
e serve para investigação. O próximo ciclo deve testar atributos normalizados,
redundância e limiar em desenvolvimento, reservando outra avaliação final.

Referência técnica: [calibração no scikit-learn](https://scikit-learn.org/1.7/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).
Fontes, contratos e reprodução: [enriquecimento da Gold](../../fiap-tech-challenge-fase2/docs/enriquecimento_ibge_censo.md).

Para a entrega acadêmica ainda faltam a gravação do vídeo executivo e o PR
com revisão colaborativa. Downloads, processamento e modelo desta rodada
foram executados localmente.
