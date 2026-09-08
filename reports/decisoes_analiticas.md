# Decisões analíticas e cobertura do enunciado

Referência: `[IAST] - Tech Challenge - Fase 3.pdf`, páginas 2–8.
Esta versão estabelece a referência reproduzível com a Gold disponível.
A execução registrada é `20260908T003806585513Z`.

## Unidade, alvo e tempo

A unidade é a avaliação de aluno, identificada por ano, município, escola,
aluno, série e rede. O alvo vem da Gold: 1 alfabetizado, 0 não alfabetizado.
São usados somente os 1.851.852 registros elegíveis de 2024. Ausência na
avaliação não é rótulo de não alfabetização.

Atributos municipais e pagamentos são do ano anterior. Isso reduz vazamento
do resultado contemporâneo, mas não comprova quando os dados foram
publicados. O estudo é retrospectivo. A previsão antes de uma avaliação
futura exige catálogo de disponibilidade e teste temporal.

## Hipóteses e decisões apoiadas na EDA do treino

| Evidência/hipótese | Decisão | Limite |
|---|---|---|
| Alfabetização municipal anterior pode carregar persistência territorial | Incluir taxa anterior, sem proficiência atual | Associação contextual, não trajetória individual |
| Cerca de 23,17% das avaliações de treino não têm histórico de alfabetização municipal | Mediana e indicador de ausência aprendidos dentro das dobras | Imputação não recupera o contexto ausente |
| Valores financeiros e quantidades têm caudas longas | log1p, seguido de padronização | P99 limita apenas a figura, não remove observações |
| Volume e valor de pagamentos são correlacionados | Preservar referência inicial; reportar correlação e atribuições | Importância compartilhada; ablação fica para outra rodada |
| Rede, UF e região podem separar contextos | One-hot com categorias desconhecidas ignoradas | Há redundância geográfica; não interpretar coeficientes como causas |
| Aproximadamente 40,19% do treino pertence à classe de risco | F1 macro, recall e precisão da classe 0; comparar pesos de classe na CV | Pesos podem prejudicar calibração |
| Alunos de uma escola compartilham contexto | Dividir e validar por escolas inteiras | Municípios podem estar nos dois conjuntos |

As estatísticas descritivas por avaliação repetem o contexto dos municípios
com mais alunos. A matriz de Spearman agrega municípios no treino para
reduzir essa repetição. Presença refere-se à prova, não à frequência escolar.
Não foram removidos extremos válidos apenas para aumentar métricas.

## Proteções contra vazamento

- A leitura aceita somente uma partição Gold de execução/ano; valida chaves,
  elegibilidade, rótulos, faixas, valores finitos e referências temporais.
- Os nove atributos formam uma lista explícita. Identificadores servem a
  chaves/divisão, sem entrar no modelo.
- Proficiência, presença atual, elegibilidade, status/distância da meta
  e pesos não validados são excluídos.
- Metas ficam fora dos preditores por falta de histórico de publicação.
- Histórico por escola foi excluído: entre as edições, 35.597 dos 36.462
  códigos compartilhados aparecem em municípios diferentes.
- EDA usa treino. Imputação, transformação e encoding são ajustados nas
  dobras, junto ao estimador, por Pipeline/ColumnTransformer.
- Escolha e limiar são congelados antes do teste. SHAP e permutação no teste
  são diagnósticos finais, sem orientar nova seleção nesta execução.

## Validação, busca e overfitting

GroupShuffleSplit, semente 42, reserva aproximadamente 60%/20%/20% das
escolas para treino, validação e teste. Zero escolas compartilhadas.
O teste contém 369.206 avaliações de 8.466 escolas. O protocolo mede escolas
novas na mesma edição, com possível contexto municipal já visto.

Para controlar custo local, a busca usa 149.967 avaliações de escolas
inteiras do treino, sem seleção pelo alvo. StratifiedGroupKFold tem três
dobras e semente fixa. A amostra limita a busca; os candidatos escolhidos
são ajustados em todo o treino.

| Modelo | Grade |
|---|---|
| Baseline prior | Sem busca |
| Logística | C = 0,1 ou 1; class_weight = None ou balanced |
| HistGradientBoosting | max_leaf_nodes = 15 ou 31; min_samples_leaf = 100 ou 300 |

O boosting usa 150 iterações, regularização L2 de 10 e early stopping
desativado para evitar um holdout interno aleatório por aluno.
Regularização logística e tamanho mínimo das folhas controlam complexidade.

A melhor logística teve F1 de treino CV 0,6009 e média de validação CV
0,6012 (desvio 0,0008). O melhor boosting teve 0,6155 no treino e 0,5785 na CV
(desvio 0,0090): diferença compatível com maior ajuste ao treino.
Depois do ajuste no treino completo, a validação independente favoreceu
a logística: 0,5945 contra 0,5880.

A regra é maior F1 macro na validação, favorecendo logística quando a
vantagem do boosting não supera 0,005. Limiar 0,5, sem otimização posterior.
O vencedor foi congelado e os candidatos reajustados em treino+validação.
As métricas dos três modelos no teste são comparação descritiva.

## Incerteza, explicação e calibração

Bootstrap de 300 reamostragens de escolas, com modelo fixo: IC 95% do F1
macro de 0,5934 a 0,6023. O intervalo não incorpora incerteza de todas as
fontes nem da escolha do modelo.

Permutação em 5.000 avaliações e cinco repetições mede queda de F1 ao
embaralhar cada atributo. Desvio entre repetições não é IC populacional.
Importâncias negativas são mantidas.

SHAP explica 2.000 avaliações em log-odds da classe 0. Categorias e
indicadores de ausência são somados à variável original. A transformação
inversa reconstruiu a probabilidade com erro máximo de 2,22e-16.

A logística balanceada teve Brier 0,2304, enquanto o boosting teve 0,2204.
O score da logística não deve ser convertido diretamente em taxa municipal.
Calibração e limiar orientado a capacidade de atendimento ficam para
validação futura, sem ajuste sobre este teste.

## Evidências por requisito

| Requisito do PDF | Evidência ou pendência |
|---|---|
| Base proveniente da Gold | src/preprocessing/base.py; reports/proveniencia.json |
| EDA, padrões, correlações e hipóteses | src/visualization; images/eda_*.png; este documento |
| Imputação, transformação, encoding e integração | src/modeling/pipelines.py |
| Separação, otimização, replicabilidade e generalização | src/preprocessing/divisao.py; src/modeling/treinar.py |
| Métricas e interpretabilidade | reports/resultados.md; reports/interpretacao.json |
| Perguntas de negócio | reports/aplicacao_estrategica.md |
| Estrutura, scripts e notebook | README; main.py; notebooks/01_analise_alfabetizacao.ipynb |
| Git, commits e branches | Histórico local e branch feat/reconstrucao-fase3-gold |
| Pull request e revisão colaborativa | Pendente, conforme combinado com o usuário |
| Vídeo executivo de até cinco minutos | Roteiro pronto; gravação pendente |
| População e educação complementar citadas no PDF | Ainda ausentes; enriquecer a Gold com IBGE/Censo Escolar |
| Antecipar metas futuras | Ainda não validado; exige outra safra e disponibilidade temporal |

## Reprodução verificada

`python main.py tudo` terminou com código 0 e gerou base, EDA, busca,
modelo, métricas, SHAP, ranking e relatório. Os 14 testes automatizados
passaram. Hashes da Gold, base, modelo e código, versões e semente constam
dos JSON. Relatórios agregados e figuras são versionados; dados individuais,
predições e modelo serializado permanecem locais e são reconstruíveis.

Uma versão nova de dados ou de parâmetros constitui outro experimento.
As análises deste documento descrevem a execução registrada acima.
