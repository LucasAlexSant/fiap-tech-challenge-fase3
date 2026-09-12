"""Relatório único a partir dos JSON de benchmark, modelo territorial e insights."""
from src import config
from src.artefatos import ler_json


def num(valor, casas=4):
    if valor is None:
        return "—"
    return f"{valor:.{casas}f}".replace(".", ",")


def pct(valor, casas=1):
    return "—" if valor is None else f"{100 * valor:.{casas}f}%".replace(".", ",")


def milhar(valor):
    return f"{int(valor):,}".replace(",", ".")


def tabela(cabecalho, linhas, alinhamento=None):
    alinhamento = alinhamento or ["---"] * len(cabecalho)
    corpo = "\n".join("| " + " | ".join(str(c) for c in linha) + " |" for linha in linhas)
    return f"| {' | '.join(cabecalho)} |\n| {' | '.join(alinhamento)} |\n{corpo}\n"


def secao_benchmark(bench):
    colapso, tetos = bench["colapso"], bench["tetos_de_informacao_no_teste"]
    linhas = []
    for nome, m in sorted(bench["teste"].items(),
                          key=lambda kv: -kv[1]["limiar_congelado"]["acuracia"]):
        v = bench["validacao"][nome]
        linhas.append([f"`{nome}`", num(v["acuracia"]), num(m["limiar_meio"]["acuracia"]),
                       num(m["limiar_congelado"]["acuracia"]), num(m["limiar_meio"]["f1_macro"]),
                       num(m["limiar_meio"]["roc_auc"])])
    aprendidos = {n: m["limiar_congelado"]["acuracia"] for n, m in bench["teste"].items() if n != "baseline_prior"}
    campeao = max(aprendidos, key=aprendidos.get)
    maioria = bench["teste"]["baseline_prior"]["limiar_congelado"]["acuracia"]
    acima = sorted(v for v in aprendidos.values() if v >= maioria - .05)
    tetos_linhas = [
        ["Prever sempre a classe majoritária", "—", num(tetos["baseline_maioria"])],
        ["Saber a taxa exata do **município** no próprio ano",
         milhar(tetos["oraculo_municipio"]["grupos"]), num(tetos["oraculo_municipio"]["acuracia_max"])],
        ["Saber a taxa exata de **município × rede** no próprio ano",
         milhar(tetos["oraculo_municipio_rede"]["grupos"]), num(tetos["oraculo_municipio_rede"]["acuracia_max"])],
        ["Saber a taxa exata da **escola** no próprio ano",
         milhar(tetos["oraculo_escola"]["grupos"]), num(tetos["oraculo_escola"]["acuracia_max"])],
    ]
    return f"""## 1. Por que a acurácia por criança não chega a 70%

O contrato enriquecido tem {len(bench['features'])} preditores. Todos são medidos em
município (ou município × rede): histórico municipal, Bolsa Família, Censo Escolar
agregado, população e IDHM. Nenhum descreve a criança.

A consequência é mensurável. Em {bench['ano_treino']},
**{milhar(colapso['avaliacoes_treino'])} avaliações assumem apenas
{milhar(colapso['unidades_treino'])} valores distintos de X** — uma combinação por
município × rede. Duas crianças da mesma rede recebem obrigatoriamente a mesma
probabilidade. O tamanho amostral efetivo do modelo não é de 1,8 milhão; é de
{milhar(colapso['unidades_treino'])}.

Isso define um teto que nenhum algoritmo ultrapassa. Um oráculo que soubesse a taxa
verdadeira de cada grupo **no próprio ano de teste** chegaria a:

{tabela(["Oráculo (limite superior)", "Grupos", f"Acurácia máxima em {bench['ano_teste']}"], tetos_linhas, ["---", "---:", "---:"])}
O teto municipal é de {num(tetos['oraculo_municipio_rede']['acuracia_max'])} — e só
{num(tetos['oraculo_municipio_rede']['acuracia_max'] - tetos['baseline_maioria'])}
acima de chutar sempre a classe majoritária. Mesmo conhecer a taxa exata de cada
escola daria {num(tetos['oraculo_escola']['acuracia_max'])}. **Pedir 70% de acurácia
por criança é pedir mais informação do que a base contém.**

### Benchmark de {len(bench['teste']) - 1} famílias de algoritmos ({bench['ano_treino']} → {bench['ano_teste']})

Busca de hiperparâmetros em validação cruzada por município; seleção congelada antes
de abrir {bench['ano_teste']}. O limiar de maior acurácia foi escolhido na validação
(`{num(bench['selecao']['limiar_congelado'], 2)}`), nunca no teste.

{tabela(["Modelo", f"Acurácia validação {bench['ano_treino']}", f"Acurácia teste {bench['ano_teste']}",
         "Acurácia teste (limiar congelado)", "F1 macro teste", "ROC AUC teste"], linhas,
        ["---", "---:", "---:", "---:", "---:", "---:"])}
**Nenhum dos {len(aprendidos)} modelos aprendidos supera a classe majoritária em
acurácia.** O melhor deles, `{campeao}`, faz {num(aprendidos[campeao])} contra
{num(maioria)} de não aprender nada — e {len(acima)} candidatos se amontoam entre
{num(min(acima))} e {num(max(acima))}, encostados no teto de informação e não no
algoritmo. (O `naive_bayes` é o único a desabar: sua independência condicional entre
preditores altamente correlacionados não sobrevive à mudança de prevalência entre
edições.) Os modelos ganham do baseline em F1 macro e ROC AUC porque passam a
encontrar não alfabetizados, mas isso custa acurácia.
**Trocar de modelo não resolve; trocar de unidade de análise resolve.**
"""


def secao_municipal(mun):
    completo, estrutural = mun["blocos"]["completo"], mun["blocos"]["estrutural"]
    linhas = []
    for nome, m in sorted(completo["teste"].items(), key=lambda kv: -kv[1]["regra_de_posto"]["acuracia"]):
        v = completo["validacao"][nome]
        e = estrutural["teste"].get(nome, {}).get("regra_de_posto", {})
        linhas.append([f"`{nome}`", num(v["regra_de_posto"]["acuracia"]),
                       num(m["regra_de_posto"]["acuracia"]), num(m["limiar_meio"]["acuracia"]),
                       num(m["limiar_meio"]["roc_auc"]), num(e.get("acuracia"))])
    vencedor = completo["selecao"]["modelo"]
    melhor = completo["teste"][vencedor]["regra_de_posto"]
    rotulos = {"rede": "Rede", "regiao_brasil": "Região", "municipio_novo": "Município inédito em 2024"}
    recortes = [[rotulos[coluna], r["grupo"], milhar(r["n"]), num(r["acuracia"]), num(r["roc_auc"])]
                for coluna in rotulos
                for r in sorted(completo["recortes_teste"][coluna], key=lambda r: -r["n"])]
    return f"""## 2. Mudando a unidade: classificar territórios, não crianças

**Unidade:** município × rede, com {mun['filtro'].replace('avaliacoes', 'avaliações').replace('escolas', 'escolas')}.
**Alvo:** `{mun['alvo']}` = 1 quando a taxa de alfabetização do território fica abaixo
da mediana nacional do ano. Classes equilibradas por construção, então a referência
de acaso é 50% — não 66%, como no nível da criança.

Desenvolvimento: {milhar(mun['desenvolvimento']['unidades_usadas'])} territórios de
{mun['ano_treino']} (corte {num(mun['desenvolvimento']['corte_taxa'])}), cobrindo
{milhar(mun['desenvolvimento']['avaliacoes_cobertas'])} avaliações.
Teste: {milhar(mun['teste']['unidades_usadas'])} territórios de {mun['ano_teste']}
(corte {num(mun['teste']['corte_taxa'])}).

Duas regras de decisão são reportadas. A segunda, **regra de posto**, marca a metade
de menor score: usa só a ordenação prevista e a definição do alvo, e absorve o
deslocamento do nível entre edições — a mediana subiu de
{num(mun['desenvolvimento']['corte_taxa'])} para {num(mun['teste']['corte_taxa'])}
entre {mun['ano_treino']} e {mun['ano_teste']}.

{tabela(["Modelo", f"Validação {mun['ano_treino']} (posto)", f"Teste {mun['ano_teste']} (posto)",
         f"Teste {mun['ano_teste']} (limiar 0,5)", "ROC AUC teste", "Teste sem histórico (posto)"], linhas,
        ["---", "---:", "---:", "---:", "---:", "---:"])}
Sob a regra de posto o F1 macro coincide com a acurácia em todas as linhas
aprendidas: o alvo tem exatamente 50% de positivos e a regra prevê exatamente 50%,
então as duas classes têm os mesmos erros. Por isso a coluna reporta o limiar fixo,
que é onde as duas regras divergem.
O modelo congelado na validação foi **`{vencedor}`**, com acurácia
**{num(melhor['acuracia'])}** e ROC AUC {num(completo['teste'][vencedor]['limiar_meio']['roc_auc'])}
no teste de {mun['ano_teste']} — {num(100 * (melhor['acuracia'] - .5), 1)} pontos
percentuais acima do acaso e
{'acima' if melhor['acuracia'] >= .7 else 'abaixo'} da meta de 70%. A coluna final repete
o experimento sem as três variáveis de histórico de alfabetização: o contexto estrutural
sozinho sustenta a maior parte do desempenho.

### Recortes do teste

{tabela(["Dimensão", "Grupo", "Territórios", "Acurácia", "ROC AUC"], recortes,
        ["---", "---", "---:", "---:", "---:"])}"""


def secao_insights(ins):
    pilares = [[r["pilar"], r["variaveis"], f"{num(r['participacao_pct'], 1)}%"]
               for r in ins["shap_por_pilar"]]
    topo = [[f"`{r['variavel']}`", r["pilar"], num(r["shap_absoluto_medio"], 3)]
            for r in ins["shap"][:15]]
    assoc = [[f"`{r['variavel']}`", r["pilar"], num(r["spearman_bruto"], 3),
              num(r["spearman_parcial"], 3), num(r.get("diferenca_padronizada"), 2)]
             for r in ins["associacoes"][:20]]
    cat = [[r["variavel"], r["grupo"], milhar(r["territorios"]), num(r["taxa_media"]), pct(r["pct_em_risco"])]
           for r in ins["contrastes_categoricos"]]
    perm = [[f"`{r['variavel']}`", r["pilar"], num(r["queda_acuracia"], 4), num(r["desvio"], 4)]
            for r in ins["permutacao"][:12]]
    return f"""## 3. O que de fato separa territórios com alta e baixa alfabetização

Três leituras independentes sobre a mesma unidade, para não depender de um método só:
contribuição SHAP ao modelo (`{ins['modelo_explicado']}`, reconstrução da probabilidade
validada com erro máximo de {ins['erro_maximo_reconstrucao_probabilidade']:.1e}), queda de
acurácia por permutação, e associação direta com a taxa observada.

### 3.1 Peso por pilar temático

{tabela(["Pilar", "Variáveis", "Participação no SHAP total"], pilares, ["---", "---:", "---:"])}
### 3.2 Variáveis individuais mais usadas pelo modelo

{tabela(["Variável", "Pilar", "SHAP absoluto médio (log-odds)"], topo, ["---", "---", "---:"])}
### 3.3 Queda de acurácia por permutação

{tabela(["Variável", "Pilar", "Queda de acurácia", "Desvio"], perm, ["---", "---", "---:", "---:"])}
### 3.4 Associação bruta e parcial com a taxa de alfabetização

A coluna **parcial** remove, por resíduos de posto, o IDHM municipal e a população
antes de medir a associação. É a diferença entre "este território é pobre" e "este
território tem tal recurso na escola". Baseada em
{milhar(ins['n_territorios_associacao'])} territórios das duas edições.

{tabela(["Variável", "Pilar", "Spearman bruto", "Spearman parcial", "Dif. padronizada (quintil alto − baixo)"],
        assoc, ["---", "---", "---:", "---:", "---:"])}
### 3.5 Contrastes categóricos

{tabela(["Dimensão", "Grupo", "Territórios", "Taxa média", "% em risco"], cat, ["---", "---", "---:", "---:", "---:"])}"""


def secao_aplicacao(apl):
    topo = [[f"{r['id_municipio_nome']}/{r['sigla_uf']}", r["rede"], milhar(r["avaliacoes"]),
             milhar(r["escolas"]), num(r["score_risco"], 3), num(r["taxa_alfabetizacao"]),
             num(r.get("meta_municipal")), "sim" if r["risco_territorial"] else "**não**"]
            for r in apl["prioridades"][:15]]
    precisao = [[f"Top {c['k']}", pct(c["precisao"]),
                 f"+{num(100 * c['ganho_sobre_o_acaso'], 1)} pp"]
                for c in apl["precisao_no_topo"]["cortes"]]
    perfis = apl["perfis"]
    grupos = [[f"Perfil {g['perfil']}", milhar(g["territorios"]), num(g["idhm"], 3),
               pct(g["pct_rural"] / 100), pct(g["pct_internet"] / 100),
               pct(g["pct_biblioteca"] / 100), milhar(g["populacao_mediana"]),
               num(g["taxa_media"]), pct(g["pct_em_risco"]),
               f"{g['regiao_predominante']} ({num(g['pct_da_regiao_predominante'], 0)}%)"]
              for g in perfis["grupos"]]
    metas = apl["metas"]
    faixas = [[r["faixa_de_score"], milhar(r["territorios"]), pct(r["pct_abaixo_da_meta"]),
               num(r["distancia_media"]), num(r["taxa_media"])]
              for r in metas["por_faixa_de_score"]]
    razao = (metas["por_faixa_de_score"][-1]["pct_abaixo_da_meta"]
             / max(metas["por_faixa_de_score"][0]["pct_abaixo_da_meta"], 1e-9))
    return f"""## 4. Aplicação: prioridade, perfis e distância da meta

O modelo territorial congelado em `{apl['modelo']['run_id']}` pontua
{milhar(apl['territorios'])} territórios de {apl['ano_teste']}, cobrindo
{milhar(apl['avaliacoes_cobertas'])} avaliações. Nenhuma seleção é refeita aqui.

### 4.1 Qualidade do topo da lista

A acurácia global dilui o erro do topo entre milhares de territórios medianos.
O que importa para quem vai agir é a precisão onde se olha primeiro, contra uma
prevalência de referência de {pct(apl['precisao_no_topo']['prevalencia_de_referencia'])}:

{tabela(["Corte", "Precisão", "Ganho sobre o acaso"], precisao, ["---", "---:", "---:"])}
{tabela(["Território", "Rede", "Avaliações", "Escolas", "Score", "Taxa observada",
         "Meta municipal", "Abaixo da mediana"], topo,
        ["---", "---", "---:", "---:", "---:", "---:", "---:", "---"])}
A taxa observada aparece ao lado do score de propósito. Os erros do topo se
concentram em territórios pequenos, onde a própria taxa observada é instável —
motivo do filtro mínimo de avaliações e escolas.

### 4.2 Perfis de contexto

{perfis['variaveis'].capitalize()}. Critério: {perfis['criterio']}.
Resultado: {perfis['k']} grupos, silhueta {num(perfis['silhueta'], 3)}.

{tabela(["Perfil", "Territórios", "IDHM", "Rural", "Internet", "Biblioteca",
         "População mediana", "Taxa média", "Em risco", "Região predominante"], grupos,
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"])}
### 4.3 Distância até a meta municipal

{milhar(metas['territorios_com_meta'])} territórios da rede Municipal têm meta
publicada; {pct(metas['pct_abaixo_da_meta'])} estão abaixo dela, com distância
mediana de {num(metas['distancia_mediana'])}. O score ordena esse desfecho com
ROC AUC {num(metas['auc_do_score_para_abaixo_da_meta'])}, de forma monótona — o
quintil de maior risco tem {num(razao, 1)} vezes a chance do de menor risco:

{tabela(["Faixa de score", "Territórios", "Abaixo da meta", "Distância média", "Taxa média"],
        faixas, ["---", "---:", "---:", "---:", "---:"])}
{metas['leitura']}"""


def gerar_relatorio_insights():
    bench = ler_json(config.RELATORIOS / "benchmark_modelos.json")
    mun = ler_json(config.RELATORIOS / "modelo_municipal.json")
    ins = ler_json(config.RELATORIOS / "insights_alfabetizacao.json")
    apl = ler_json(config.RELATORIOS / "aplicacao_territorial.json")
    limites = "\n".join(f"- {t}" for t in mun["limites"] + ins["limites"] + apl["limites"])
    texto = f"""# Diagnóstico de modelagem e insights sobre alfabetização

Execução da Gold enriquecida `{bench['execucao_gold']}`; desenvolvimento em
{bench['ano_treino']}, teste em {bench['ano_teste']}. Gerado por
`python main.py benchmark`, `python main.py municipal` e `python main.py insights`.
Nenhuma métrica de teste participou de escolha de modelo, limiar ou variável.

{secao_benchmark(bench)}
![Benchmark](../images/benchmark_acuracia.png)

{secao_municipal(mun)}

{secao_insights(ins)}
![SHAP por pilar](../images/insights_shap_pilares.png)
![SHAP por variável](../images/insights_shap_variaveis.png)
![Associações](../images/insights_associacoes.png)
![Decis](../images/insights_decis.png)

{secao_aplicacao(apl)}
![Prioridades](../images/aplicacao_prioridades.png)

## 5. Limites

{limites}

## 6. Reprodução

```powershell
python main.py benchmark --execution-date {bench['execucao_gold']}
python main.py municipal --execution-date {mun['execucao_gold']}
python main.py insights  --execution-date {ins.get('execucao_gold', bench['execucao_gold'])}
python main.py aplicacao --execution-date {apl['execucao_gold']}
python main.py relatorio-insights
```

Artefatos: `reports/benchmark_modelos.json`, `reports/modelo_municipal.json`,
`reports/insights_alfabetizacao.json` e as cópias por execução em `reports/runs/`.
"""
    caminho = config.RELATORIOS / "diagnostico_e_insights.md"
    caminho.write_text(texto, encoding="utf-8")
    print(f"[RELATORIO] {caminho}")
    return caminho
