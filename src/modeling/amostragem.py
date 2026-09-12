"""Ajuste com pesos, e réplica controlada para estimadores que não os aceitam."""
import inspect

import numpy as np


def aceita_peso(estimador):
    passo = estimador[-1] if hasattr(estimador, "steps") else estimador
    return "sample_weight" in inspect.signature(passo.fit).parameters


def normalizar(peso):
    """Média 1, preservando exatamente a proporção entre unidades.

    As contagens de avaliações somam milhões. Nessa escala o termo de perda
    domina a penalização (o `saga` da elasticnet não converge nem em 5.000
    iterações) e limites por peso, como `min_child_weight`, deixam de restringir
    qualquer folha. Normalizar mantém o peso relativo de cada unidade — o que
    define o ajuste — e expressa regularização e limites na escala colapsada,
    que é onde as grades de hiperparâmetros os procuram.
    """
    peso = np.asarray(peso, dtype=float)
    if peso.sum() <= 0:
        raise ValueError("Pesos devem somar valor positivo")
    return peso * (len(peso) / peso.sum())


def ajustar(pipeline, X, y, peso, max_replicas=60_000, semente=42):
    """Usa sample_weight quando houver; senão sorteia réplicas proporcionais ao peso.

    No scikit-learn 1.7 quase todo o catálogo aceita peso; sobram o discriminante
    linear e o empilhamento, que caem na réplica. O orçamento padrão de 60.000
    linhas é amplo para 6.536 unidades distintas e mantém o custo previsível.
    """
    if aceita_peso(pipeline):
        chave = "modelo__sample_weight" if hasattr(pipeline, "steps") else "sample_weight"
        return pipeline.fit(X, y, **{chave: normalizar(peso)}), {"ajuste": "sample_weight_normalizado"}
    peso = np.asarray(peso, dtype=float)
    n = int(min(max_replicas, peso.sum()))
    sorteio = np.random.default_rng(semente).choice(len(y), size=n, replace=True, p=peso / peso.sum())
    pipeline.fit(X.iloc[sorteio], np.asarray(y)[sorteio])
    return pipeline, {"ajuste": "replica_proporcional", "linhas_replicadas": n}
