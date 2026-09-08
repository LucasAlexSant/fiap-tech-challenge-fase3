import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.modeling.temporal import verificar_ordem, calibrar_congelado
from src.preprocessing.enriquecida import validar_contextos, NOVAS_NUMERICAS


def contexto():
    return pd.DataFrame({"ano": [2025], "ano_referencia_ibge": [2024], "ano_referencia_censo": [2024],
                         "tem_populacao_ibge": [True], "tem_censo_escolar": [True],
                         **{c: [10.] for c in NOVAS_NUMERICAS}})


def test_teste_precisa_ser_posterior_ao_treino():
    verificar_ordem(2024, 2025)
    with pytest.raises(ValueError, match="posterior"):
        verificar_ordem(2025, 2024)
    with pytest.raises(ValueError, match="posterior"):
        verificar_ordem(2024, 2024)


def test_contexto_rejeita_futuro_medida_sem_ano_e_percentual_invalido():
    validar_contextos(contexto())
    with pytest.raises(ValueError, match="Referência"):
        validar_contextos(contexto().assign(ano_referencia_censo=2025))
    with pytest.raises(ValueError, match="sem referência"):
        validar_contextos(contexto().assign(ano_referencia_ibge=np.nan))
    with pytest.raises(ValueError, match="inválida"):
        validar_contextos(contexto().assign(pct_escolas_internet_censo=101))


def test_calibracao_nao_reajusta_estimador_com_rotulos_reservados():
    X = pd.DataFrame({"x": [-3., -2., -1., 1., 2., 3.]})
    modelo = LogisticRegression().fit(X, [0, 0, 0, 1, 1, 1])
    antes = modelo.coef_.copy()
    calibrado = calibrar_congelado(modelo, X + .3, [0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(modelo.coef_, antes)
    p = calibrado.predict_proba(X)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
