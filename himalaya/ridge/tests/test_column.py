import pytest
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from himalaya.backend import set_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.utils import assert_array_almost_equal

from himalaya.ridge import ColumnTransformerNoStack
from himalaya.ridge import make_column_transformer_no_stack
from himalaya.ridge import GroupRidgeCV


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_column_transformer_all_columns(backend):
    backend = set_backend(backend)
    X = np.random.randn(10, 5)

    ct = ColumnTransformerNoStack([("name", StandardScaler(), slice(0, 5))])
    Xt = ct.fit_transform(X)
    assert len(Xt) == 1
    assert Xt[0].shape == (10, 5)


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_column_transformer_passthrough(backend):
    backend = set_backend(backend)
    X = np.random.randn(10, 5)

    ct = ColumnTransformerNoStack([("name", "passthrough", slice(0, 5))])
    Xt = ct.fit_transform(X)
    assert len(Xt) == 1
    assert Xt[0].shape == (10, 5)
    assert_array_almost_equal(X, Xt[0])


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_column_transformer_remainder(backend):
    backend = set_backend(backend)
    X = np.random.randn(10, 5)

    ct = ColumnTransformerNoStack([("name", "passthrough", slice(0, 0))],
                                  remainder="passthrough")
    Xt = ct.fit_transform(X)
    assert len(Xt) == 2
    assert Xt[0].shape == (10, 0)
    assert Xt[1].shape == (10, 5)
    assert_array_almost_equal(X, Xt[1])


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_column_transformer_multiple(backend):
    backend = set_backend(backend)
    X = np.random.randn(10, 5)

    ct = ColumnTransformerNoStack([
        ("name0", StandardScaler(), [0, 1]),
        ("name1", StandardScaler(with_mean=False), [2, 3]),
    ], remainder="passthrough")
    Xt = ct.fit_transform(X)
    assert len(Xt) == 3
    assert Xt[0].shape == (10, 2)
    assert Xt[1].shape == (10, 2)
    assert Xt[2].shape == (10, 1)


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_make_column_transformer(backend):
    backend = set_backend(backend)

    trans = StandardScaler()
    ct = make_column_transformer_no_stack((trans, slice(0, 3)))

    assert isinstance(ct, ColumnTransformerNoStack)
    assert len(ct.transformers) == 1
    assert len(ct.transformers[0]) == 3
    assert ct.transformers[0][0] == "standardscaler"
    assert ct.transformers[0][1] == trans
    assert ct.transformers[0][2] == slice(0, 3)

    trans = StandardScaler()
    ct = make_column_transformer_no_stack((trans, slice(0, 3)),
                                          ("passthrough", [3, 4]))

    assert isinstance(ct, ColumnTransformerNoStack)
    assert len(ct.transformers) == 2
    assert len(ct.transformers[0]) == 3
    assert len(ct.transformers[1]) == 3
    assert ct.transformers[0][0] == "standardscaler"
    assert ct.transformers[0][1] == trans
    assert ct.transformers[0][2] == slice(0, 3)
    assert ct.transformers[1][0] == "passthrough"
    assert ct.transformers[1][1] == "passthrough"
    assert ct.transformers[1][2] == [3, 4]


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_column_transformer_in_pipeline(backend):
    backend = set_backend(backend)

    X = np.random.randn(10, 5)
    Y = np.random.randn(10, 3)

    ct = make_column_transformer_no_stack(
        (StandardScaler(), slice(0, 4)),
        (StandardScaler(), slice(4, 6)),
    )
    pipe = make_pipeline(
        ct,
        GroupRidgeCV(
            groups="input", solver_params=dict(n_iter=np.ones((1, 2)),
                                               progress_bar=False)))
    pipe.fit(X, Y)
