import pytest

import sklearn.linear_model

from himalaya.backend import set_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.utils import assert_array_almost_equal

from himalaya.lasso import solve_group_lasso


def _create_dataset(backend):
    n_samples, n_features, n_targets = 10, 5, 3

    X = backend.asarray(backend.randn(n_samples, n_features), backend.float64)
    Y = backend.asarray(backend.randn(n_samples, n_targets), backend.float64)

    return X, Y


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_group_lasso_vs_ols(backend):
    backend = set_backend(backend)
    X, Y = _create_dataset(backend)

    coef = solve_group_lasso(X, Y, groups=None, l21_reg=0.0, l1_reg=0.0,
                             max_iter=1000, tol=1e-8, progress_bar=False)

    ols = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(
        backend.to_numpy(X), backend.to_numpy(Y))
    assert_array_almost_equal(coef, ols.coef_.T, decimal=4)


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_group_lasso_decreasing(backend):
    backend = set_backend(backend)
    X, Y = _create_dataset(backend)

    coef, losses = solve_group_lasso(X, Y, max_iter=500, tol=1e-8,
                                     progress_bar=False, debug=True,
                                     momentum=False)

    assert backend.all(losses[1:] - losses[:-1] < 1e-14)


@pytest.mark.parametrize('n_targets_batch', [None, 2])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_group_lasso_vs_lasso(backend, n_targets_batch):
    backend = set_backend(backend)
    X, Y = _create_dataset(backend)

    for l1_reg in backend.logspace(-5, 5, 5):

        coef = solve_group_lasso(X, Y, groups=None, l21_reg=0.0, l1_reg=l1_reg,
                                 max_iter=1000, tol=1e-8, progress_bar=False,
                                 debug=False, momentum=False,
                                 n_targets_batch=n_targets_batch)

        ols = sklearn.linear_model.Lasso(fit_intercept=False, alpha=l1_reg,
                                         max_iter=1000,
                                         tol=1e-8).fit(backend.to_numpy(X),
                                                       backend.to_numpy(Y))
        assert_array_almost_equal(coef, ols.coef_.T)
