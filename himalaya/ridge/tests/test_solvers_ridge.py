import pytest

import numpy as np
import sklearn.linear_model
import scipy.linalg

from himalaya.backend import set_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.utils import assert_array_almost_equal

from himalaya.ridge import RIDGE_SOLVERS


def _create_dataset(backend, many_targets=False):
    if many_targets:
        n_samples, n_features, n_targets = 10, 5, 20
    else:
        n_samples, n_features, n_targets = 30, 10, 3

    X = backend.asarray(backend.randn(n_samples, n_features), backend.float64)
    Y = backend.asarray(backend.randn(n_samples, n_targets), backend.float64)
    weights = backend.asarray(backend.randn(n_features, n_targets),
                              backend.float64)

    return X, Y, weights


@pytest.mark.parametrize('many_targets', [False, True])
@pytest.mark.parametrize('solver_name', RIDGE_SOLVERS)
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_kernel_ridge(solver_name, backend, many_targets):
    backend = set_backend(backend)

    X, Y, weights = _create_dataset(backend, many_targets=many_targets)
    alphas = backend.asarray_like(backend.logspace(-2, 5, 7), Y)

    solver = RIDGE_SOLVERS[solver_name]
    XTX = X.T @ X
    XTY = X.T @ Y

    for alpha in alphas:
        alpha = backend.full_like(Y, fill_value=alpha, shape=Y.shape[1])
        b2 = solver(X, Y, alpha=alpha, fit_intercept=False)
        b2 = backend.to_gpu(b2)
        assert b2.shape == (X.shape[1], Y.shape[1])

        n_features, n_targets = weights.shape
        for ii in range(n_targets):
            # compare primal coefficients with scipy.linalg.solve
            reg = backend.asarray_like(np.eye(n_features), Y) * alpha[ii]
            b1 = scipy.linalg.solve(backend.to_numpy(XTX + reg),
                                    backend.to_numpy(XTY[:, ii]))
            assert_array_almost_equal(b1, b2[:, ii], decimal=6)

            # compare predictions with sklearn.linear_model.Ridge
            prediction = backend.matmul(X, b2[:, ii])
            model = sklearn.linear_model.Ridge(
                alpha=backend.to_numpy(alpha[ii]), max_iter=1000, tol=1e-6,
                fit_intercept=False)
            model.fit(backend.to_numpy(X), backend.to_numpy(Y[:, ii]))
            prediction_sklearn = model.predict(backend.to_numpy(X))
            assert_array_almost_equal(prediction, prediction_sklearn,
                                      decimal=6)

            assert_array_almost_equal(model.coef_, b2[:, ii], decimal=5)


@pytest.mark.parametrize('solver_name', RIDGE_SOLVERS)
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_kernel_ridge_intercept(solver_name, backend):
    backend = set_backend(backend)

    X, Y, weights = _create_dataset(backend)
    Y += 100
    X += 10
    alphas = backend.asarray_like(backend.logspace(-2, 5, 7), Y)

    solver = RIDGE_SOLVERS[solver_name]

    for alpha in alphas:
        alpha = backend.full_like(Y, fill_value=alpha, shape=Y.shape[1])
        b2, i2 = solver(X, Y, alpha=alpha, fit_intercept=True)
        assert b2.shape == (X.shape[1], Y.shape[1])
        assert i2.shape == (Y.shape[1], )
        b2 = backend.to_gpu(b2)
        i2 = backend.to_gpu(i2)

        n_features, n_targets = weights.shape
        for ii in range(n_targets):

            # compare predictions with sklearn.linear_model.Ridge
            prediction = backend.matmul(X, b2[:, ii]) + i2[ii]
            model = sklearn.linear_model.Ridge(
                alpha=backend.to_numpy(alpha[ii]), max_iter=1000, tol=1e-6,
                fit_intercept=True)
            model.fit(backend.to_numpy(X), backend.to_numpy(Y[:, ii]))
            prediction_sklearn = model.predict(backend.to_numpy(X))
            assert_array_almost_equal(prediction, prediction_sklearn,
                                      decimal=5)

            assert_array_almost_equal(model.coef_, b2[:, ii], decimal=5)


@pytest.mark.parametrize('solver_name', RIDGE_SOLVERS)
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_warning_kernel_ridge_ridge(solver_name, backend):
    backend = set_backend(backend)
    X, Y, weights = _create_dataset(backend)
    solver = RIDGE_SOLVERS[solver_name]

    with pytest.warns(UserWarning,
                      match="ridge is slower than solving kernel"):
        solver(X[:4], Y[:4])


@pytest.mark.parametrize('solver_name', RIDGE_SOLVERS)
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_different_number_of_samples(solver_name, backend):
    backend = set_backend(backend)
    X, Y, weights = _create_dataset(backend)
    solver = RIDGE_SOLVERS[solver_name]

    with pytest.raises(ValueError, match="same number of samples"):
        solver(X[:4], Y[:3])
