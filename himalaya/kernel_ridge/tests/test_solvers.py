import pytest

import numpy as np
import sklearn.linear_model
import scipy.linalg

from himalaya.backend import set_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.utils import assert_array_almost_equal

from himalaya.kernel_ridge import WEIGHTED_KERNEL_RIDGE_SOLVERS
from himalaya.kernel_ridge import KERNEL_RIDGE_SOLVERS
from himalaya.kernel_ridge._solvers import _weighted_kernel_ridge_gradient


def _create_dataset(backend):
    n_samples, n_targets = 30, 3

    Xs = [
        backend.asarray(backend.randn(n_samples, n_features), backend.float64)
        for n_features in [100, 200]
    ]
    Ks = backend.stack([backend.matmul(X, X.T) for X in Xs])
    Y = backend.asarray(backend.randn(n_samples, n_targets), backend.float64)
    dual_weights = backend.asarray(backend.randn(n_samples, n_targets),
                                   backend.float64)
    exp_deltas = backend.asarray(backend.rand(Ks.shape[0], n_targets),
                                 backend.float64)
    deltas = backend.log(exp_deltas)

    return Xs, Ks, Y, deltas, dual_weights


@pytest.mark.parametrize("double_K", [False, True])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_weighted_kernel_ridge_gradient(backend, double_K):
    backend = set_backend(backend)

    _, Ks, Y, deltas, dual_weights = _create_dataset(backend)
    exp_deltas = backend.exp(deltas)
    alpha = 1.

    n_targets = Y.shape[1]
    grad = backend.zeros_like(dual_weights, dtype=backend.float64)
    func = backend.zeros_like(dual_weights, dtype=backend.float64,
                              shape=(n_targets))
    for tt in range(n_targets):
        K = backend.sum(
            backend.stack([K * g for K, g in zip(Ks, exp_deltas[:, tt])]), 0)
        grad[:, tt] = (backend.matmul(K, dual_weights[:, tt]) - Y[:, tt] +
                       alpha * dual_weights[:, tt])

        pred = backend.matmul(K, dual_weights[:, tt])
        func[tt] = backend.sum((pred - Y[:, tt]) ** 2, 0)
        func[tt] += alpha * (dual_weights[:, tt] @ K @ dual_weights[:, tt])

        if double_K:
            grad[:, tt] = backend.matmul(K.T, grad[:, tt])

    ########################
    grad2, func2 = _weighted_kernel_ridge_gradient(Ks, Y, dual_weights,
                                                   exp_deltas=exp_deltas,
                                                   alpha=alpha,
                                                   double_K=double_K,
                                                   return_objective=True)
    assert_array_almost_equal(grad, grad2, decimal=1e-3)
    assert_array_almost_equal(func, func2, decimal=1e-3)


@pytest.mark.parametrize('solver_name', WEIGHTED_KERNEL_RIDGE_SOLVERS)
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_weighted_kernel_ridge(solver_name, backend):
    backend = set_backend(backend)

    solver = WEIGHTED_KERNEL_RIDGE_SOLVERS[solver_name]
    decimal = 1 if solver_name == "neumann_series" else 3

    Xs, Ks, Y, deltas, dual_weights = _create_dataset(backend)
    exp_deltas = backend.exp(deltas)

    for alpha in backend.asarray_like(backend.logspace(-2, 3, 7), Ks):
        c2 = solver(Ks, Y, deltas, alpha=alpha, max_iter=3000, tol=1e-6)

        n_targets = Y.shape[1]
        for ii in range(n_targets):
            # compare dual coefficients with scipy.linalg.solve
            K = backend.matmul(Ks.T, exp_deltas[:, ii]).T
            reg = backend.asarray_like(np.eye(K.shape[0]), K) * alpha
            c1 = scipy.linalg.solve(backend.to_numpy(K + reg),
                                    backend.to_numpy(Y[:, ii]))
            assert_array_almost_equal(c1, c2[:, ii], decimal=decimal)

            if solver_name != "neumann_series":
                # compare predictions with sklearn.linear_model.Ridge
                X_scaled = backend.concatenate([
                    t * backend.sqrt(g) for t, g in zip(Xs, exp_deltas[:, ii])
                ], 1)
                prediction = backend.matmul(K, c2[:, ii])
                model = sklearn.linear_model.Ridge(
                    alpha=backend.to_numpy(alpha), solver="lsqr",
                    max_iter=1000, tol=1e-6, fit_intercept=False)
                model.fit(backend.to_numpy(X_scaled),
                          backend.to_numpy(Y[:, ii]))
                prediction_sklearn = model.predict(backend.to_numpy(X_scaled))
                assert_array_almost_equal(prediction, prediction_sklearn,
                                          decimal=decimal)


@pytest.mark.parametrize('solver_name', KERNEL_RIDGE_SOLVERS)
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_kernel_ridge(solver_name, backend):
    backend = set_backend(backend)

    Xs, Ks, Y, deltas, dual_weights = _create_dataset(backend)
    alphas = backend.asarray_like(backend.logspace(-2, 5, 7), Ks)

    solver = KERNEL_RIDGE_SOLVERS[solver_name]

    deltas = deltas[:, 0]
    exp_deltas = backend.exp(deltas)
    K = backend.matmul(Ks.T, exp_deltas).T

    for alpha in alphas:
        alpha = backend.full_like(K, fill_value=alpha, shape=Y.shape[1])
        if solver_name == "eigenvalues":
            c2 = solver(K, Y, alpha=alpha)
        else:
            c2 = solver(K, Y, alpha=alpha, max_iter=3000, tol=1e-6)

        n_targets = Y.shape[1]
        for ii in range(n_targets):
            # compare dual coefficients with scipy.linalg.solve
            reg = backend.asarray_like(np.eye(K.shape[0]), K) * alpha[ii]
            c1 = scipy.linalg.solve(backend.to_numpy(K + reg),
                                    backend.to_numpy(Y[:, ii]))
            assert_array_almost_equal(c1, c2[:, ii], decimal=3)

            # compare predictions with sklearn.linear_model.Ridge
            X_scaled = backend.concatenate(
                [t * backend.sqrt(g) for t, g in zip(Xs, exp_deltas)], 1)
            prediction = backend.matmul(K, c2[:, ii])
            model = sklearn.linear_model.Ridge(
                alpha=backend.to_numpy(alpha[ii]), solver="lsqr",
                max_iter=1000, tol=1e-6, fit_intercept=False)
            model.fit(backend.to_numpy(X_scaled), backend.to_numpy(Y[:, ii]))
            prediction_sklearn = model.predict(backend.to_numpy(X_scaled))
            assert_array_almost_equal(prediction, prediction_sklearn,
                                      decimal=3)
