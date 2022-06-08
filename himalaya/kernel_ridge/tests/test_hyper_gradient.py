import pytest

import numpy as np
import sklearn.linear_model
import sklearn.model_selection
import scipy.linalg

from himalaya.backend import set_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.scoring import r2_score
from himalaya.utils import assert_array_almost_equal

from himalaya.kernel_ridge._hyper_gradient import _compute_delta_gradient
from himalaya.kernel_ridge._hyper_gradient import _compute_delta_loss
from himalaya.kernel_ridge import solve_multiple_kernel_ridge_hyper_gradient
from himalaya.kernel_ridge import solve_multiple_kernel_ridge_random_search
from himalaya.kernel_ridge import solve_weighted_kernel_ridge_conjugate_gradient  # noqa
from himalaya.kernel_ridge import generate_dirichlet_samples
from himalaya.kernel_ridge import predict_and_score_weighted_kernel_ridge


def _create_dataset(backend):
    n_featuress = [100, 200, 150]
    n_samples_train = 80
    n_samples_val = 20
    n_targets = 4

    Xs = [
        backend.asarray(backend.randn(n_samples_train, n_features),
                        backend.float64) for n_features in n_featuress
    ]
    Ks = backend.stack([X @ X.T for X in Xs])
    Xs_val = [
        backend.asarray(backend.randn(n_samples_val, n_features),
                        backend.float64) for n_features in n_featuress
    ]
    Ks_val = backend.stack([X_val @ X.T for X, X_val in zip(Xs, Xs_val)])

    true_gammas = backend.asarray(backend.rand(len(Xs), n_targets),
                                  backend.float64)

    ws = [
        backend.asarray(backend.randn(n_features, n_targets), backend.float64)
        for n_features in n_featuress
    ]
    Ys = backend.stack(
        [X @ w * backend.sqrt(g) for X, w, g in zip(Xs, ws, true_gammas)])
    Y = Ys.sum(0)
    Ys_val = backend.stack(
        [X @ w * backend.sqrt(g) for X, w, g in zip(Xs_val, ws, true_gammas)])
    Y_val = Ys_val.sum(0)

    gammas = backend.asarray(backend.rand(len(Xs), n_targets), backend.float64)

    dual_weights = backend.asarray(backend.randn(*Y.shape), backend.float64)

    return Ks, Y, dual_weights, gammas, Ks_val, Y_val, Xs


@pytest.mark.parametrize('n_targets_batch', [None, 3])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_delta_gradient_direct(backend, n_targets_batch):
    backend = set_backend(backend)

    Ks, Y, dual_weights, gammas, Ks_val, Y_val, _ = _create_dataset(backend)
    alphas = backend.asarray_like(backend.logspace(-1, 1, Y.shape[1]), Ks)
    deltas = backend.log(gammas / alphas)
    epsilons = backend.asarray_like(backend.randn(*deltas.shape), Ks)
    epsilons /= backend.norm(epsilons, axis=1)[:, None]
    step = 0.0000001
    deltas2 = deltas + epsilons * step

    # check direct gradient with a finite difference
    gradients = _compute_delta_gradient(Ks, Y, deltas, dual_weights,
                                        hyper_gradient_method='direct')[0]
    scores = _compute_delta_loss(Ks, Y, deltas, dual_weights)
    scores2 = _compute_delta_loss(Ks, Y, deltas2, dual_weights)

    directional_derivatives = (scores2 - scores) / step
    gradient_direction_product = (gradients * epsilons[:, :]).sum(0)
    norm = backend.norm(gradient_direction_product)

    assert_array_almost_equal(gradient_direction_product / norm,
                              directional_derivatives / norm, decimal=5)


@pytest.mark.parametrize('n_targets_batch', [None, 3])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_delta_gradient_indirect(backend, n_targets_batch):
    backend = set_backend(backend)

    Ks, Y, _, gammas, Ks_val, Y_val, _ = _create_dataset(backend)
    alphas = backend.asarray_like(backend.logspace(-1, 1, Y.shape[1]), Ks)
    deltas = backend.log(gammas / alphas)
    epsilons = backend.asarray_like(backend.randn(*deltas.shape), Ks)
    epsilons /= backend.norm(epsilons, axis=1)[:, None]
    step = 0.0000001
    deltas2 = deltas + epsilons * step

    # check direct and indirect gradient with a finite difference
    # to get the indirect gradient, we need to refit the kernel ridge during
    # the validation loss computation.

    def score_func(Y_val, Ypred):
        return 0.5 * backend.norm(Ypred - Y_val, axis=0) ** 2

    def compute_loss(deltas):
        dual_weights = solve_weighted_kernel_ridge_conjugate_gradient(
            Ks, Y, deltas=deltas, initial_dual_weights=None, alpha=1,
            max_iter=1000, tol=1e-5)
        loss = predict_and_score_weighted_kernel_ridge(Ks_val, dual_weights,
                                                       deltas, Y_val,
                                                       score_func=score_func)
        return loss, dual_weights

    loss, dual_weights = compute_loss(deltas)
    loss2, dual_weights2 = compute_loss(deltas2)

    gradients = _compute_delta_gradient(
        Ks_val, Y_val, deltas, dual_weights, Ks_train=Ks,
        hyper_gradient_method='conjugate_gradient', tol=1e-5)[0]

    directional_derivatives = (loss2 - loss) / step
    gradient_direction_product = (gradients * epsilons[:, :]).sum(0)
    norm = backend.norm(gradient_direction_product)

    assert_array_almost_equal(gradient_direction_product / norm,
                              directional_derivatives / norm, decimal=4)


@pytest.mark.parametrize('n_targets_batch', [None, 3])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_multiple_kernel_ridge_hyper_gradient_n_targets_batch(
        backend, n_targets_batch):
    _test_solve_multiple_kernel_ridge_hyper_gradient(
        backend=backend, n_targets_batch=n_targets_batch)


@pytest.mark.parametrize('method', ["direct", "conjugate_gradient", "neumann"])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_multiple_kernel_ridge_hyper_gradient_method(backend, method):
    _test_solve_multiple_kernel_ridge_hyper_gradient(backend=backend,
                                                     method=method)


@pytest.mark.parametrize('initial_deltas', [0, 5, 'ridgecv'])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_multiple_kernel_ridge_hyper_gradient_initial_deltas(
        backend, initial_deltas):
    _test_solve_multiple_kernel_ridge_hyper_gradient(
        backend=backend, initial_deltas=initial_deltas)


@pytest.mark.parametrize('kernel_ridge',
                         ["conjugate_gradient", "gradient_descent"])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_multiple_kernel_ridge_hyper_gradient_kernel_ridge(
        backend, kernel_ridge):
    _test_solve_multiple_kernel_ridge_hyper_gradient(backend=backend,
                                                     kernel_ridge=kernel_ridge)


def _test_solve_multiple_kernel_ridge_hyper_gradient(
        backend, n_targets_batch=None, method="direct", initial_deltas=0,
        kernel_ridge="conjugate_gradient"):
    backend = set_backend(backend)
    Ks, Y, dual_weights, gammas, Ks_val, Y_val, Xs = _create_dataset(backend)
    cv = 3
    progress_bar = False

    # compare bilinear gradient descent and dirichlet sampling
    alphas = backend.logspace(-5, 5, 11)
    gammas = generate_dirichlet_samples(50, len(Ks), concentration=[.1, 1.],
                                        random_state=0)
    _, _, cv_scores = \
        solve_multiple_kernel_ridge_random_search(
            Ks, Y, gammas, alphas, n_targets_batch=n_targets_batch,
            score_func=r2_score, cv=cv, progress_bar=progress_bar)
    scores_2 = backend.max(backend.asarray(cv_scores), axis=0)

    max_iter = 10
    for _ in range(5):
        try:
            _, _, cv_scores = \
                solve_multiple_kernel_ridge_hyper_gradient(
                    Ks, Y, max_iter=max_iter, n_targets_batch=n_targets_batch,
                    max_iter_inner_dual=1, max_iter_inner_hyper=1, tol=None,
                    score_func=r2_score, cv=cv,
                    hyper_gradient_method=method,
                    initial_deltas=initial_deltas,
                    kernel_ridge_method=kernel_ridge,
                    progress_bar=progress_bar)
            cv_scores = backend.asarray(cv_scores)
            scores_1 = cv_scores[cv_scores.sum(axis=1) != 0][-1]

            assert_array_almost_equal(scores_1, scores_2, decimal=1)
            break
        except AssertionError:
            max_iter *= 5
    else:
        raise AssertionError


@pytest.mark.parametrize('n_targets_batch', [None, 3])
@pytest.mark.parametrize('return_weights', ['primal', 'dual'])
@pytest.mark.parametrize('method', ['hyper_gradient', 'random_search'])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_multiple_kernel_ridge_return_weights(backend, method,
                                                    return_weights,
                                                    n_targets_batch):
    backend = set_backend(backend)

    Ks, Y, _, _, Ks_val, Y_val, Xs = _create_dataset(backend)
    n_targets = Y.shape[1]
    cv = sklearn.model_selection.check_cv(10)

    ############
    # run solver
    if method == "hyper_gradient":
        results = solve_multiple_kernel_ridge_hyper_gradient(
            Ks, Y, score_func=r2_score, cv=cv, max_iter=1,
            n_targets_batch=n_targets_batch, Xs=Xs, progress_bar=False,
            return_weights=return_weights)
        best_deltas, refit_weights, cv_scores = results
    elif method == "random_search":
        alphas = backend.asarray_like(backend.logspace(-3, 5, 2), Ks)
        results = solve_multiple_kernel_ridge_random_search(
            Ks, Y, n_iter=1, alphas=alphas, score_func=r2_score, cv=cv,
            n_targets_batch=n_targets_batch, Xs=Xs, progress_bar=False,
            return_weights=return_weights)
        best_deltas, refit_weights, cv_scores = results
    else:
        raise ValueError("Unknown parameter method=%r." % (method, ))

    ######################
    # test refited_weights
    for tt in range(n_targets):
        gamma = backend.exp(best_deltas[:, tt])
        alpha = 1.0

        if return_weights == 'primal':
            # compare primal weights with sklearn.linear_model.Ridge
            X = backend.concatenate(
                [t * backend.sqrt(g) for t, g in zip(Xs, gamma)], 1)
            model = sklearn.linear_model.Ridge(fit_intercept=False,
                                               alpha=backend.to_numpy(alpha))
            w1 = model.fit(backend.to_numpy(X),
                           backend.to_numpy(Y[:, tt])).coef_
            w1 = np.split(w1, np.cumsum([X.shape[1] for X in Xs][:-1]), axis=0)
            w1 = [backend.asarray(w) for w in w1]
            w1_scaled = backend.concatenate(
                [w * backend.sqrt(g) for w, g, in zip(w1, gamma)])
            assert_array_almost_equal(w1_scaled, refit_weights[:, tt],
                                      decimal=5)

        elif return_weights == 'dual':
            # compare dual weights with scipy.linalg.solve
            Ks_64 = backend.asarray(Ks, dtype=backend.float64)
            gamma_64 = backend.asarray(gamma, dtype=backend.float64)
            K = backend.matmul(Ks_64.T, gamma_64).T
            reg = backend.asarray_like(np.eye(K.shape[0]), K) * alpha
            Y_64 = backend.asarray(Y, dtype=backend.float64)
            c1 = scipy.linalg.solve(backend.to_numpy(K + reg),
                                    backend.to_numpy(Y_64[:, tt]))
            c1 = backend.asarray_like(c1, K)
            assert_array_almost_equal(c1, refit_weights[:, tt], decimal=5)


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_hyper_gradient_early_stopping(backend):
    # Non-regression test for https://github.com/gallantlab/himalaya/pull/37
    backend = set_backend(backend)
    Ks, Y, _, gammas, _, _, _ = _create_dataset(backend)

    # to reproduce, need different early stooping times for different batches
    n_targets_batch = 2
    # trigger early stopping
    tol = 1e-1
    # good init for all targets for early early stopping
    initial_deltas = backend.log(gammas)
    # poor init for last target for later early stopping
    initial_deltas[:, -1] = 100

    _, _, cv_scores = solve_multiple_kernel_ridge_hyper_gradient(
        Ks, Y, max_iter=100, n_targets_batch=n_targets_batch,
        max_iter_inner_dual=1, max_iter_inner_hyper=1, tol=tol,
        score_func=r2_score, cv=3, hyper_gradient_method="direct",
        initial_deltas=initial_deltas, kernel_ridge_method="conjugate_gradient",
        progress_bar=False)
    cv_scores = backend.asarray(cv_scores)
