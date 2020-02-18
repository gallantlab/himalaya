import pytest

import numpy as np
import sklearn.linear_model
import sklearn.model_selection
import scipy.linalg

from himalaya.backend import change_backend
from himalaya.backend import ALL_BACKENDS

from himalaya.scoring import r2_score
from himalaya.regression.multiple_kernel_ridge import solve_multiple_kernel_ridge_random_search  # noqa


def _create_dataset(backend):
    n_featuress = (100, 200)
    n_samples = 80
    n_targets = 4
    n_gammas = 2

    Xs = [
        backend.asarray(backend.randn(n_samples, n_features), backend.float64)
        for n_features in n_featuress
    ]
    Ks = backend.stack([X @ X.T for X in Xs])

    ws = [
        backend.asarray(backend.randn(n_features, n_targets), backend.float64)
        for n_features in n_featuress
    ]
    Ys = backend.stack([X @ w for X, w in zip(Xs, ws)])
    Y = Ys.sum(0)

    gammas = backend.asarray(backend.rand(n_gammas, Ks.shape[0]),
                             backend.float64)

    return Ks, Y, gammas, Xs


@pytest.mark.parametrize('n_targets_batch', [None, 3])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_multiple_kernel_ridge_random_search_n_targets_batch(
        backend, n_targets_batch):
    _test_solve_multiple_kernel_ridge_random_search(
        backend=backend, n_targets_batch=n_targets_batch)


@pytest.mark.parametrize('n_alphas_batch', [None, 2])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_multiple_kernel_ridge_random_search_n_alphas_batch(
        backend, n_alphas_batch):
    _test_solve_multiple_kernel_ridge_random_search(
        backend=backend, n_alphas_batch=n_alphas_batch)


@pytest.mark.parametrize('compute_weights', ['primal', 'dual'])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_multiple_kernel_ridge_random_search_compute_weights(
        backend, compute_weights):
    _test_solve_multiple_kernel_ridge_random_search(
        backend=backend, compute_weights=compute_weights)


def _test_solve_multiple_kernel_ridge_random_search(backend,
                                                    n_targets_batch=None,
                                                    n_alphas_batch=None,
                                                    compute_weights="dual"):
    backend = change_backend(backend)

    Ks, Y, gammas, Xs = _create_dataset(backend)
    alphas = backend.asarray(backend.logspace(-3, 5, 9), backend.float64)
    n_targets = Y.shape[1]
    cv = sklearn.model_selection.check_cv(10)

    ############
    # run solver
    results = solve_multiple_kernel_ridge_random_search(
        Ks, Y, gammas, alphas, score_func=r2_score, cv_splitter=cv,
        n_targets_batch=n_targets_batch, Xs=Xs,
        compute_weights=compute_weights, n_alphas_batch=n_alphas_batch)
    all_scores_mean, best_gammas, best_alphas, refit_weights = results

    #########################################
    # compare with sklearn.linear_model.Ridge
    test_scores = []
    for gamma in backend.sqrt(gammas):
        X = backend.concatenate([x * g for x, g in zip(Xs, gamma)], 1)
        for train, test in cv.split(X):
            for alpha in alphas:
                model = sklearn.linear_model.Ridge(alpha=alpha,
                                                   fit_intercept=False)
                model = model.fit(X[train], Y[train])
                predictions = backend.asarray(model.predict(X[test]),
                                              dtype=Y.dtype)
                test_scores.append(r2_score(Y[test], predictions))

    test_scores = backend.stack(test_scores)
    test_scores = test_scores.reshape(len(gammas), cv.get_n_splits(),
                                      len(alphas), n_targets)
    test_scores_mean = backend.max(test_scores.mean(1), 1)
    backend.assert_allclose(all_scores_mean, test_scores_mean, rtol=1e-5,
                            atol=1e-7)

    ###############################################
    # test best_primal_weights or best_dual_weights
    for tt in range(n_targets):
        gamma = best_gammas[:, tt]
        alpha = best_alphas[tt]

        if compute_weights == 'primal':
            # compare primal weights with sklearn.linear_model.Ridge
            X = backend.concatenate(
                [t * backend.sqrt(g) for t, g in zip(Xs, gamma)], 1)
            model = sklearn.linear_model.Ridge(fit_intercept=False,
                                               alpha=alpha)
            w1 = model.fit(X, Y[:, tt]).coef_
            w1 = np.split(w1, [X.shape[1] for X in Xs][:-1], axis=0)
            w1 = [backend.asarray(w) for w in w1]
            w1_scaled = backend.concatenate(
                [w * backend.sqrt(g) for w, g, in zip(w1, gamma)])
            backend.assert_allclose(w1_scaled, refit_weights[:, tt], rtol=1e-5,
                                    atol=1e-7)

        elif compute_weights == 'dual':
            # compare dual weights with scipy.linalg.solve
            Ks_64 = backend.asarray(Ks, dtype=backend.float64)
            gamma_64 = backend.asarray(gamma, dtype=backend.float64)
            K = backend.matmul(Ks_64.T, gamma_64).T
            K_reg = K + backend.eye(K.shape[0], dtype=backend.float64) * alpha
            Y_64 = backend.asarray(Y, dtype=backend.float64)
            c1 = scipy.linalg.solve(K_reg, Y_64[:, tt])
            c1 = backend.asarray(c1, dtype=refit_weights.dtype)
            backend.assert_allclose(c1, refit_weights[:, tt], rtol=1e-5,
                                    atol=1e-7)
