import itertools
import pytest

import sklearn.linear_model
import sklearn.model_selection

from himalaya.backend import set_backend
from himalaya.backend import ALL_BACKENDS

from himalaya.kernel_ridge import solve_multiple_kernel_ridge_random_search
from himalaya.kernel_ridge import solve_multiple_kernel_ridge_hyper_gradient


def _create_dataset(backend):
    n_featuress = (50, 80)
    n_samples = 30
    n_targets = 2
    n_gammas = 3

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
    gammas /= gammas.sum(1)[:, None]

    return Ks, Y, gammas


@pytest.mark.parametrize('Ks_in_cpu', [True, False])
@pytest.mark.parametrize('Y_in_cpu', [True, False])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_random_search(backend, Ks_in_cpu, Y_in_cpu):
    backend = set_backend(backend)

    Ks, Y, gammas = _create_dataset(backend)
    gammas = gammas[:1]
    alphas = backend.asarray_like(backend.logspace(-3, 5, 3), Ks)
    cv = sklearn.model_selection.check_cv(2)

    for Ks_, Y_, gammas_, alphas_ in itertools.product(
        [Ks, backend.to_numpy(Ks),
         backend.to_cpu(Ks)],
        [Y, backend.to_numpy(Y), backend.to_cpu(Y)],
        [gammas, backend.to_numpy(gammas),
         backend.to_cpu(gammas), 2],
        [alphas, backend.to_numpy(alphas),
         backend.to_cpu(alphas)],
    ):

        deltas, _, _ = solve_multiple_kernel_ridge_random_search(
            Ks_, Y_, n_iter=gammas_, alphas=alphas_, cv=cv, progress_bar=False,
            Ks_in_cpu=Ks_in_cpu, Y_in_cpu=Y_in_cpu)

        assert deltas.dtype == Ks.dtype
        assert getattr(deltas, "device", None) == getattr(Ks, "device", None)


@pytest.mark.parametrize('Y_in_cpu', [True, False])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_hyper_gradient(backend, Y_in_cpu):
    backend = set_backend(backend)

    Ks, Y, _ = _create_dataset(backend)
    cv = sklearn.model_selection.check_cv(2)

    for Ks_, Y_ in itertools.product(
        [Ks, backend.to_numpy(Ks),
         backend.to_cpu(Ks)],
        [Y, backend.to_numpy(Y), backend.to_cpu(Y)],
    ):

        deltas, _, _ = solve_multiple_kernel_ridge_hyper_gradient(
            Ks_, Y_, max_iter=1, cv=cv, progress_bar=False, Y_in_cpu=Y_in_cpu)

        assert deltas.dtype == Ks.dtype
        assert getattr(deltas, "device", None) == getattr(Ks, "device", None)
