import warnings

import pytest
import sklearn.kernel_ridge

from himalaya.backend import set_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.utils import assert_array_almost_equal

from himalaya.kernel_ridge import KernelRidge


def _create_dataset(backend):
    n_samples, n_targets = 30, 3

    Xs = [
        backend.asarray(backend.randn(n_samples, n_features), backend.float64)
        for n_features in [100, 200]
    ]
    Ks = backend.stack([backend.matmul(X, X.T) for X in Xs])
    Y = backend.asarray(backend.randn(n_samples, n_targets), backend.float64)
    exp_deltas = backend.asarray(backend.rand(Ks.shape[0], n_targets),
                                 backend.float64)
    deltas = backend.log(exp_deltas)

    return Xs, Ks, Y, deltas


@pytest.mark.parametrize('kernel', [
    'linear', 'polynomial', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'
])
@pytest.mark.parametrize('multitarget', [True, False])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_vs_scikit_learn(backend, multitarget, kernel):
    backend = set_backend(backend)
    Xs, Ks, Y, _ = _create_dataset(backend)

    if not multitarget:
        Y = Y[:, 0]

    if kernel == "precomputed":
        X = Ks[0]
    else:
        X = Xs[0]

    for alpha in backend.asarray_like(backend.logspace(0, 3, 7), Ks):
        model = KernelRidge(alpha=alpha, kernel=kernel)
        model.fit(X, Y)

        reference = sklearn.kernel_ridge.KernelRidge(
            alpha=backend.to_numpy(alpha), kernel=kernel)
        reference.fit(backend.to_numpy(X), backend.to_numpy(Y))

        assert model.dual_coef_.shape == Y.shape
        assert_array_almost_equal(model.dual_coef_, reference.dual_coef_)


@pytest.mark.parametrize(
    'kernel', ['linear', 'polynomial', 'poly', 'rbf', 'sigmoid', 'cosine'])
@pytest.mark.parametrize('format', ['coo', 'csr', 'csc'])
def test_kernel_ridge_vs_scikit_learn_sparse(kernel, format):
    backend = set_backend("numpy")
    Xs, _, Y, _ = _create_dataset(backend)

    try:
        import scipy.sparse
    except ImportError:
        pytest.skip("Scipy is not installed.")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', scipy.sparse.SparseEfficiencyWarning)
        X = scipy.sparse.rand(*Xs[0].shape, density=0.1, format=format)

    for alpha in backend.asarray_like(backend.logspace(0, 3, 7), Y):
        model = KernelRidge(alpha=alpha, kernel=kernel)
        model.fit(X, Y)

        reference = sklearn.kernel_ridge.KernelRidge(
            alpha=backend.to_numpy(alpha), kernel=kernel)
        reference.fit(backend.to_numpy(X), backend.to_numpy(Y))

        assert model.dual_coef_.shape == Y.shape
        assert_array_almost_equal(model.dual_coef_, reference.dual_coef_)


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_precomputed(backend):
    backend = set_backend(backend)
    Xs, Ks, Y, _ = _create_dataset(backend)

    for alpha in backend.asarray_like(backend.logspace(-2, 3, 7), Ks):
        model_1 = KernelRidge(alpha=alpha, kernel="linear")
        model_1.fit(Xs[0], Y)
        model_2 = KernelRidge(alpha=alpha, kernel="precomputed")
        model_2.fit(Ks[0], Y)

        assert_array_almost_equal(model_1.dual_coef_, model_2.dual_coef_)


@pytest.mark.parametrize('solver', ['eigenvalues', 'conjugate', 'gradient'])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_solvers(solver, backend):
    backend = set_backend("numpy")
    Xs, _, Y, _ = _create_dataset(backend)

    kernel = "linear"
    X = Xs[0]

    if solver == "eigenvalues":
        solver_params = dict()
    elif solver == "conjugate":
        solver_params = dict(max_iter=300, tol=1e-6)
    elif solver == "gradient":
        solver_params = dict(max_iter=300, tol=1e-6)

    for alpha in backend.asarray_like(backend.logspace(0, 3, 7), Y):
        model = KernelRidge(alpha=alpha, kernel=kernel, solver=solver,
                            solver_params=solver_params)
        model.fit(X, Y)

        reference = sklearn.kernel_ridge.KernelRidge(
            alpha=backend.to_numpy(alpha), kernel=kernel)
        reference.fit(backend.to_numpy(X), backend.to_numpy(Y))

        assert model.dual_coef_.shape == Y.shape
        assert_array_almost_equal(model.dual_coef_, reference.dual_coef_)
