import warnings

import pytest
import sklearn.kernel_ridge
import sklearn.utils.estimator_checks

from himalaya.backend import set_backend
from himalaya.backend import get_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.utils import assert_array_almost_equal

from himalaya.ridge import Ridge
from himalaya.ridge import RidgeCV
from himalaya.kernel_ridge import KernelRidge
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.kernel_ridge import MultipleKernelRidgeCV
from himalaya.kernel_ridge import WeightedKernelRidge


def _create_dataset(backend):
    n_samples, n_targets = 30, 3

    Xs = [
        backend.asarray(backend.randn(n_samples, n_features), backend.float64)
        for n_features in [100, 200]
    ]
    Ks = backend.stack([backend.matmul(X, X.T) for X in Xs])
    Y = backend.asarray(backend.randn(n_samples, n_targets), backend.float64)

    return Xs, Ks, Y


@pytest.mark.parametrize('kernel', [
    'linear', 'polynomial', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'
])
@pytest.mark.parametrize('multitarget', [True, False])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_vs_scikit_learn(backend, multitarget, kernel):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

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
        assert_array_almost_equal(model.predict(X),
                                  reference.predict(backend.to_numpy(X)))
        assert_array_almost_equal(
            model.score(X, Y).mean(),
            reference.score(backend.to_numpy(X), backend.to_numpy(Y)))


@pytest.mark.parametrize('fit_intercept', [False, True])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_vs_ridge(backend, fit_intercept):
    # useful to test the intercept as well
    backend = set_backend(backend)
    Xs, _, Y = _create_dataset(backend)
    X = Xs[0]
    if fit_intercept:
        Y += 10
        X += 1

    # torch with cuda has more limited precision in mean
    decimal = 3 if backend.name == "torch_cuda" else 6

    for alpha in backend.asarray_like(backend.logspace(0, 3, 7), X):
        model = KernelRidge(alpha=alpha, fit_intercept=fit_intercept)
        model.fit(X, Y)
        reference = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        reference.fit(backend.to_numpy(X), backend.to_numpy(Y))

        assert_array_almost_equal(model.predict(X), reference.predict(X),
                                  decimal=decimal)


@pytest.mark.parametrize('fit_intercept', [False, True])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_cv_vs_ridge_cv(backend, fit_intercept):
    # useful to test the intercept as well
    backend = set_backend(backend)
    Xs, _, Y = _create_dataset(backend)
    X = Xs[0]
    if fit_intercept:
        Y += 10
        Xs[0] += 1
    alphas = backend.asarray_like(backend.logspace(-2, 3, 21), Y)  # XXX

    # torch with cuda has more limited precision in mean
    decimal = 4 if backend.name == "torch_cuda" else 6

    model = KernelRidgeCV(alphas=alphas, fit_intercept=fit_intercept)
    model.fit(X, Y)
    reference = RidgeCV(alphas=alphas, fit_intercept=fit_intercept)
    reference.fit(X, Y)

    assert_array_almost_equal(model.best_alphas_, reference.best_alphas_,
                              decimal=5)
    assert_array_almost_equal(model.predict(X), reference.predict(X),
                              decimal=decimal)


@pytest.mark.parametrize(
    'kernel', ['linear', 'polynomial', 'poly', 'rbf', 'sigmoid', 'cosine'])
@pytest.mark.parametrize('format', ['coo', 'csr', 'csc'])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_vs_scikit_learn_sparse(backend, kernel, format):
    backend = set_backend(backend)
    Xs, _, Y = _create_dataset(backend)

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
        reference.fit(X, backend.to_numpy(Y))

        assert model.dual_coef_.shape == Y.shape
        assert_array_almost_equal(model.dual_coef_, reference.dual_coef_)
        assert_array_almost_equal(model.predict(X), reference.predict(X))
        assert_array_almost_equal(
            model.score(X, Y).mean(), reference.score(X, backend.to_numpy(Y)))


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_precomputed(backend):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    for alpha in backend.asarray_like(backend.logspace(-2, 3, 7), Ks):
        model_1 = KernelRidge(alpha=alpha, kernel="linear")
        model_1.fit(Xs[0], Y)
        model_2 = KernelRidge(alpha=alpha, kernel="precomputed")
        model_2.fit(Ks[0], Y)

        assert_array_almost_equal(model_1.dual_coef_, model_2.dual_coef_)
        assert_array_almost_equal(model_1.predict(Xs[0]),
                                  model_2.predict(Ks[0]))


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_get_primal_coef(backend):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    model = KernelRidge(kernel="linear")
    model.fit(Xs[0], Y)
    primal_coef = model.get_primal_coef()
    predictions_primal = Xs[0] @ backend.asarray(primal_coef)
    predictions_dual = model.predict(Xs[0])
    assert_array_almost_equal(predictions_primal, predictions_dual)

    model = KernelRidge(kernel="precomputed")
    model.fit(Ks[0], Y)
    primal_coef = model.get_primal_coef(X_fit=Xs[0])
    predictions_primal = Xs[0] @ backend.asarray(primal_coef)
    predictions_dual = model.predict(Ks[0])
    assert_array_almost_equal(predictions_primal, predictions_dual)

    model = KernelRidge(kernel="precomputed")
    model.fit(Ks[0], Y)
    with pytest.raises(ValueError):
        model.get_primal_coef(X_fit=None)

    model = KernelRidge(kernel="poly")
    model.fit(Xs[0], Y)
    with pytest.raises(ValueError):
        model.get_primal_coef()


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_weighted_kernel_ridge_get_primal_coef(backend):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    model = WeightedKernelRidge(kernels="precomputed")
    model.fit(Ks, Y)

    with pytest.raises(ValueError):
        primal_coef = model.get_primal_coef(Xs_fit=None)

    primal_coef = model.get_primal_coef(Xs_fit=Xs)
    predictions_primal = backend.stack(
        [X @ backend.asarray(w) for X, w in zip(Xs, primal_coef)]).sum(0)
    predictions_dual = model.predict(Ks)
    assert_array_almost_equal(predictions_primal, predictions_dual)


@pytest.mark.parametrize(
    'solver', ['eigenvalues', 'conjugate_gradient', 'gradient_descent'])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_solvers(solver, backend):
    backend = set_backend(backend)
    Xs, _, Y = _create_dataset(backend)

    kernel = "linear"
    X = Xs[0]

    if solver == "eigenvalues":
        solver_params = dict()
    elif solver == "conjugate_gradient":
        solver_params = dict(max_iter=300, tol=1e-6)
    elif solver == "gradient_descent":
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
        assert_array_almost_equal(model.predict(X),
                                  reference.predict(backend.to_numpy(X)),
                                  decimal=5)


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_wrong_solver(backend):
    backend = set_backend(backend)
    Xs, _, Y = _create_dataset(backend)
    X = Xs[0]

    model = KernelRidge(solver="wrong")
    with pytest.raises(ValueError, match="Unknown solver"):
        model.fit(X, Y)


@pytest.mark.parametrize('solver', ['eigenvalues'])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_cv_precomputed(backend, solver):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    model_1 = KernelRidgeCV(kernel="linear")
    model_1.fit(Xs[0], Y)
    model_2 = KernelRidgeCV(kernel="precomputed")
    model_2.fit(Ks[0], Y)

    assert_array_almost_equal(model_1.dual_coef_, model_2.dual_coef_)
    assert_array_almost_equal(model_1.predict(Xs[0]), model_2.predict(Ks[0]))


@pytest.mark.parametrize('solver', ['random_search', 'hyper_gradient'])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_multiple_kernel_ridge_cv_precomputed(backend, solver):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    if solver == "random_search":
        kwargs = dict(solver="random_search", random_state=0,
                      solver_params=dict(n_iter=2, progress_bar=False))
    elif solver == "hyper_gradient":
        kwargs = dict(solver="hyper_gradient", random_state=0,
                      solver_params=dict(max_iter=2, progress_bar=False))

    model_1 = MultipleKernelRidgeCV(kernels=["linear"], **kwargs)
    model_1.fit(Xs[0], Y)
    model_2 = MultipleKernelRidgeCV(kernels="precomputed", **kwargs)
    model_2.fit(Ks[0][None], Y)

    assert_array_almost_equal(model_1.dual_coef_, model_2.dual_coef_)
    assert_array_almost_equal(model_1.predict(Xs[0]),
                              model_2.predict(Ks[0][None]))


@pytest.mark.parametrize('solver', ['conjugate_gradient', 'gradient_descent'])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_weighted_kernel_ridge_precomputed(backend, solver):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    model_1 = WeightedKernelRidge(kernels=["linear"])
    model_1.fit(Xs[0], Y)
    model_2 = WeightedKernelRidge(kernels="precomputed")
    model_2.fit(Ks[0][None], Y)

    assert_array_almost_equal(model_1.dual_coef_, model_2.dual_coef_)
    assert_array_almost_equal(model_1.predict(Xs[0]),
                              model_2.predict(Ks[0][None]))


@pytest.mark.parametrize('Estimator',
                         [WeightedKernelRidge, MultipleKernelRidgeCV])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_weighted_kernel_ridge_split_predict(backend, Estimator):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    # multiple targets
    model = Estimator(kernels="precomputed")
    model.fit(Ks, Y)
    Y_pred = model.predict(Ks)
    Y_pred_split = model.predict(Ks, split=True)
    assert Y_pred.shape == Y.shape
    assert_array_almost_equal(Y_pred, Y_pred_split.sum(0))

    # single targets
    model = Estimator(kernels="precomputed")
    model.fit(Ks, Y[:, 0])
    Y_pred = model.predict(Ks)
    Y_pred_split = model.predict(Ks, split=True)
    assert Y_pred.shape == Y[:, 0].shape
    assert_array_almost_equal(Y_pred, Y_pred_split.sum(0))


@pytest.mark.parametrize('Estimator',
                         [WeightedKernelRidge, MultipleKernelRidgeCV])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_weighted_kernel_ridge_split_score(backend, Estimator):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    if issubclass(Estimator, MultipleKernelRidgeCV):
        solver_params = dict(n_iter=2, progress_bar=False)
    else:
        solver_params = dict()

    # multiple targets
    model = Estimator(kernels="precomputed", solver_params=solver_params)
    model.fit(Ks, Y)
    score = model.score(Ks, Y)
    score_split = model.score(Ks, Y, split=True)
    assert score_split.shape == (len(Ks), Y.shape[1])
    assert_array_almost_equal(score, score_split.sum(0), decimal=5)

    # single targets
    model = Estimator(kernels="precomputed", solver_params=solver_params)
    model.fit(Ks, Y[:, 0])
    score = model.score(Ks, Y[:, 0])
    score_split = model.score(Ks, Y[:, 0], split=True)
    assert score_split.shape == (len(Ks), )
    assert_array_almost_equal(score, score_split.sum(0), decimal=5)


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_duplicate_solver_parameters(backend):
    backend = set_backend(backend)
    Xs, _, Y = _create_dataset(backend)

    model = KernelRidge(solver_params=dict(alpha=1))
    with pytest.raises(ValueError):
        model.fit(Xs[0], Y)


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_cv_predict(backend):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    n_iter = backend.ones_like(Ks, shape=(1, Ks.shape[0]))
    alphas = backend.logspace(1, 2, 3)

    model_0 = KernelRidgeCV(kernel="precomputed",
                            alphas=alphas).fit(Ks.sum(0), Y)
    model_1 = MultipleKernelRidgeCV(
        kernels="precomputed", solver_params=dict(n_iter=n_iter,
                                                  alphas=alphas)).fit(Ks, Y)

    assert_array_almost_equal(model_0.predict(Ks.sum(0)), model_1.predict(Ks))


@pytest.mark.parametrize('solver', ['eigenvalues'])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_cv_Y_in_cpu(backend, solver):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    model_1 = KernelRidgeCV(solver=solver, Y_in_cpu=True)
    model_1.fit(Xs[0], Y)
    model_2 = KernelRidgeCV(solver=solver, Y_in_cpu=False)
    model_2.fit(Xs[0], Y)

    assert_array_almost_equal(model_1.dual_coef_, model_2.dual_coef_)
    assert_array_almost_equal(model_1.predict(Xs[0]), model_2.predict(Xs[0]))


@pytest.mark.parametrize('solver', ['random_search', 'hyper_gradient'])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_multiple_kernel_ridge_cv_Y_in_cpu(backend, solver):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    model_1 = MultipleKernelRidgeCV(kernels="precomputed", solver=solver,
                                    Y_in_cpu=True, random_state=0)
    model_1.fit(Ks, Y)
    model_2 = MultipleKernelRidgeCV(kernels="precomputed", solver=solver,
                                    Y_in_cpu=False, random_state=0)
    model_2.fit(Ks, Y)

    assert_array_almost_equal(model_1.dual_coef_, model_2.dual_coef_)
    assert_array_almost_equal(model_1.predict(Ks), model_2.predict(Ks))


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_weighted_kernel_ridge_cv_array_deltas(backend):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    # correct deltas array
    deltas = backend.zeros_like(Ks, shape=(len(Ks), ))
    model_1 = WeightedKernelRidge(kernels="precomputed", deltas=deltas,
                                  random_state=0)
    model_1.fit(Ks, Y)

    # wrong number of kernels
    deltas = backend.zeros_like(Ks, shape=(len(Ks) + 1, ))
    model_1 = WeightedKernelRidge(kernels="precomputed", deltas=deltas,
                                  random_state=0)
    with pytest.raises(ValueError, match="Inconsistent number of kernels"):
        model_1.fit(Ks, Y)

    # wrong number of targets
    deltas = backend.zeros_like(Ks, shape=(len(Ks), Y.shape[1] + 1))
    model_1 = WeightedKernelRidge(kernels="precomputed", deltas=deltas,
                                  random_state=0)
    with pytest.raises(ValueError, match="Inconsistent number of targets"):
        model_1.fit(Ks, Y)


###############################################################################
###############################################################################
###############################################################################
# scikit-learn.utils.estimator_checks


class KernelRidge_(KernelRidge):
    """Cast predictions to numpy arrays, to be used in scikit-learn tests.

    Used for testing only.
    """

    def predict(self, X):
        backend = get_backend()
        return backend.to_numpy(super().predict(X))

    def score(self, X, y):
        from himalaya.validation import check_array
        from himalaya.scoring import r2_score
        backend = get_backend()

        y_pred = super().predict(X)
        y_true = check_array(y, dtype=self.dtype_, ndim=self.dual_coef_.ndim)

        if y_true.ndim == 1:
            return backend.to_numpy(
                r2_score(y_true[:, None], y_pred[:, None])[0])
        else:
            return backend.to_numpy(r2_score(y_true, y_pred))


class KernelRidgeCV_(KernelRidgeCV):
    """Cast predictions to numpy arrays, to be used in scikit-learn tests.

    Used for testing only.
    """

    def __init__(self, alphas=(0.1, 1), kernel="linear", kernel_params=None,
                 solver="eigenvalues", solver_params=None, cv=2):
        super().__init__(alphas=alphas, kernel=kernel,
                         kernel_params=kernel_params, solver=solver,
                         solver_params=solver_params, cv=cv)

    def predict(self, X):
        backend = get_backend()
        return backend.to_numpy(super().predict(X))

    def score(self, X, y):
        from himalaya.validation import check_array
        from himalaya.scoring import r2_score
        backend = get_backend()

        y_pred = super().predict(X)
        y_true = check_array(y, dtype=self.dtype_, ndim=self.dual_coef_.ndim)

        if y_true.ndim == 1:
            return backend.to_numpy(
                r2_score(y_true[:, None], y_pred[:, None])[0])
        else:
            return backend.to_numpy(r2_score(y_true, y_pred))


class MultipleKernelRidgeCV_(MultipleKernelRidgeCV):
    """Cast predictions to numpy arrays, to be used in scikit-learn tests.

    Used for testing only.
    """

    def __init__(self, kernels=("linear", "polynomial"), kernels_params=None,
                 solver="hyper_gradient", solver_params=None, cv=2,
                 random_state=None):
        super().__init__(kernels=kernels, kernels_params=kernels_params,
                         solver=solver, solver_params=solver_params, cv=cv,
                         random_state=random_state)

    def predict(self, X, split=False):
        backend = get_backend()
        return backend.to_numpy(super().predict(X, split=split))

    def score(self, X, y, split=False):
        backend = get_backend()
        return backend.to_numpy(super().score(X, y, split=split))


class WeightedKernelRidge_(WeightedKernelRidge):
    """Cast predictions to numpy arrays, to be used in scikit-learn tests.

    Used for testing only.
    """

    def __init__(self, alpha=1., deltas="zeros",
                 kernels=("linear", "polynomial"), kernels_params=None,
                 solver="conjugate_gradient", solver_params=None,
                 random_state=None):
        super().__init__(alpha=alpha, deltas=deltas, kernels=kernels,
                         kernels_params=kernels_params, solver=solver,
                         solver_params=solver_params,
                         random_state=random_state)

    def predict(self, X, split=False):
        backend = get_backend()
        return backend.to_numpy(super().predict(X, split=split))

    def score(self, X, y, split=False):
        backend = get_backend()
        return backend.to_numpy(super().score(X, y, split=split))


@sklearn.utils.estimator_checks.parametrize_with_checks([
    KernelRidge_(),
    KernelRidgeCV_(),
    MultipleKernelRidgeCV_(),
    WeightedKernelRidge_(),
])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_check_estimator(estimator, check, backend):
    backend = set_backend(backend)
    check(estimator)
