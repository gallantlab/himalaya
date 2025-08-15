import warnings

import numpy as np
import pytest
import sklearn
import sklearn.kernel_ridge
import sklearn.utils.estimator_checks
from packaging import version

from himalaya.backend import ALL_BACKENDS, get_backend, set_backend
from himalaya.kernel_ridge import (
    KernelRidge,
    KernelRidgeCV,
    MultipleKernelRidgeCV,
    WeightedKernelRidge,
)
from himalaya.ridge import Ridge, RidgeCV
from himalaya.utils import assert_array_almost_equal


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

    # torch with cuda and torch_mps have more limited precision due to float32
    decimal = 3 if backend.name in ["torch_cuda", "torch_mps"] else 6

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
    except ImportError as error:
        pytest.skip("Scipy is not installed.")
        raise ImportError("Scipy not installed.") from error

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


@pytest.mark.parametrize('solver', ['eigenvalues', 'svd'])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_cv_precomputed(backend, solver):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    model_1 = KernelRidgeCV(kernel="linear", solver=solver)
    model_1.fit(Xs[0], Y)
    model_2 = KernelRidgeCV(kernel="precomputed", solver=solver)
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
    assert_array_almost_equal(score, score_split.sum(0), decimal=4)

    # single targets
    model = Estimator(kernels="precomputed", solver_params=solver_params)
    model.fit(Ks, Y[:, 0])
    score = model.score(Ks, Y[:, 0])
    score_split = model.score(Ks, Y[:, 0], split=True)
    assert score_split.shape == (len(Ks), )
    assert_array_almost_equal(score, score_split.sum(0), decimal=4)


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


@pytest.mark.parametrize('solver', ['eigenvalues', 'svd'])
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


@pytest.mark.parametrize(
    'Estimator',
    [
        Ridge,
        RidgeCV,
        KernelRidge,
        KernelRidgeCV,
        # MultipleKernelRidgeCV,  # too long
        WeightedKernelRidge,
    ])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_n_targets_batch(backend, Estimator):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    for solver in Estimator.ALL_SOLVERS.keys():
        model = Estimator(solver=solver, solver_params=dict(n_targets_batch=2))
        model.random_state = 0
        model.fit(Xs[0], Y)

        reference = Estimator(solver=solver)
        reference.random_state = 0
        reference.fit(Xs[0], Y)

        for attribute in ["coef_", "dual_coef_", "deltas_"]:
            if hasattr(model, attribute):
                assert_array_almost_equal(getattr(model, attribute),
                                          getattr(reference, attribute),
                                          decimal=5)

        assert_array_almost_equal(model.predict(Xs[0]),
                                  reference.predict(Xs[0]), decimal=4)
        assert_array_almost_equal(model.score(Xs[0], Y),
                                  reference.score(Xs[0], Y), decimal=4)


@pytest.mark.parametrize('Estimator', [KernelRidge, KernelRidgeCV])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_warning_kernel_ridge_ridge(backend, Estimator):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    with pytest.warns(UserWarning,
                      match="kernel ridge is slower than solving ridge"):
        Estimator(kernel="linear").fit(Xs[0][:, :10], Y)


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_auto_solver(backend):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)
    model_1 = KernelRidge(solver="auto").fit(Xs[0], Y)
    assert model_1.solver_ == "eigenvalues"

    alpha = backend.ones_like(Y, shape=(Y.shape[1], ))
    model_2 = KernelRidge(solver="auto", alpha=alpha).fit(Xs[0], Y)
    assert model_2.solver_ == "conjugate_gradient"

    assert_array_almost_equal(model_1.dual_coef_, model_2.dual_coef_,
                              decimal=5)

    # array with 1 element
    alpha = backend.ones_like(Y, shape=(1, ))
    model_1 = KernelRidge(solver="auto", alpha=alpha).fit(Xs[0], Y)
    assert model_1.solver_ == "eigenvalues"

    # first element of an array (!= float with torch)
    alpha = backend.ones_like(Y, shape=(1, ))[0]
    model_1 = KernelRidge(solver="auto", alpha=alpha).fit(Xs[0], Y)
    assert model_1.solver_ == "eigenvalues"


@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_multiple_kernel_ridge_intercept_return_alphas(backend, fit_intercept):
    backend = set_backend(backend)
    Xs, Ks, Y = _create_dataset(backend)

    solver_params = dict(n_iter=1, return_alphas=True)

    model_1 = MultipleKernelRidgeCV(kernels="precomputed",
                                    fit_intercept=fit_intercept,
                                    solver_params=solver_params)
    model_1.fit(Ks, Y)
    Y_pred_1 = model_1.predict(Ks)
    scores_1 = model_1.score(Ks, Y)

    if fit_intercept:
        Y += 10
    model_2 = MultipleKernelRidgeCV(kernels="precomputed",
                                    fit_intercept=fit_intercept,
                                    solver_params=solver_params).fit(Ks, Y)
    model_2.fit(Ks, Y)
    Y_pred_2 = model_2.predict(Ks)
    scores_2 = model_2.score(Ks, Y)
    if fit_intercept:
        Y_pred_2 -= 10

    assert_array_almost_equal(model_1.dual_coef_, model_2.dual_coef_)
    assert_array_almost_equal(Y_pred_1, Y_pred_2)
    assert_array_almost_equal(scores_1, scores_2)


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
        from himalaya.scoring import r2_score
        from himalaya.validation import check_array
        backend = get_backend()

        y_pred = super().predict(X)
        y_true = check_array(y, dtype=self.dtype_, ndim=self.dual_coef_.ndim)

        if y_true.ndim == 1:
            return backend.to_numpy(
                r2_score(y_true[:, None], y_pred[:, None])[0])
        else:
            return backend.to_numpy(r2_score(y_true, y_pred))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


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
        from himalaya.scoring import r2_score
        from himalaya.validation import check_array
        backend = get_backend()

        y_pred = super().predict(X)
        y_true = check_array(y, dtype=self.dtype_, ndim=self.dual_coef_.ndim)

        if y_true.ndim == 1:
            return backend.to_numpy(
                r2_score(y_true[:, None], y_pred[:, None])[0])
        else:
            return backend.to_numpy(r2_score(y_true, y_pred))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


def expected_failed_checks(estimator):
    """Return expected failed checks for sklearn 1.6+ compatibility.

    This replaces the deprecated _xfail_checks mechanism.
    """
    estimator_name = estimator.__class__.__name__

    # Only handle estimators that previously had _xfail_checks
    if estimator_name in ['KernelRidgeCV_', 'MultipleKernelRidgeCV_']:
        return {
            'check_sample_weight_equivalence_on_dense_data':
            'zero sample_weight is not equivalent to removing samples, '
            'because of the cross-validation splits.',
            'check_sample_weight_equivalence_on_sparse_data':
            'zero sample_weight is not equivalent to removing samples, '
            'because of the cross-validation splits.',
        }

    return {}


# Handle sklearn version compatibility for expected_failed_checks parameter
parametrize_kwargs = {}
if version.parse(sklearn.__version__) >= version.parse("1.6"):
    parametrize_kwargs['expected_failed_checks'] = expected_failed_checks

@sklearn.utils.estimator_checks.parametrize_with_checks([
    KernelRidge_(),
    KernelRidgeCV_(),
    MultipleKernelRidgeCV_(),
    WeightedKernelRidge_(),
], **parametrize_kwargs)
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_check_estimator(estimator, check, backend):
    backend = set_backend(backend)
    check(estimator)


###############################################################################
###############################################################################
###############################################################################
# Sample weight tests


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_sample_weight_basic(backend):
    """Test basic sample_weight functionality for KernelRidge."""
    backend = set_backend(backend)
    
    # Create small test dataset
    n_samples, n_features, n_targets = 10, 3, 2
    X = backend.asarray(backend.randn(n_samples, n_features), backend.float64)
    y = backend.asarray(backend.randn(n_samples, n_targets), backend.float64)
    
    # Create sample weights
    sample_weight = backend.asarray([1.0, 2.0, 0.5, 1.5, 0.1, 
                                   2.5, 1.0, 0.8, 1.2, 0.3], backend.float64)
    
    # Test that sample_weight affects predictions
    model_no_weight = KernelRidge(alpha=1.0, kernel="linear")
    model_no_weight.fit(X, y)
    pred_no_weight = model_no_weight.predict(X)
    
    model_with_weight = KernelRidge(alpha=1.0, kernel="linear")
    model_with_weight.fit(X, y, sample_weight=sample_weight)
    pred_with_weight = model_with_weight.predict(X)
    
    # Predictions should be different when sample weights are used
    assert not np.allclose(backend.to_numpy(pred_no_weight), 
                          backend.to_numpy(pred_with_weight))
    
    # Test with zero weights should raise no error
    zero_weight = backend.zeros_like(sample_weight)
    model_zero_weight = KernelRidge(alpha=1.0, kernel="linear")
    model_zero_weight.fit(X, y, sample_weight=zero_weight)
    
    # Test comparison with sklearn KernelRidge (they may differ significantly)
    import sklearn.kernel_ridge
    sk_model = sklearn.kernel_ridge.KernelRidge(alpha=1.0, kernel="linear")
    sk_model.fit(backend.to_numpy(X), backend.to_numpy(y), 
                 sample_weight=backend.to_numpy(sample_weight))
    sk_pred = sk_model.predict(backend.to_numpy(X))
    
    # Compare implementation - should now be very close
    max_diff = np.max(np.abs(backend.to_numpy(pred_with_weight) - sk_pred))
    print(f"Max difference between Himalaya and sklearn: {max_diff}")
    
    # After fixing dual coefficient scaling, should match sklearn closely
    assert_array_almost_equal(pred_with_weight, sk_pred, decimal=10)


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernel_ridge_sample_weight_equivalence(backend):
    """Test sample weight equivalence (similar to sklearn's check).
    
    This test examines whether using sample_weight=[2, 1, 3] is equivalent
    to duplicating samples [0, 0, 1, 2, 2, 2]. This helps understand why
    the sklearn check_sample_weight_equivalence tests fail.
    """
    backend = set_backend(backend)
    
    # Create small test dataset
    n_samples, n_features = 6, 3
    X = backend.asarray([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0], 
                        [7.0, 8.0, 9.0],
                        [2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0],
                        [8.0, 9.0, 1.0]], backend.float64)
    y = backend.asarray([[1.0], [2.0], [3.0], [1.5], [2.5], [3.5]], backend.float64)
    
    # Use simple integer weights for exact replication
    sample_weight = backend.asarray([2.0, 1.0, 3.0, 1.0, 2.0, 1.0], backend.float64)
    
    # Fit with sample weights
    model_weighted = KernelRidge(alpha=1.0, kernel="linear")
    model_weighted.fit(X, y, sample_weight=sample_weight)
    pred_weighted = model_weighted.predict(X)
    
    # Create equivalent dataset by repeating samples according to weights
    X_repeated = backend.concatenate([
        X[0:1], X[0:1],  # sample 0 repeated 2 times
        X[1:2],          # sample 1 repeated 1 time  
        X[2:3], X[2:3], X[2:3],  # sample 2 repeated 3 times
        X[3:4],          # sample 3 repeated 1 time
        X[4:5], X[4:5],  # sample 4 repeated 2 times
        X[5:6],          # sample 5 repeated 1 time
    ], axis=0)
    
    y_repeated = backend.concatenate([
        y[0:1], y[0:1],  # sample 0 repeated 2 times
        y[1:2],          # sample 1 repeated 1 time
        y[2:3], y[2:3], y[2:3],  # sample 2 repeated 3 times  
        y[3:4],          # sample 3 repeated 1 time
        y[4:5], y[4:5],  # sample 4 repeated 2 times
        y[5:6],          # sample 5 repeated 1 time
    ], axis=0)
    
    # Fit with repeated samples
    model_repeated = KernelRidge(alpha=1.0, kernel="linear")
    model_repeated.fit(X_repeated, y_repeated)
    pred_repeated = model_repeated.predict(X)
    
    # Compare predictions
    print(f"Backend: {backend.name}")
    print(f"Weighted predictions shape: {pred_weighted.shape}")
    print(f"Repeated predictions shape: {pred_repeated.shape}")
    max_diff = np.max(np.abs(backend.to_numpy(pred_weighted) - backend.to_numpy(pred_repeated)))
    print(f"Max absolute difference: {max_diff}")
    print(f"Are they close? {np.allclose(backend.to_numpy(pred_weighted), backend.to_numpy(pred_repeated), atol=1e-10)}")
    
    # After the fix, equivalence should now work
    assert_array_almost_equal(pred_weighted, pred_repeated, decimal=10)
    print("Sample weight equivalence PASSED - fix successful!")


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_weighted_kernel_ridge_sample_weight_basic(backend):
    """Test that WeightedKernelRidge sample weight fix works."""
    backend = set_backend(backend)
    
    # Create small test dataset
    n_samples, n_features, n_targets = 8, 3, 2
    X = backend.asarray(backend.randn(n_samples, n_features), backend.float64)
    y = backend.asarray(backend.randn(n_samples, n_targets), backend.float64)
    
    # Create sample weights
    sample_weight = backend.asarray([1.0, 2.0, 0.5, 1.5, 2.5, 1.0, 0.8, 1.2], 
                                   backend.float64)
    
    # Test that sample_weight affects predictions
    model_no_weight = WeightedKernelRidge(kernels=["linear"])
    model_no_weight.fit(X, y)
    pred_no_weight = model_no_weight.predict(X)
    
    model_with_weight = WeightedKernelRidge(kernels=["linear"])
    model_with_weight.fit(X, y, sample_weight=sample_weight)
    pred_with_weight = model_with_weight.predict(X)
    
    # Predictions should be different when sample weights are used
    assert not np.allclose(backend.to_numpy(pred_no_weight), 
                          backend.to_numpy(pred_with_weight))
    
    print("WeightedKernelRidge sample weight functionality verified!")
