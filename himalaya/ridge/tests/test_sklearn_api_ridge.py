from himalaya.scoring import r2_score
import pytest
import sklearn.kernel_ridge
import sklearn.utils.estimator_checks

from himalaya.backend import set_backend
from himalaya.backend import get_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.utils import assert_array_almost_equal

from himalaya.ridge import Ridge
from himalaya.ridge import RidgeCV
from himalaya.ridge import GroupRidgeCV
from himalaya.ridge import solve_group_ridge_random_search


def _create_dataset(backend):
    n_samples, n_features, n_targets = 30, 10, 3
    X = backend.asarray(backend.randn(n_samples, n_features), backend.float64)
    w = backend.asarray(backend.randn(n_features, n_targets), backend.float64)
    Y = X @ w
    Y += backend.asarray(backend.randn(*Y.shape), backend.float64)
    return X, Y


@pytest.mark.parametrize('multitarget', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_ridge_vs_scikit_learn(backend, multitarget, fit_intercept):
    backend = set_backend(backend)
    X, Y = _create_dataset(backend)

    if not multitarget:
        Y = Y[:, 0]
    if fit_intercept:
        Y += 10
        X += 10

    for alpha in backend.asarray_like(backend.logspace(0, 3, 7), Y):
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        model.fit(X, Y)

        reference = sklearn.linear_model.Ridge(alpha=backend.to_numpy(alpha),
                                               fit_intercept=fit_intercept)
        reference.fit(backend.to_numpy(X), backend.to_numpy(Y))

        if multitarget:
            assert model.coef_.shape == (X.shape[1], Y.shape[1])
        else:
            assert model.coef_.shape == (X.shape[1], )

        assert_array_almost_equal(model.coef_, reference.coef_.T)
        if fit_intercept:
            assert_array_almost_equal(model.intercept_, reference.intercept_,
                                      decimal=5)
        assert_array_almost_equal(model.predict(X),
                                  reference.predict(backend.to_numpy(X)),
                                  decimal=5)
        assert_array_almost_equal(
            model.score(X, Y).mean(),
            reference.score(backend.to_numpy(X), backend.to_numpy(Y)))


@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_ridge_cv_vs_scikit_learn(backend, fit_intercept):
    backend = set_backend(backend)
    X, Y = _create_dataset(backend)
    y = Y[:, 0]
    del Y
    if fit_intercept:
        y += 10
        X += 1

    alphas = backend.asarray_like(backend.logspace(-2, 3, 21), y)

    model = RidgeCV(alphas=alphas, cv=5, fit_intercept=fit_intercept,
                    solver_params=dict(score_func=r2_score))
    model.fit(X, y)

    reference = sklearn.linear_model.RidgeCV(alphas=backend.to_numpy(alphas),
                                             fit_intercept=fit_intercept, cv=5)
    reference.fit(backend.to_numpy(X), backend.to_numpy(y))
    assert model.coef_.shape == (X.shape[1], )

    assert_array_almost_equal(model.best_alphas_[0], reference.alpha_,
                              decimal=5)
    assert_array_almost_equal(model.coef_, reference.coef_.T)
    assert_array_almost_equal(model.predict(X),
                              reference.predict(backend.to_numpy(X)))
    assert_array_almost_equal(
        model.score(X, y).mean(),
        reference.score(backend.to_numpy(X), backend.to_numpy(y)))


@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_banded_ridge_cv_vs_ridge_cv(backend, fit_intercept):
    backend = set_backend(backend)
    X, Y = _create_dataset(backend)
    alphas = backend.asarray_like(backend.logspace(-2, 3, 21), Y)
    if fit_intercept:
        Y += 10

    ref = RidgeCV(alphas=alphas, cv=5, fit_intercept=fit_intercept)
    ref.fit(X, Y)

    model = GroupRidgeCV(groups=None, solver_params=dict(alphas=alphas), cv=5,
                         fit_intercept=fit_intercept)
    model.fit(X, Y)

    assert_array_almost_equal(model.best_alphas_, ref.best_alphas_)
    assert_array_almost_equal(model.coef_, ref.coef_)
    assert_array_almost_equal(model.predict(X), ref.predict(X))
    assert_array_almost_equal(model.score(X, Y), ref.score(X, Y))


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_group_ridge_split_score(backend):
    backend = set_backend(backend)
    X, Y = _create_dataset(backend)
    Y -= Y.mean(axis=0)
    groups = backend.randn(X.shape[1]) > 0
    alphas = backend.asarray_like(backend.logspace(-2, 3, 21), Y)

    model = GroupRidgeCV(groups=groups,
                         solver_params=dict(alphas=alphas, progress_bar=False))
    model.fit(X, Y)
    score = model.score(X, Y)
    score_split = model.score(X, Y, split=True)
    assert score_split.shape == (model.deltas_.shape[0], Y.shape[1])
    assert_array_almost_equal(score, score_split.sum(0), decimal=5)


###############################################################################
###############################################################################
###############################################################################
# scikit-learn.utils.estimator_checks


class Ridge_(Ridge):
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
        y_true = check_array(y, dtype=self.dtype_, ndim=self.coef_.ndim)

        if y_true.ndim == 1:
            return backend.to_numpy(
                r2_score(y_true[:, None], y_pred[:, None])[0])
        else:
            return backend.to_numpy(r2_score(y_true, y_pred))


class RidgeCV_(RidgeCV):
    """Cast predictions to numpy arrays, to be used in scikit-learn tests.

    Used for testing only.
    """

    def __init__(self, alphas=(0.1, 1), solver="svd", solver_params=None,
                 cv=2):
        super().__init__(alphas=alphas, solver=solver,
                         solver_params=solver_params, cv=cv)

    def predict(self, X):
        backend = get_backend()
        return backend.to_numpy(super().predict(X))

    def score(self, X, y):
        from himalaya.validation import check_array
        from himalaya.scoring import r2_score
        backend = get_backend()

        y_pred = super().predict(X)
        y_true = check_array(y, dtype=self.dtype_, ndim=self.coef_.ndim)

        if y_true.ndim == 1:
            return backend.to_numpy(
                r2_score(y_true[:, None], y_pred[:, None])[0])
        else:
            return backend.to_numpy(r2_score(y_true, y_pred))


# Dirty monkey-patch of n_iter,
# since check_estimator does not allow dict parameters
new_defaults = list(solve_group_ridge_random_search.__defaults__)
new_defaults[0] = 1
solve_group_ridge_random_search.__defaults__ = tuple(new_defaults)


class GroupRidgeCV_(GroupRidgeCV):
    """Cast predictions to numpy arrays, to be used in scikit-learn tests.

    Used for testing only.
    """

    def __init__(self, groups=None, solver="random_search", solver_params=None,
                 cv=2, random_state=0):
        super().__init__(groups=groups, solver=solver,
                         solver_params=solver_params, cv=cv,
                         random_state=random_state)

    def predict(self, X, split=False):
        backend = get_backend()
        return backend.to_numpy(super().predict(X, split=split))

    def score(self, X, y, split=False):
        backend = get_backend()
        return backend.to_numpy(super().score(X, y, split=split))


@sklearn.utils.estimator_checks.parametrize_with_checks([
    Ridge_(),
    RidgeCV_(),
    GroupRidgeCV_(),
])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_check_estimator(estimator, check, backend):
    backend = set_backend(backend)
    check(estimator)
