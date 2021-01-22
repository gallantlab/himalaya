from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import check_cv

from ._solvers import RIDGE_SOLVERS
from ._random_search import BANDED_RIDGE_SOLVERS
from ._random_search import solve_ridge_cv_svd

from ..validation import check_array
from ..validation import _get_string_dtype
from ..backend import get_backend
from ..scoring import r2_score


class _BaseRidge(ABC, MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Base class for ridge estimators"""

    @property
    @classmethod
    @abstractmethod
    def ALL_SOLVERS(cls):
        ...

    def _call_solver(self, **direct_params):
        if self.solver not in self.ALL_SOLVERS:
            raise ValueError("Unknown solver=%r." % self.solver)

        function = self.ALL_SOLVERS[self.solver]
        solver_params = self.solver_params or {}

        # check duplicated parameters
        intersection = set(direct_params.keys()).intersection(
            set(solver_params.keys()))
        if intersection:
            raise ValueError(
                'Parameters %s should not be given in solver_params, since '
                'they are either fixed or have a direct parameter in %s.' %
                (intersection, self.__class__.__name__))

        return function(**direct_params, **solver_params)

    def _more_tags(self):
        return {'requires_y': True}


class Ridge(_BaseRidge):
    """Ridge regression.

    Solve the ridge regression::

        b* = argmin_b ||X @ b - Y||^2 + alpha ||b||^2.

    Parameters
    ----------
    alpha : float, or array of shape (n_targets, )
        L2 regularization parameter.

    fit_intercept : boolean
        Whether to fit an intercept.
        If False, X and Y must be zero-mean over samples.

    solver : str
        Algorithm used during the fit, in {"svd"}.

    solver_params : dict or None
        Additional parameters for the solver.
        See more details in the docstring of the function:
        ``Ridge.ALL_SOLVERS[solver]``

    Attributes
    ----------
    coef_ : array of shape (n_features) or (n_features, n_targets)
        Ridge coefficients.

    intercept_ : float or array of shape (n_targets, )
        Intercept. Only present if fit_intercept is True.

    n_features_in_ : int
        Number of features used during the fit.

    dtype_ : str
        Dtype of input data.

    Examples
    --------
    >>> from himalaya.ridge import Ridge
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 10, 5, 3
    >>> X = np.random.randn(n_samples, n_features)
    >>> Y = np.random.randn(n_samples, n_targets)
    >>> model = Ridge()
    >>> model.fit(X, Y)
    Ridge()
    """
    ALL_SOLVERS = RIDGE_SOLVERS

    def __init__(self, alpha=1, fit_intercept=False, solver="svd",
                 solver_params=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.solver_params = solver_params

    def fit(self, X, y=None):
        """Fit the model.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data.

        y : array of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        # backend = get_backend()
        X = check_array(X, ndim=2)
        self.dtype_ = _get_string_dtype(X)
        y = check_array(y, dtype=self.dtype_, ndim=[1, 2])
        if X.shape[0] != y.shape[0]:
            raise ValueError("Inconsistent number of samples.")

        self.n_features_in_ = X.shape[1]

        ravel = False
        if y.ndim == 1:
            y = y[:, None]
            ravel = True

        # ------------------ call the solver
        tmp = self._call_solver(X=X, Y=y, alpha=self.alpha,
                                fit_intercept=self.fit_intercept)
        if self.fit_intercept:
            self.coef_, self.intercept_ = tmp
        else:
            self.coef_ = tmp

        if ravel:
            self.coef_ = self.coef_[:, 0]
            if self.fit_intercept:
                self.intercept_ = self.intercept_[0]

        return self

    def predict(self, X):
        """Predict using the model.

        Parameters
        ----------
        X : array of shape (n_samples_test, n_features)
            Testing features.

        Returns
        -------
        Y_hat : array of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        check_is_fitted(self)
        backend = get_backend()
        X = check_array(X, dtype=self.dtype_, ndim=2)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                'Different number of features in X than during fit.')

        Y_hat = backend.to_cpu(X) @ backend.to_cpu(self.coef_)
        if self.fit_intercept:
            Y_hat += backend.to_cpu(self.intercept_)
        return Y_hat

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array of shape (n_samples_test, n_features)
            Testing features.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for X.

        Returns
        -------
        score : array of shape (n_targets, )
            R^2 of self.predict(X) versus y.
        """
        y_pred = self.predict(X)
        y_true = check_array(y, dtype=self.dtype_, ndim=self.coef_.ndim)

        if y_true.ndim == 1:
            return r2_score(y_true[:, None], y_pred[:, None])[0]
        else:
            return r2_score(y_true, y_pred)


class RidgeCV(Ridge):
    """Ridge regression with efficient cross-validation over alpha.

    Solve the ridge regression::

        b* = argmin_b ||X @ b - Y||^2 + alpha ||b||^2,

    with a grid-search over cross-validation to find the best alpha.

    Parameters
    ----------
    alphas : array of shape (n_alphas, )
        List of L2 regularization parameter to try.

    fit_intercept : boolean
        Whether to fit an intercept.
        If False, X and Y must be zero-mean over samples.

    solver : str
        Algorithm used during the fit, "svd" only for now.

    solver_params : dict or None
        Additional parameters for the solver.
        See more details in the docstring of the function:
        ``RidgeCV.ALL_SOLVERS[solver]``

    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.

    Y_in_cpu : bool
        If True, keep the target values ``y`` in CPU memory (slower).

    Attributes
    ----------
    coef_ : array of shape (n_features) or (n_features, n_targets)
        Ridge coefficients.

    intercept_ : float or array of shape (n_targets, )
        Intercept. Only returned when fit_intercept is True.

    best_alphas_ : array of shape (n_targets, )
        Selected best hyperparameter alphas.

    cv_scores_ : array of shape (n_targets, )
        Cross-validation scores averaged over splits, for the best alpha.

    n_features_in_ : int
        Number of features used during the fit.

    Examples
    --------
    >>> from himalaya.ridge import RidgeCV
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 10, 5, 3
    >>> X = np.random.randn(n_samples, n_features)
    >>> Y = np.random.randn(n_samples, n_targets)
    >>> clf = RidgeCV()
    >>> clf.fit(X, Y)
    RidgeCV()
    """
    ALL_SOLVERS = dict(svd=solve_ridge_cv_svd)

    def __init__(self, alphas=[0.1, 1], fit_intercept=False, solver="svd",
                 solver_params=None, cv=5, Y_in_cpu=False):
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.solver_params = solver_params
        self.cv = cv
        self.Y_in_cpu = Y_in_cpu

    def fit(self, X, y=None):
        """Fit ridge regression model

        Parameters
        ----------
        X : array of shape (n_samples, n_features).
            Training data.

        y : array of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        # backend = get_backend()
        X = check_array(X, ndim=2)
        self.dtype_ = _get_string_dtype(X)
        device = "cpu" if self.Y_in_cpu else None
        y = check_array(y, dtype=self.dtype_, ndim=[1, 2], device=device)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Inconsistent number of samples.")

        alphas = check_array(self.alphas, dtype=self.dtype_, ndim=1)
        self.n_features_in_ = X.shape[1]

        ravel = False
        if y.ndim == 1:
            y = y[:, None]
            ravel = True

        cv = check_cv(self.cv)

        # ------------------ call the solver
        tmp = self._call_solver(X=X, Y=y, cv=cv, alphas=alphas,
                                fit_intercept=self.fit_intercept,
                                Y_in_cpu=self.Y_in_cpu)
        if self.fit_intercept:
            self.best_alphas_, self.coef_, self.cv_scores_ = tmp[:3]
            self.intercept_, = tmp[3:]
        else:
            self.best_alphas_, self.coef_, self.cv_scores_ = tmp

        self.cv_scores_ = self.cv_scores_[0]

        if ravel:
            self.coef_ = self.coef_[:, 0]
            if self.fit_intercept:
                self.intercept_ = self.intercept_[0]

        return self


###############################################################################
###############################################################################
###############################################################################
###############################################################################


class BandedRidgeCV(_BaseRidge):
    """Banded ridge regression with cross-validation.

    Solve the banded ridge regression::

        b* = argmin_b ||Z @ b - Y||^2 + ||b||^2

    where the feature space X_i is scaled by a group scaling ::

        Z_i = exp(deltas[i] / 2) X_i

    The solver optimizes the log group scalings ``deltas`` over
    cross-validation, using random search (``solver="random_search"``).

    Parameters
    ----------
    groups : array of shape (n_features, ), "input", or None
        Encoding of the group of each feature. If None, all features are
        gathered in one group, and the problem is equivalent to RidgeCV.
        If "input", the input features ``X`` should be a list of 2D arrays,
        corresponding to each group.

    solver : str
        Algorithm used during the fit, only "random_search" for now.

    solver_params : dict or None
        Additional parameters for the solver.
        See more details in the docstring of the function:
        ``BandedRidgeCV.ALL_SOLVERS[solver]``

    fit_intercept : boolean
        Whether to fit an intercept.
        If False, X and Y must be zero-mean over samples.

    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.

    random_state : int, or None
        Random generator seed. Use an int for deterministic search.

    Y_in_cpu : bool
        If True, keep the target values ``y`` in CPU memory (slower).

    Attributes
    ----------
    coef_ : array of shape (n_features) or (n_features, n_targets)
        Ridge coefficients.

    intercept_ : float or array of shape (n_targets, )
        Intercept. Only returned when fit_intercept is True.

    deltas_ : array of shape (n_groups, n_targets)
        Log of the group scalings.

    cv_scores_ : array of shape (n_iter, n_targets)
        Cross-validation scores, averaged over splits.

    n_features_in_ : int
        Number of features used during the fit.

    dtype_ : str
        Dtype of input data.

    best_alphas_ : array of shape (n_targets, )
        Equal to ``1. / exp(self.deltas_).sum(0)``. For the "random_search"
        solver, it corresponds to the best hyperparameter alphas, assuming that
        each squared group scaling vector sums to one (in particular, it is the
        case when ``solver_params['n_iter']`` is an integer).

    Examples
    --------
    >>> from himalaya.ridge import BandedRidgeCV
    >>> from himalaya.ridge import ColumnTransformerNoStack
    >>> from sklearn.pipeline import make_pipeline

    >>> # create a dataset
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 10, 5, 3
    >>> X = np.random.randn(n_samples, n_features)
    >>> Y = np.random.randn(n_samples, n_targets)

    >>> # Separate the first three columns and the last two
    >>> # columns, creating two groups of shape (n_samples, n_feature_i).
    >>> from sklearn.preprocessing import StandardScaler
    >>> ct = ColumnTransformerNoStack(
    ...     [("group_1", StandardScaler(), [0, 1, 2]),
    ...      ("group_2", StandardScaler(), slice(3, 5))])

    >>> # A model with automatic groups, as output by ColumnTransformerNoStack
    >>> model = BandedRidgeCV(groups="input")
    >>> pipe = make_pipeline(ct, model)
    >>> _ = pipe.fit(X, Y)
    """
    ALL_SOLVERS = BANDED_RIDGE_SOLVERS

    def __init__(self, groups=None, solver="random_search", solver_params=None,
                 fit_intercept=False, cv=5, random_state=None, Y_in_cpu=False):

        self.groups = groups
        self.solver = solver
        self.solver_params = solver_params
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.random_state = random_state
        self.Y_in_cpu = Y_in_cpu

    def fit(self, X, y=None):
        """Fit the model.

        Parameters
        ----------
        X : array of shape (n_samples, n_features), or list of length \
                (n_groups) with arrays of shape (n_samples, n_features)
            Training data.
            Must be a 2D array if ``groups`` is given.
            Must be a list of 2D arrays if ``groups="input"``.

        y : array of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        backend = get_backend()

        Xs = self._split_groups(X, check=True)
        del X

        self.n_features_in_ = sum(Xi.shape[1] for Xi in Xs)

        self.dtype_ = _get_string_dtype(Xs[0])
        device = "cpu" if self.Y_in_cpu else None
        y = check_array(y, dtype=self.dtype_, ndim=[1, 2], device=device)

        if any([Xi.shape[0] != y.shape[0] for Xi in Xs]):
            raise ValueError("Inconsistent number of samples.")

        ravel = False
        if y.ndim == 1:
            y = y[:, None]
            ravel = True

        cv = check_cv(self.cv)

        # ------------------ call the solver
        tmp = self._call_solver(Xs=Xs, Y=y, cv=cv, return_weights=True,
                                random_state=self.random_state,
                                fit_intercept=self.fit_intercept,
                                Y_in_cpu=self.Y_in_cpu)
        if self.fit_intercept:
            self.deltas_, self.coef_, self.cv_scores_ = tmp[:3]
            self.intercept_, = tmp[3:]
        else:
            self.deltas_, self.coef_, self.cv_scores_ = tmp

        if self.solver == "random_search":
            self.best_alphas_ = 1. / backend.exp(self.deltas_).sum(0)
        else:
            self.best_alphas_ = None

        if ravel:
            self.coef_ = self.coef_[:, 0]
            self.deltas_ = self.deltas_[:, 0]
            if self.fit_intercept:
                self.intercept_ = self.intercept_[0]

        return self

    def predict(self, X, split=False):
        """Predict using the model.

        Parameters
        ----------
        X : array of shape (n_samples_test, n_features), or list of length \
                (n_groups) with arrays of shape (n_samples_test, n_features)
            Training data.
            Must be a 2D array if ``groups`` is given.
            Must be a list of 2D arrays if ``groups="input"``.

        split : bool
            If True, the prediction is split on each feature space, and this
            method returns an array with one extra dimension (first dimension).
            The sum over this extra dimension corresponds to split=False.

        Returns
        -------
        Y_hat : array of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
            If parameter split is True, the array is of shape
            (n_groups, n_samples,) or (n_groups, n_samples, n_targets).
        """
        backend = get_backend()
        check_is_fitted(self)

        Xs = self._split_groups(X, dtype=self.dtype_, check=True)

        n_features = sum(Xi.shape[1] for Xi in Xs)
        if n_features != self.n_features_in_:
            raise ValueError(
                'Different number of features in X than during fit.')

        X = backend.to_cpu(backend.concatenate(Xs, 1))
        del Xs

        Y_hat = X @ backend.to_cpu(self.coef_)
        if self.fit_intercept:
            Y_hat += backend.to_cpu(self.intercept_)
        return Y_hat

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array of shape (n_samples_test, n_features), or list of length \
                (n_groups) with arrays of shape (n_samples_test, n_features)
            Training data.
            Must be a 2D array if ``groups`` is given.
            Must be a list of 2D arrays if ``groups="input"``.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for X.

        Returns
        -------
        score : array of shape (n_targets, )
            R^2 of self.predict(X) versus y.
        """
        y_pred = self.predict(X)
        y_true = check_array(y, dtype=self.dtype_, ndim=self.coef_.ndim)

        if y_true.ndim == 1:
            return r2_score(y_true[:, None], y_pred[:, None])[0]
        else:
            return r2_score(y_true, y_pred)

    def _split_groups(self, X, check=True, **check_kwargs):
        backend = get_backend()

        # groups defined in X
        if self.groups == "input":
            if check:
                X = [check_array(Xi, ndim=2, **check_kwargs) for Xi in X]
            return X

        # groups defined in self.groups
        if check:
            X = check_array(X, ndim=2, **check_kwargs)
        if self.groups is None:
            groups = backend.zeros((X.shape[1]))
        else:
            groups = self.groups
        groups = backend.asarray(groups)[:]
        Xs = [X[:, groups == u] for u in backend.unique(groups) if u >= 0]
        return Xs
