from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import check_cv

from ._group_lasso import solve_sparse_group_lasso_cv

from ..validation import check_array
from ..validation import _get_string_dtype
from ..backend import get_backend
from ..scoring import r2_score


class SparseGroupLassoCV(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Sparse group Lasso

    Solved with hyperparameter grid-search over cross-validation.

    Parameters
    ----------
    groups : array of shape (n_features, ) or None
        Encoding of the group of each feature. If None, all features are
        gathered in one group, and the problem is equivalent to the Lasso.

    l21_regs : array of shape (n_l21_regs, )
        All the Group Lasso regularization parameter tested.

    l1_regs : array of shape (n_l1_regs, )
        All the Lasso regularization parameter tested.

    solver : str
        Algorithm used during the fit, "proximal_gradient" only for now.

    solver_params : dict or None
        Additional parameters for the solver.
        See more details in the docstring of the function:
        ``SparseGroupLassoCV.ALL_SOLVERS[solver]``

    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.

    Attributes
    ----------
    coef_ : array of shape (n_samples) or (n_samples, n_targets)
        Coefficient of the linear model. Always on CPU.

    best_l21_reg_ : array of shape (n_targets, )
        Best hyperparameter per target.

    best_l1_reg_ : array of shape (n_targets, )
        Best hyperparameter per target.

    cv_scores_ : array of shape (n_l21_regs * n_l1_regs, n_targets)
        Cross-validation scores of all tested hyperparameters.

    n_features_in_ : int
        Number of features used during the fit.

    Examples
    --------
    >>> from himalaya.lasso import SparseGroupLassoCV
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 10, 5, 3
    >>> X = np.random.randn(n_samples, n_features)
    >>> Y = np.random.randn(n_samples, n_targets)
    >>> clf = SparseGroupLassoCV()
    >>> clf.fit(X, Y)
    SparseGroupLassoCV()
    """
    ALL_SOLVERS = dict(proximal_gradient=solve_sparse_group_lasso_cv)

    def __init__(self, groups=None, l1_regs=[0], l21_regs=[0],
                 solver="proximal_gradient", solver_params=None, cv=5):
        self.groups = groups
        self.l1_regs = l1_regs
        self.l21_regs = l21_regs
        self.solver = solver
        self.solver_params = solver_params
        self.cv = cv

    def fit(self, X, y):
        """Fit the model

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
        X = check_array(X, accept_sparse=False, ndim=2)
        self.dtype_ = _get_string_dtype(X)
        y = check_array(y, dtype=self.dtype_, ndim=[1, 2])
        if X.shape[0] != y.shape[0]:
            raise ValueError("Inconsistent number of samples.")

        self.n_features_in_ = X.shape[1]
        cv = check_cv(self.cv)
        ravel = False
        if y.ndim == 1:
            y = y[:, None]
            ravel = True

        results = self._call_solver(X=X, Y=y, groups=self.groups, cv=cv,
                                    l21_regs=self.l21_regs,
                                    l1_regs=self.l1_regs)
        self.coef_, self.best_l21_reg_, self.best_l1_reg_ = results[:3]
        self.cv_scores_ = results[3]

        if ravel:
            self.coef_ = self.coef_[:, 0]

        return self

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

    def predict(self, X):
        """Predict using the model.

        Parameters
        ----------
        X : array of shape (n_samples_test, n_features)
            Samples.

        Returns
        -------
        Y_hat : array of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        backend = get_backend()
        check_is_fitted(self)
        X = check_array(X, dtype=self.dtype_, accept_sparse=False, ndim=2)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                'Different number of features in X than during fit.')
        Y_hat = backend.to_numpy(X) @ backend.to_numpy(self.coef_)
        return backend.asarray_like(Y_hat, ref=X)

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array of shape (n_samples_test, n_features)
            Samples.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for X.

        Returns
        -------
        score : array of shape (n_targets, )
            R^2 of self.predict(X) versus y.
        """
        y_pred = self.predict(X, y)
        y_true = check_array(y, dtype=self.dtype_, ndim=self.coef_.ndim)

        if y_true.ndim == 1:
            return r2_score(y_true[:, None], y_pred[:, None])[0]
        else:
            return r2_score(y_true, y_pred)

    def _more_tags(self):
        return {'requires_y': True}
