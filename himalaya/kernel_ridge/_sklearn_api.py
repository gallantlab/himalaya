from abc import ABC, abstractmethod
import warnings

from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.utils.validation import check_is_fitted

from ._solvers import KERNEL_RIDGE_SOLVERS
from ._solvers import WEIGHTED_KERNEL_RIDGE_SOLVERS
from ._hyper_gradient import MULTIPLE_KERNEL_RIDGE_SOLVERS
from ._random_search import solve_kernel_ridge_cv_eigenvalues
from ._kernels import pairwise_kernels
from ._kernels import PAIRWISE_KERNEL_FUNCTIONS
from ._predictions import predict_weighted_kernel_ridge
from ._predictions import predict_and_score_weighted_kernel_ridge
from ._predictions import primal_weights_weighted_kernel_ridge

from ..validation import check_array
from ..validation import check_cv
from ..validation import issparse
from ..validation import _get_string_dtype
from ..backend import get_backend
from ..backend import force_cpu_backend
from ..scoring import r2_score
from ..scoring import r2_score_split


class _BaseKernelRidge(ABC, MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Base class for kernel ridge estimators"""

    ALL_KERNELS = PAIRWISE_KERNEL_FUNCTIONS

    @property
    @classmethod
    @abstractmethod
    def ALL_SOLVERS(cls):
        ...

    def _call_solver(self, **direct_params):
        """Helper function common to all classes, merging solver parameters."""

        # use self.solver_ if it exists, otherwise use self.solver
        solver = getattr(self, "solver_", self.solver)

        if solver not in self.ALL_SOLVERS:
            raise ValueError("Unknown solver=%r." % solver)

        function = self.ALL_SOLVERS[solver]
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


class KernelRidge(_BaseKernelRidge):
    """Kernel ridge regression.

    Solve the kernel ridge regression::

        w* = argmin_w ||K @ w - Y||^2 + alpha (w.T @ K @ w)

    where K is a kernel computed on the input X.
    The default kernel is linear (K = X @ X.T).

    Parameters
    ----------
    alpha : float, or array of shape (n_targets, )
        L2 regularization parameter.

    kernel : str or callable, default="linear"
        Kernel mapping. Available kernels are: 'linear',
        'polynomial, 'poly', 'rbf', 'sigmoid', 'cosine', or 'precomputed'.
        Set to 'precomputed' in order to pass a precomputed kernel matrix to
        the estimator methods instead of samples.
        A callable should accept two arguments and the keyword arguments passed
        to this object as kernel_params, and should return a floating point
        number.

    kernel_params : dict or None
        Additional parameters for the kernel function.
        See more details in the docstring of the function:
        ``KernelRidge.ALL_KERNELS[kernel]``

    solver : str
        Algorithm used during the fit, "eigenvalues", "conjugate_gradient",
        "gradient_descent", or "auto".
        If "auto", use "eigenvalues" if ``alpha`` is a float, and
        "conjugate_gradient" is ``alpha`` is an array.

    solver_params : dict or None
        Additional parameters for the solver.
        See more details in the docstring of the function:
        ``KernelRidge.ALL_SOLVERS[solver]``

    fit_intercept : boolean
        Whether to fit an intercept.
        If False, X and Y must be zero-mean over samples.

    force_cpu : bool
        If True, computations will be performed on CPU, ignoring the
        current backend. If False, use the current backend.

    warn : bool
        If True, warn if the number of samples is larger than the number of
        features, and if the kernel is linear.

    Attributes
    ----------
    dual_coef_ : array of shape (n_samples) or (n_samples, n_targets)
        Representation of weight vectors in kernel space.

    intercept_ : float or array of shape (n_targets, )
        Intercept. Only present if fit_intercept is True.

    X_fit_ : array of shape (n_samples, n_features)
        Training data. If kernel == "precomputed" this is None.

    n_features_in_ : int
        Number of features (or number of samples if kernel == "precomputed")
        used during the fit.

    dtype_ : str
        Dtype of input data.

    Examples
    --------
    >>> from himalaya.kernel_ridge import KernelRidge
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 10, 5, 3
    >>> X = np.random.randn(n_samples, n_features)
    >>> Y = np.random.randn(n_samples, n_targets)
    >>> model = KernelRidge()
    >>> model.fit(X, Y)
    KernelRidge()
    """
    ALL_SOLVERS = KERNEL_RIDGE_SOLVERS

    def __init__(self, alpha=1, kernel="linear", kernel_params=None,
                 solver="auto", solver_params=None, fit_intercept=False,
                 force_cpu=False, warn=True):
        self.alpha = alpha
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.solver = solver
        self.solver_params = solver_params
        self.fit_intercept = fit_intercept
        self.force_cpu = force_cpu
        self.warn = warn

    @force_cpu_backend
    def fit(self, X, y=None, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data. If kernel == "precomputed" this is instead
            a precomputed kernel array of shape (n_samples, n_samples).

        y : array of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : None, or array of shape (n_samples, )
            Individual weights for each sample, ignored if None is passed.

        Returns
        -------
        self : returns an instance of self.
        """
        backend = get_backend()
        accept_sparse = False if self.kernel == "precomputed" else ("csr",
                                                                    "csc")
        X = check_array(X, accept_sparse=accept_sparse, ndim=2)
        self.dtype_ = _get_string_dtype(X)
        y = check_array(y, dtype=self.dtype_, ndim=[1, 2])
        if X.shape[0] != y.shape[0]:
            raise ValueError("Inconsistent number of samples.")

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, dtype=self.dtype_,
                                        ndim=1)
            if sample_weight.shape[0] != y.shape[0]:
                raise ValueError("Inconsistent number of samples.")

        n_samples, n_features = X.shape
        if n_samples > n_features and self.kernel == "linear" and self.warn:
            warnings.warn(
                "Solving linear kernel ridge is slower than solving ridge when"
                f" n_samples > n_features (here {n_samples} > {n_features}). "
                "Using himalaya.ridge.Ridge would be faster. Use warn=False to"
                " silence this warning.", UserWarning)

        K = self._get_kernel(X)

        self.X_fit_ = _to_cpu(X) if self.kernel != "precomputed" else None
        self.n_features_in_ = X.shape[1]
        del X

        ravel = False
        if y.ndim == 1:
            y = y[:, None]
            ravel = True

        if sample_weight is not None:
            # We need to support sample_weight directly because K might be a
            # precomputed kernel.
            sw = backend.sqrt(sample_weight)[:, None]
            y = y * sw
            K *= sw @ sw.T

        # select solver based on the presence of multiple alphas
        if self.solver == "auto":
            if backend.atleast_1d(self.alpha).shape[0] == 1:
                self.solver_ = "eigenvalues"
            else:
                self.solver_ = "conjugate_gradient"
        else:
            self.solver_ = self.solver

        # ------------------ call the solver
        tmp = self._call_solver(K=K, Y=y, alpha=self.alpha,
                                fit_intercept=self.fit_intercept)
        if self.fit_intercept:
            self.dual_coef_, self.intercept_ = tmp
        else:
            self.dual_coef_ = tmp

        if ravel:
            self.dual_coef_ = self.dual_coef_[:, 0]
            if self.fit_intercept:
                self.intercept_ = self.intercept_[0]

        return self

    @force_cpu_backend
    def predict(self, X):
        """Predict using the model.

        Parameters
        ----------
        X : array of shape (n_samples_test, n_features)
            Samples. If kernel == "precomputed" this is instead a precomputed
            kernel array of shape (n_samples_test, n_samples_train).

        Returns
        -------
        Y_hat : array of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        check_is_fitted(self)
        backend = get_backend()
        accept_sparse = False if self.kernel == "precomputed" else ("csr",
                                                                    "csc")
        X = check_array(X, dtype=self.dtype_, accept_sparse=accept_sparse,
                        ndim=2)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                'Different number of features in X than during fit.')

        K = self._get_kernel(X, self.X_fit_)
        del X

        Y_hat = backend.to_cpu(K) @ backend.to_cpu(self.dual_coef_)
        if self.fit_intercept:
            Y_hat += backend.to_cpu(self.intercept_)
        return Y_hat

    @force_cpu_backend
    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array of shape (n_samples_test, n_features)
            Samples. If kernel == "precomputed" this is instead a precomputed
            kernel array of shape (n_samples_test, n_samples_train).

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for X.

        Returns
        -------
        score : array of shape (n_targets, )
            R^2 of self.predict(X) versus y.
        """
        y_pred = self.predict(X)
        y_true = check_array(y, dtype=self.dtype_, ndim=self.dual_coef_.ndim)

        if y_true.ndim == 1:
            return r2_score(y_true[:, None], y_pred[:, None])[0]
        else:
            return r2_score(y_true, y_pred)

    def _get_kernel(self, X, Y=None):
        backend = get_backend()
        kernel_params = self.kernel_params or {}
        if Y is not None and not issparse(X):
            Y = backend.asarray_like(Y, ref=X)
        kernel = pairwise_kernels(X, Y, metric=self.kernel, **kernel_params)
        return backend.asarray(kernel)

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def get_primal_coef(self, X_fit=None):
        """Returns the primal coefficients, assuming the kernel is linear.

        When the kernel is linear, kernel ridge regression is equivalent to
        ridge rergession, and the ridge regression (primal) coefficients can be
        computed from the kernel ridge regression (dual) coefficients, using
        the training features X_fit.

        Parameters
        ----------
        X_fit : array of shape (n_samples_fit, n_features) or None
            Training features. Used only if self.kernel == "precomputed".
            If you used a Kernelizer, you can use the method `get_X_fit`.

        Returns
        -------
        primal_coef : array of shape (n_features, n_targets)
            Coefficient of the equivalent ridge regression. The coefficients
            are computed on CPU memory, since they can be large.
        """
        check_is_fitted(self)
        backend = get_backend()

        if self.kernel == "linear":
            X_fit_T = self.X_fit_.T
        elif self.kernel == "precomputed":
            if X_fit is None:
                raise ValueError(
                    "get_primal_coef requires the training features `X_fit`. "
                    "If you used a Kernelizer, you can use the method "
                    "`get_X_fit`.")
            X_fit_T = X_fit.T
        else:
            raise ValueError("The primal coefficients can only be computed "
                             "when using a linear kernel.")

        X_fit_T = backend.to_cpu(X_fit_T)
        dual_coef = backend.to_cpu(self.dual_coef_)
        return X_fit_T @ dual_coef


class KernelRidgeCV(KernelRidge):
    """Kernel ridge regression with efficient cross-validation over alpha.

    Parameters
    ----------
    alphas : array of shape (n_alphas, )
        List of L2 regularization parameter to try.

    kernel : str or callable, default="linear"
        Kernel mapping. Available kernels are: 'linear',
        'polynomial, 'poly', 'rbf', 'sigmoid', 'cosine', or 'precomputed'.
        Set to 'precomputed' in order to pass a precomputed kernel matrix to
        the estimator methods instead of samples.
        A callable should accept two arguments and the keyword arguments passed
        to this object as kernel_params, and should return a floating point
        number.

    kernel_params : dict or None
        Additional parameters for the kernel function.
        See more details in the docstring of the function:
        ``KernelRidgeCV.ALL_KERNELS[kernel]``

    solver : str
        Algorithm used during the fit, "eigenvalues" only for now.

    solver_params : dict or None
        Additional parameters for the solver.
        See more details in the docstring of the function:
        ``KernelRidgeCV.ALL_SOLVERS[solver]``

    fit_intercept : boolean
        Whether to fit an intercept.
        If False, X and Y must be zero-mean over samples.

    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.

    Y_in_cpu : bool
        If True, keep the target values ``y`` in CPU memory (slower).

    force_cpu : bool
        If True, computations will be performed on CPU, ignoring the
        current backend. If False, use the current backend.

    warn : bool
        If True, warn if the number of samples is larger than the number of
        features, and if the kernel is linear.

    Attributes
    ----------
    dual_coef_ : array of shape (n_samples) or (n_samples, n_targets)
        Representation of weight vectors in kernel space.

    best_alphas_ : array of shape (n_targets, )
        Selected best hyperparameter alphas.

    cv_scores_ : array of shape (n_targets, )
        Cross-validation scores averaged over splits, for the best alpha.
        By default, the scores are computed with l2_neg_loss (in ]-inf, 0]).
        The scoring function can be changed with solver_params["score_func"].

    X_fit_ : array of shape (n_samples, n_features)
        Training data. If kernel == "precomputed" this is None.

    n_features_in_ : int
        Number of features (or number of samples if kernel == "precomputed")
        used during the fit.

    Examples
    --------
    >>> from himalaya.ridge import KernelRidgeCV
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 10, 5, 3
    >>> X = np.random.randn(n_samples, n_features)
    >>> Y = np.random.randn(n_samples, n_targets)
    >>> clf = KernelRidgeCV()
    >>> clf.fit(X, Y)
    KernelRidgeCV()
    """
    ALL_SOLVERS = dict(eigenvalues=solve_kernel_ridge_cv_eigenvalues)

    def __init__(self, alphas=[0.1, 1], kernel="linear", kernel_params=None,
                 solver="eigenvalues", solver_params=None, fit_intercept=False,
                 cv=5, Y_in_cpu=False, force_cpu=False, warn=True):
        self.alphas = alphas
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.solver = solver
        self.solver_params = solver_params
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.Y_in_cpu = Y_in_cpu
        self.force_cpu = force_cpu
        self.warn = warn

    @force_cpu_backend
    def fit(self, X, y=None, sample_weight=None):
        """Fit kernel ridge regression model

        Parameters
        ----------
        X : array of shape (n_samples, n_features).
            Training data. If kernel == "precomputed" this is instead
            a precomputed kernel array of shape (n_samples, n_samples).

        y : array of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : None, or array of shape (n_samples, )
            Individual weights for each sample, ignored if None is passed.

        Returns
        -------
        self : returns an instance of self.
        """
        backend = get_backend()
        X = check_array(X, accept_sparse=("csr", "csc"), ndim=2)
        self.dtype_ = _get_string_dtype(X)
        device = "cpu" if self.Y_in_cpu else None
        y = check_array(y, dtype=self.dtype_, ndim=[1, 2], device=device)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Inconsistent number of samples.")

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, dtype=self.dtype_,
                                        ndim=1)
            if sample_weight.shape[0] != y.shape[0]:
                raise ValueError("Inconsistent number of samples.")

        alphas = check_array(self.alphas, dtype=self.dtype_, ndim=1)

        n_samples, n_features = X.shape
        if n_samples > n_features and self.kernel == "linear" and self.warn:
            warnings.warn(
                "Solving linear kernel ridge is slower than solving ridge when"
                f" n_samples > n_features (here {n_samples} > {n_features}). "
                "Using himalaya.ridge.RidgeCV would be faster. "
                "Use warn=False to silence this warning.", UserWarning)

        K = self._get_kernel(X)

        self.X_fit_ = _to_cpu(X) if self.kernel != "precomputed" else None
        self.n_features_in_ = X.shape[1]
        del X

        ravel = False
        if y.ndim == 1:
            y = y[:, None]
            ravel = True

        if sample_weight is not None:
            # We need to support sample_weight directly because K might be a
            # pre-computed kernel.
            sw = backend.sqrt(sample_weight)[:, None]
            y = y * backend.to_cpu(sw) if self.Y_in_cpu else y * sw
            K *= sw @ sw.T

        cv = check_cv(self.cv, y)

        # ------------------ call the solver
        tmp = self._call_solver(K=K, Y=y, cv=cv, alphas=alphas,
                                Y_in_cpu=self.Y_in_cpu,
                                fit_intercept=self.fit_intercept)
        if self.fit_intercept:
            self.best_alphas_, self.dual_coef_, self.cv_scores_ = tmp[:3]
            self.intercept_, = tmp[3:]
        else:
            self.best_alphas_, self.dual_coef_, self.cv_scores_ = tmp
        self.cv_scores_ = self.cv_scores_[0]

        if ravel:
            self.dual_coef_ = self.dual_coef_[:, 0]
            if self.fit_intercept:
                self.intercept_ = self.intercept_[0]

        return self

    def _more_tags(self):
        return {
            '_xfail_checks': {
                'check_sample_weights_invariance':
                'zero sample_weight is not equivalent to removing samples, '
                'because of the cross-validation splits.',
            }
        }


###############################################################################
###############################################################################
###############################################################################
###############################################################################


class _BaseWeightedKernelRidge(_BaseKernelRidge):
    """Private class for shared implementations.
    """

    @force_cpu_backend
    def predict(self, X, split=False):
        """Predict using the model.

        Parameters
        ----------
        X : array of shape (n_samples_test, n_features)
            Samples. If kernels == "precomputed" this is instead a precomputed
            kernel array of shape (n_kernels, n_samples_test, n_samples_train).

        split : bool
            If True, the prediction is split on each kernel, and this method
            returns an array with one extra dimension (first dimension).
            The sum over this extra dimension corresponds to split=False.

        Returns
        -------
        Y_hat : array of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
            If parameter split is True, the array is of shape
            (n_kernels, n_samples,) or (n_kernels, n_samples, n_targets).
        """
        check_is_fitted(self)

        ndim = 3 if self.kernels == "precomputed" else 2
        accept_sparse = False if self.kernels == "precomputed" else ("csr",
                                                                     "csc")
        X = check_array(X, dtype=self.dtype_, accept_sparse=accept_sparse,
                        ndim=ndim)
        if X.shape[-1] != self.n_features_in_:
            raise ValueError(
                'Different number of features in X than during fit.')

        Ks = self._get_kernels(X, self.X_fit_)
        del X

        if (self.solver_params is not None
                and "n_targets_batch" in self.solver_params):
            n_targets_batch = self.solver_params["n_targets_batch"]
        else:
            n_targets_batch = None

        if self.dual_coef_.ndim == 1:
            Y_hat = predict_weighted_kernel_ridge(
                Ks=Ks, dual_weights=self.dual_coef_[:, None],
                deltas=self.deltas_[:, None], split=split,
                n_targets_batch=n_targets_batch)[..., 0]
        else:
            Y_hat = predict_weighted_kernel_ridge(
                Ks=Ks, dual_weights=self.dual_coef_, deltas=self.deltas_,
                split=split, n_targets_batch=n_targets_batch)
        return Y_hat

    @force_cpu_backend
    def score(self, X, y, split=False):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array of shape (n_samples_test, n_features)
            Samples. If kernels == "precomputed" this is instead a precomputed
            kernel array of shape (n_kernels, n_samples_test, n_samples_train).

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for X.

        split : bool
            If True, the prediction is split on each kernel, and the R2 score
            is decomposed over sub-predictions, adding an extra dimension
            in the first axis. The sum over this extra dimension corresponds to
            split=False.

        Returns
        -------
        score : array of shape (n_targets, ) or (n_kernels, n_targets)
            R^2 of self.predict(X) versus y.
            If parameter split is True, the array is of shape
            (n_kernels, n_targets).
        """
        check_is_fitted(self)

        ndim = 3 if self.kernels == "precomputed" else 2
        accept_sparse = False if self.kernels == "precomputed" else ("csr",
                                                                     "csc")
        X = check_array(X, dtype=self.dtype_, accept_sparse=accept_sparse,
                        ndim=ndim)
        y = check_array(y, dtype=self.dtype_, ndim=self.dual_coef_.ndim)
        if X.shape[-1] != self.n_features_in_:
            raise ValueError(
                'Different number of features in X than during fit.')

        Ks = self._get_kernels(X, self.X_fit_)
        del X

        if (self.solver_params is not None
                and "n_targets_batch" in self.solver_params):
            n_targets_batch = self.solver_params["n_targets_batch"]
        else:
            n_targets_batch = None

        score_func = r2_score_split if split else r2_score

        if self.dual_coef_.ndim == 1:
            score = predict_and_score_weighted_kernel_ridge(
                Ks=Ks, dual_weights=self.dual_coef_[:, None],
                deltas=self.deltas_[:, None], Y=y[:, None], split=split,
                score_func=score_func, n_targets_batch=n_targets_batch)[..., 0]
        else:
            score = predict_and_score_weighted_kernel_ridge(
                Ks=Ks, dual_weights=self.dual_coef_, deltas=self.deltas_, Y=y,
                split=split, score_func=score_func,
                n_targets_batch=n_targets_batch)
        return score

    def _get_kernels(self, X, Y=None):
        backend = get_backend()
        if isinstance(self.kernels, str) and self.kernels == "precomputed":
            kernels = backend.asarray(X)

        elif not isinstance(self.kernels, (list, tuple)):
            raise ValueError("Parameter 'kernels' has to be a list or a tuple "
                             "of kernel parameters. Got %r instead." %
                             (self.kernels, ))
        else:
            n_kernels = len(self.kernels)
            kernels_params = self.kernels_params or [{}] * n_kernels

            if Y is not None and not issparse(X):
                Y = backend.asarray_like(Y, ref=X)

            kernels = []
            for metric, params in zip(self.kernels, kernels_params):
                kernel = pairwise_kernels(X, Y, metric=metric, **params)
                kernels.append(kernel)
            kernels = backend.stack(kernels)

        return kernels

    @force_cpu_backend
    def get_primal_coef(self, Xs_fit):
        """Returns the primal coefficients, assuming all kernels are linear.

        When all kernels are linear, weighted kernel ridge regression is
        equivalent to weighted ridge rergession, and the ridge regression
        (primal) coefficients can be computed from the kernel ridge regression
        (dual) coefficients, using the training features Xs_fit.

        This currently only works when self.kernels == "precomputed".

        Parameters
        ----------
        Xs_fit : list of array of shape (n_samples_fit, n_features)
            Training features. If you used a ColumnKernelizer, you can use the
            method `get_X_fit` to get this list of arrays.

        Returns
        -------
        primal_coef : list of array of shape (n_features, n_targets)
            Coefficient of the equivalent ridge regression. The coefficients
            are computed on CPU memory, since they can be large.
        """
        check_is_fitted(self)

        if self.kernels == "precomputed":
            if Xs_fit is None:
                raise ValueError(
                    "get_primal_coef requires the training features `Xs_fit`. "
                    "If you used a ColumnKernelizer, you can use the method "
                    "`get_X_fit`.")
            primal_coef = primal_weights_weighted_kernel_ridge(
                self.dual_coef_, self.deltas_, Xs_fit)
        else:
            raise ValueError("The primal coefficients can only be computed "
                             "when using precomputed kernels.")

        return primal_coef


class MultipleKernelRidgeCV(_BaseWeightedKernelRidge):
    """Multiple-kernel ridge regression with cross-validation.

    Solve the kernel ridge regression::

        w* = argmin_w ||K @ w - Y||^2 + (w.T @ K @ w)

    where the kernel K is a weighted sum of multiple kernels Ks::

        K = sum_i exp(deltas[i]) Ks[i]

    The solver optimizes the log kernel weight ``deltas`` over
    cross-validation, using random search (``solver="random_search"``), or
    hyperparameter gradient descent (``solver="hyper_gradient"``).

    Parameters
    ----------
    kernels : list of (str or callable), default=["linear", "polynomial"]
        List of kernel mapping. Available kernels are: 'linear',
        'polynomial, 'poly', 'rbf', 'sigmoid', 'cosine'.
        Set to 'precomputed' in order to pass a precomputed kernel matrix to
        the estimator methods instead of samples.
        A callable should accept two arguments and the keyword arguments passed
        to this object as kernel_params, and should return a floating point
        number.

    kernels_params : list of dict, or None
        Additional parameters for the kernel functions.
        See more details in the docstring of the function:
        ``MultipleKernelRidgeCV.ALL_KERNELS[kernel]``

    solver : str
        Algorithm used during the fit, "random_search", or "hyper_gradient".

    solver_params : dict or None
        Additional parameters for the solver.
        See more details in the docstring of the function:
        ``MultipleKernelRidgeCV.ALL_SOLVERS[solver]``

    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.

    random_state : int, or None
        Random generator seed. Use an int for deterministic search.

    Y_in_cpu : bool
        If True, keep the target values ``y`` in CPU memory (slower).

    force_cpu : bool
        If True, computations will be performed on CPU, ignoring the
        current backend. If False, use the current backend.

    Attributes
    ----------
    dual_coef_ : array of shape (n_samples) or (n_samples, n_targets)
        Representation of weight vectors in kernel space.

    deltas_ : array of shape (n_kernels, n_targets)
        Log of kernel weights.

    cv_scores_ : array of shape (n_iter, n_targets)
        Cross-validation scores, averaged over splits.
        By default, the scores are computed with l2_neg_loss (in ]-inf, 0]).
        The scoring function can be changed with solver_params["score_func"].

    X_fit_ : array of shape (n_samples, n_features)
        Training data. If ``kernels == "precomputed"`` this is None.

    n_features_in_ : int
        Number of features (or number of samples if
        ``kernels == "precomputed"``) used during the fit.

    dtype_ : str
        Dtype of input data.

    best_alphas_ : array of shape (n_targets, )
        Equal to ``1. / exp(self.deltas_).sum(0)``. For the "random_search"
        solver, it corresponds to the best hyperparameter alphas, assuming that
        each kernel weight vector sums to one (in particular, it is the case
        when ``solver_params['n_iter']`` is an integer).

    Examples
    --------
    >>> from himalaya.kernel_ridge import MultipleKernelRidgeCV
    >>> from himalaya.kernel_ridge import ColumnKernelizer
    >>> from himalaya.kernel_ridge import Kernelizer
    >>> from sklearn.pipeline import make_pipeline

    >>> # create a dataset
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 10, 5, 3
    >>> X = np.random.randn(n_samples, n_features)
    >>> Y = np.random.randn(n_samples, n_targets)

    >>> # Kernelize separately the first three columns and the last two
    >>> # columns, creating two kernels of shape (n_samples, n_samples).
    >>> ck = ColumnKernelizer(
    ...     [("kernel_1", Kernelizer(kernel="linear"), [0, 1, 2]),
    ...      ("kernel_2", Kernelizer(kernel="polynomial"), slice(3, 5))])

    >>> # A model with precomputed kernels, as output by ColumnKernelizer
    >>> model = MultipleKernelRidgeCV(kernels="precomputed")
    >>> pipe = make_pipeline(ck, model)
    >>> _ = pipe.fit(X, Y)
    """
    ALL_SOLVERS = MULTIPLE_KERNEL_RIDGE_SOLVERS

    def __init__(self, kernels=["linear", "polynomial"], kernels_params=None,
                 solver="random_search", solver_params=None, cv=5,
                 random_state=None, Y_in_cpu=False, force_cpu=False):
        self.kernels = kernels
        self.kernels_params = kernels_params
        self.solver = solver
        self.solver_params = solver_params
        self.cv = cv
        self.random_state = random_state
        self.Y_in_cpu = Y_in_cpu
        self.force_cpu = force_cpu

    @force_cpu_backend
    def fit(self, X, y=None, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data. If kernels == "precomputed" this is instead
            a precomputed kernel array of shape
            (n_kernels, n_samples, n_samples).

        y : array of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : None, or array of shape (n_samples, )
            Individual weights for each sample, ignored if None is passed.

        Returns
        -------
        self : returns an instance of self.
        """
        backend = get_backend()

        ndim = 3 if self.kernels == "precomputed" else 2
        accept_sparse = False if self.kernels == "precomputed" else ("csr",
                                                                     "csc")
        X = check_array(X, accept_sparse=accept_sparse, ndim=ndim)
        self.dtype_ = _get_string_dtype(X)
        device = "cpu" if self.Y_in_cpu else None
        y = check_array(y, dtype=self.dtype_, ndim=[1, 2], device=device)

        n_samples = X.shape[1] if self.kernels == "precomputed" else X.shape[0]
        if n_samples != y.shape[0]:
            raise ValueError("Inconsistent number of samples.")

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, dtype=self.dtype_,
                                        ndim=1)
            if sample_weight.shape[0] != y.shape[0]:
                raise ValueError("Inconsistent number of samples.")

        Ks = self._get_kernels(X)

        self.X_fit_ = _to_cpu(X) if self.kernels != "precomputed" else None
        self.n_features_in_ = X.shape[-1]
        del X

        ravel = False
        if y.ndim == 1:
            y = y[:, None]
            ravel = True

        if sample_weight is not None:
            # We need to support sample_weight directly because K might be a
            # precomputed kernel.
            sw = backend.sqrt(sample_weight)[:, None]
            y = y * backend.to_cpu(sw) if self.Y_in_cpu else y * sw
            Ks *= (sw @ sw.T)[None]

        cv = check_cv(self.cv, y)

        # ------------------ call the solver
        tmp = self._call_solver(Ks=Ks, Y=y, cv=cv, return_weights="dual",
                                Xs=None, random_state=self.random_state,
                                Y_in_cpu=self.Y_in_cpu)
        self.deltas_, self.dual_coef_, self.cv_scores_ = tmp

        if self.solver == "random_search":
            self.best_alphas_ = 1. / backend.exp(self.deltas_).sum(0)
        else:
            self.best_alphas_ = None

        if ravel:
            self.dual_coef_ = self.dual_coef_[:, 0]
            self.deltas_ = self.deltas_[:, 0]

        return self

    def _more_tags(self):
        return {
            '_xfail_checks': {
                'check_sample_weights_invariance':
                'zero sample_weight is not equivalent to removing samples, '
                'because of the cross-validation splits.',
            }
        }


class WeightedKernelRidge(_BaseWeightedKernelRidge):
    """Weighted kernel ridge regression.

    Solve the kernel ridge regression::

        w* = argmin_w ||K @ w - Y||^2 + alpha (w.T @ K @ w)

    where the kernel K is a weighted sum of multiple kernels::

        K = sum_i exp(deltas[i]) Ks[i]

    Contrarily to ``MultipleKernelRidgeCV``, this model does not optimize the
    log kernel-weights ``deltas``. However, it is not equivalent to
    ``KernelRidge``, since the log kernel-weights ``deltas`` can be different
    for each target, therefore the kernel sum is not precomputed.

    Parameters
    ----------
    alpha : float, or array of shape (n_targets, )
        L2 regularization parameter.

    deltas : array of shape (n_kernels, ) or (n_kernels, n_targets)
        Kernel weights.
        Default to "zeros", an array of shape (n_kernels, ) filled with zeros.

    kernels : list of (str or callable), default=["linear", "polynomial"]
        List of kernel mapping. Available kernels are: 'linear',
        'polynomial, 'poly', 'rbf', 'sigmoid', 'cosine'.
        Set to 'precomputed' in order to pass a precomputed kernel matrix to
        the estimator methods instead of samples.
        A callable should accept two arguments and the keyword arguments passed
        to this object as kernel_params, and should return a floating point
        number.

    kernels_params : list of dict, or None
        Additional parameters for the kernel functions.
        See more details in the docstring of the function:
        ``WeightedKernelRidge.ALL_KERNELS[kernel]``

    solver : str
        Algorithm used during the fit, "conjugate_gradient", or
        "gradient_descent".

    solver_params : dict or None
        Additional parameters for the solver.
        See more details in the docstring of the function:
        ``WeightedKernelRidge.ALL_SOLVERS[solver]``

    random_state : int, or None
        Random generator seed. Use an int for deterministic search.

    force_cpu : bool
        If True, computations will be performed on CPU, ignoring the
        current backend. If False, use the current backend.

    Attributes
    ----------
    dual_coef_ : array of shape (n_samples) or (n_samples, n_targets)
        Representation of weight vectors in kernel space.

    deltas_ : array of shape (n_kernels, n_targets) or (n_kernels, )
        Log of kernel weights.

    X_fit_ : array of shape (n_samples, n_features)
        Training data. If kernels == "precomputed" this is None.

    n_features_in_ : int
        Number of features (or number of samples if kernels == "precomputed")
        used during the fit.

    dtype_ : str
        Dtype of input data.

    Examples
    --------
    >>> from himalaya.kernel_ridge import WeightedKernelRidge
    >>> from himalaya.kernel_ridge import ColumnKernelizer
    >>> from himalaya.kernel_ridge import Kernelizer
    >>> from sklearn.pipeline import make_pipeline

    >>> # create a dataset
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 10, 5, 3
    >>> X = np.random.randn(n_samples, n_features)
    >>> Y = np.random.randn(n_samples, n_targets)

    >>> # Kernelize separately the first three columns and the last two
    >>> # columns, creating two kernels of shape (n_samples, n_samples).
    >>> ck = ColumnKernelizer(
    ...     [("kernel_1", Kernelizer(kernel="linear"), [0, 1, 2]),
    ...      ("kernel_2", Kernelizer(kernel="polynomial"), slice(3, 5))])

    >>> # A model with precomputed kernels, as output by ColumnKernelizer
    >>> model = WeightedKernelRidge(kernels="precomputed")
    >>> pipe = make_pipeline(ck, model)
    >>> pipe.fit(X, Y)
    """
    ALL_SOLVERS = WEIGHTED_KERNEL_RIDGE_SOLVERS

    def __init__(self, alpha=1., deltas="zeros",
                 kernels=["linear", "polynomial"], kernels_params=None,
                 solver="conjugate_gradient", solver_params=None,
                 random_state=None, force_cpu=False):
        self.alpha = alpha
        self.deltas = deltas
        self.kernels = kernels
        self.kernels_params = kernels_params
        self.solver = solver
        self.solver_params = solver_params
        self.random_state = random_state
        self.force_cpu = force_cpu

    @force_cpu_backend
    def fit(self, X, y=None, sample_weight=None):
        """Fit kernel ridge regression model

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data. If kernels == "precomputed" this is instead
            a precomputed kernel array of shape
            (n_kernels, n_samples, n_samples).

        y : array of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : None, or array of shape (n_samples, )
            Individual weights for each sample, ignored if None is passed.

        Returns
        -------
        self : returns an instance of self.
        """
        backend = get_backend()

        ndim = 3 if self.kernels == "precomputed" else 2
        accept_sparse = False if self.kernels == "precomputed" else ("csr",
                                                                     "csc")
        X = check_array(X, accept_sparse=accept_sparse, ndim=ndim)
        self.dtype_ = _get_string_dtype(X)
        y = check_array(y, dtype=self.dtype_, ndim=[1, 2])

        n_samples = X.shape[1] if self.kernels == "precomputed" else X.shape[0]
        if n_samples != y.shape[0]:
            raise ValueError("Inconsistent number of samples.")

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, dtype=self.dtype_,
                                        ndim=1)
            if sample_weight.shape[0] != y.shape[0]:
                raise ValueError("Inconsistent number of samples.")

        Ks = self._get_kernels(X)

        self.X_fit_ = _to_cpu(X) if self.kernels != "precomputed" else None
        self.n_features_in_ = X.shape[-1]
        del X

        ravel = False
        if y.ndim == 1:
            y = y[:, None]
            ravel = True

        if sample_weight is not None:
            # We need to support sample_weight directly because K might be a
            # precomputed kernel.
            sw = backend.sqrt(sample_weight)[:, None]
            y = y * sw
            Ks *= (sw @ sw.T)[None]

        n_kernels = Ks.shape[0]
        if isinstance(self.deltas, str) and self.deltas == "zeros":
            self.deltas_ = backend.zeros_like(Ks, shape=(n_kernels, 1))
        else:
            self.deltas_ = check_array(self.deltas, ndim=[1, 2])
            if self.deltas_.shape[0] != n_kernels:
                raise ValueError("Inconsistent number of kernels.")
            if (self.deltas.ndim == 2 and y.ndim == 2
                    and self.deltas_.shape[1] != y.shape[1]):
                raise ValueError("Inconsistent number of targets.")
            if self.deltas_.ndim == 1:
                self.deltas_ = self.deltas_[:, None]

        # ------------------ call the solver
        self.dual_coef_ = self._call_solver(Ks=Ks, Y=y, alpha=self.alpha,
                                            deltas=self.deltas_,
                                            random_state=self.random_state)

        if ravel or self.deltas_.shape[1] != self.dual_coef_.shape[1]:
            self.deltas_ = self.deltas_[:, 0]
        if ravel:
            self.dual_coef_ = self.dual_coef_[:, 0]

        return self


def _to_cpu(X):
    backend = get_backend()
    if issparse(X):
        return X
    else:
        return backend.to_cpu(X)
