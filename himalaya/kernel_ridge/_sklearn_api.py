from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import check_cv

from ._solvers import KERNEL_RIDGE_SOLVERS
from ._solvers import WEIGHTED_KERNEL_RIDGE_SOLVERS
from ._hyper_gradient import MULTIPLE_KERNEL_RIDGE_SOLVERS
from ._random_search import solve_kernel_ridge_cv_eigenvalues
from ._kernels import pairwise_kernels
from ._kernels import PAIRWISE_KERNEL_FUNCTIONS
from ._predictions import predict_weighted_kernel_ridge
from ._predictions import predict_and_score_weighted_kernel_ridge

from ..validation import check_array
from ..validation import issparse
from ..validation import _get_string_dtype
from ..backend import get_backend
from ..scoring import r2_score


class _BaseKernelRidge(ABC, MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Base class for kernel ridge estimators"""

    ALL_KERNELS = PAIRWISE_KERNEL_FUNCTIONS

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


class KernelRidge(_BaseKernelRidge):
    """Kernel ridge regression.

    Solve the kernel ridge regression

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
            KernelRidge.ALL_KERNELS[kernel]

    solver : str
        Algorithm used during the fit, "eigenvalues", "conjugate_gradient", or
        "gradient_descent".

    solver_params : dict or None
        Additional parameters for the solver.
        See more details in the docstring of the function:
            KernelRidge.ALL_SOLVERS[solver]

    Attributes
    ----------
    dual_coef_ : array of shape (n_samples) or (n_samples, n_targets)
        Representation of weight vectors in kernel space.

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
                 solver="eigenvalues", solver_params=None):
        self.alpha = alpha
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.solver = solver
        self.solver_params = solver_params

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

        # ------------------ call the solver
        self.dual_coef_ = self._call_solver(K=K, Y=y, alpha=self.alpha)

        if self.solver not in self.ALL_SOLVERS:
            raise ValueError("Unknown solver=%r." % self.solver)

        if ravel:
            self.dual_coef_ = self.dual_coef_[:, 0]

        return self

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
        accept_sparse = False if self.kernel == "precomputed" else ("csr",
                                                                    "csc")
        X = check_array(X, dtype=self.dtype_, accept_sparse=accept_sparse,
                        ndim=2)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                'Different number of features in X than during fit.')

        K = self._get_kernel(X, self.X_fit_)
        del X

        Y_hat = K @ self.dual_coef_
        return Y_hat

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
            KernelRidgeCV.ALL_KERNELS[kernel]

    solver : str
        Algorithm used during the fit, "eigenvalues" only for now.

    solver_params : dict or None
        Additional parameters for the solver.
        See more details in the docstring of the function:
            KernelRidgeCV.ALL_SOLVERS[solver]

    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.

    Attributes
    ----------
    dual_coef_ : array of shape (n_samples) or (n_samples, n_targets)
        Representation of weight vectors in kernel space.

    best_alphas_ : array of shape (n_targets, )
        Selected best hyperparameter alphas.

    cv_scores_ : array of shape (n_targets, )
        Cross-validation scores averaged over splits, for the best alpha.

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
                 solver="eigenvalues", solver_params=None, cv=5):
        self.alphas = alphas
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.solver = solver
        self.solver_params = solver_params
        self.cv = cv

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
        y = check_array(y, dtype=self.dtype_, ndim=[1, 2])
        if X.shape[0] != y.shape[0]:
            raise ValueError("Inconsistent number of samples.")

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, dtype=self.dtype_,
                                        ndim=1)
            if sample_weight.shape[0] != y.shape[0]:
                raise ValueError("Inconsistent number of samples.")

        alphas = check_array(self.alphas, dtype=self.dtype_, ndim=1)

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
            y = y * sw
            K *= sw @ sw.T

        cv = check_cv(self.cv)

        # ------------------ call the solver
        tmp = self._call_solver(K=K, Y=y, cv=cv, alphas=alphas)
        self.best_alphas_, self.dual_coef_, self.cv_scores_ = tmp
        self.cv_scores_ = self.cv_scores_[0]

        if ravel:
            self.dual_coef_ = self.dual_coef_[:, 0]

        return self


###############################################################################
###############################################################################
###############################################################################
###############################################################################


class _BaseWeightedKernelRidge(_BaseKernelRidge):
    """Private class for shared implementations.
    """

    def predict(self, X):
        """Predict using the model.

        Parameters
        ----------
        X : array of shape (n_samples_test, n_features)
            Samples. If kernels == "precomputed" this is instead a precomputed
            kernel array of shape (n_kernels, n_samples_test, n_samples_train).

        Returns
        -------
        Y_hat : array of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
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

        if self.dual_coef_.ndim == 1:
            Y_hat = predict_weighted_kernel_ridge(
                Ks=Ks, dual_weights=self.dual_coef_[:, None],
                deltas=self.deltas_[:, None])[:, 0]
        else:
            Y_hat = predict_weighted_kernel_ridge(Ks=Ks,
                                                  dual_weights=self.dual_coef_,
                                                  deltas=self.deltas_)
        return Y_hat

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array of shape (n_samples_test, n_features)
            Samples. If kernels == "precomputed" this is instead a precomputed
            kernel array of shape (n_kernels, n_samples_test, n_samples_train).

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for X.

        Returns
        -------
        score : array of shape (n_targets, )
            R^2 of self.predict(X) versus y.
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

        if self.dual_coef_.ndim == 1:
            score = predict_and_score_weighted_kernel_ridge(
                Ks=Ks, dual_weights=self.dual_coef_[:, None],
                deltas=self.deltas_[:, None], Y=y[:, None],
                score_func=r2_score, n_targets_batch=n_targets_batch)[0]
        else:
            score = predict_and_score_weighted_kernel_ridge(
                Ks=Ks, dual_weights=self.dual_coef_, deltas=self.deltas_, Y=y,
                score_func=r2_score, n_targets_batch=n_targets_batch)
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


class MultipleKernelRidgeCV(_BaseWeightedKernelRidge):
    """Multiple-kernel ridge regression with cross-validation.

    Solve the kernel ridge regression

        w* = argmin_w ||K @ w - Y||^2 + (w.T @ K @ w)

    where the kernel K is a weighted sum of multiple kernels Ks:

        K = sum_i exp(deltas[i]) Ks[i]

    The solver optimizes the log kernel weight `deltas` over cross-validation,
    using random search (solver="random_search"), or hyperparameter gradient
    descent (solver="hyper_gradient").

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
            MultipleKernelRidgeCV.ALL_KERNELS[kernel]

    solver : str
        Algorithm used during the fit, "random_search", or "hyper_gradient".

    solver_params : dict or None
        Additional parameters for the solver.
        See more details in the docstring of the function:
            MultipleKernelRidgeCV.ALL_SOLVERS[solver]

    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.

    Attributes
    ----------
    dual_coef_ : array of shape (n_samples) or (n_samples, n_targets)
        Representation of weight vectors in kernel space.

    deltas_ : array of shape (n_kernels, n_targets)
        Log of kernel weights.

    cv_scores_ : array of shape (n_iter, n_targets)
        Cross-validation scores, averaged over splits.

    X_fit_ : array of shape (n_samples, n_features)
        Training data. If kernels == "precomputed" this is None.

    n_features_in_ : int
        Number of features (or number of samples if kernels == "precomputed")
        used during the fit.

    dtype_ : str
        Dtype of input data.

    Examples
    --------
    >>> from himalaya.kernel_ridge import MultipleKernelRidgeCV
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 10, 5, 3
    >>> X = np.random.randn(n_samples, n_features)
    >>> Y = np.random.randn(n_samples, n_targets)
    >>> model = MultipleKernelRidgeCV()
    >>> model.fit(X, Y)
    MultipleKernelRidgeCV()
    """
    ALL_SOLVERS = MULTIPLE_KERNEL_RIDGE_SOLVERS

    def __init__(self, kernels=["linear", "polynomial"], kernels_params=None,
                 solver="random_search", solver_params=None, cv=5):
        self.kernels = kernels
        self.kernels_params = kernels_params
        self.solver = solver
        self.solver_params = solver_params
        self.cv = cv

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

        cv = check_cv(self.cv)

        # ------------------ call the solver
        tmp = self._call_solver(Ks=Ks, Y=y, cv=cv, return_weights="dual",
                                Xs=None)
        self.deltas_, self.dual_coef_, self.cv_scores_ = tmp

        if ravel:
            self.dual_coef_ = self.dual_coef_[:, 0]
            self.deltas_ = self.deltas_[:, 0]

        return self


class WeightedKernelRidge(_BaseWeightedKernelRidge):
    """Weighted kernel ridge regression.

    Solve the kernel ridge regression

        w* = argmin_w ||K @ w - Y||^2 + alpha (w.T @ K @ w)

    where the kernel K is a weighted sum of multiple kernels:

        K = sum_i exp(deltas[i]) Ks[i]

    Contrarily to MultipleKernelRidgeCV, this model does not optimize the log
    kernel weights `deltas`. However, it is not equivalent to KernelRidge,
    since the log kernel weights `deltas` can be different for each target,
    therefore the kernel sum is not precomputed.

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
            WeightedKernelRidge.ALL_KERNELS[kernel]

    solver : str
        Algorithm used during the fit, "conjugate_gradient", or
        "gradient_descent".

    solver_params : dict or None
        Additional parameters for the solver.
        See more details in the docstring of the function:
            WeightedKernelRidge.ALL_SOLVERS[solver]

    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.

    Attributes
    ----------
    dual_coef_ : array of shape (n_samples) or (n_samples, n_targets)
        Representation of weight vectors in kernel space.

    deltas_ : array of shape (n_kernels, n_targets)
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
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 10, 5, 3
    >>> X = np.random.randn(n_samples, n_features)
    >>> Y = np.random.randn(n_samples, n_targets)
    >>> model = WeightedKernelRidge()
    >>> model.fit(X, Y)
    WeightedKernelRidge()
    """
    ALL_SOLVERS = WEIGHTED_KERNEL_RIDGE_SOLVERS

    def __init__(self, alpha=1., deltas="zeros",
                 kernels=["linear", "polynomial"], kernels_params=None,
                 solver="conjugate_gradient", solver_params=None):
        self.alpha = alpha
        self.deltas = deltas
        self.kernels = kernels
        self.kernels_params = kernels_params
        self.solver = solver
        self.solver_params = solver_params

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
            if (self.deltas_ndim == 2 and y.ndim == 2
                    and self.deltas_.shape[1] != y.shape[1]):
                raise ValueError("Inconsistent number of targets.")
            if self.deltas_.ndim == 1:
                self.deltas_ = self.deltas_[:, None]

        # ------------------ call the solver
        self.dual_coef_ = self._call_solver(Ks=Ks, Y=y, deltas=self.deltas_)

        if ravel:
            self.dual_coef_ = self.dual_coef_[:, 0]
            self.deltas_ = self.deltas_[:, 0]

        return self


def _to_cpu(X):
    backend = get_backend()
    if issparse(X):
        return X
    else:
        return backend.to_cpu(X)