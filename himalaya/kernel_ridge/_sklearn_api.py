from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import check_cv

from ._solvers import solve_kernel_ridge_eigenvalues
from ._solvers import solve_kernel_ridge_gradient_descent
from ._solvers import solve_kernel_ridge_conjugate_gradient
from ._random_search import solve_multiple_kernel_ridge_random_search
from ._hyper_gradient import solve_multiple_kernel_ridge_hyper_gradient
from ._kernels import pairwise_kernels
from ._predictions import predict_weighted_kernel_ridge
from ..validation import check_array
from ..validation import _get_string_dtype
from ..backend import get_backend


class KernelRidge(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Kernel ridge regression.

    Parameters
    ----------
    alpha : float, or array of shape (n_targets, )
        L2 regularization parameter.

    kernel : string or callable, default="linear"
        Kernel mapping used internally. Available kernels are: 'linear',
        'polynomial, 'poly', 'rbf', 'sigmoid', 'cosine', or 'precomputed'.
        Set to 'precomputed' in order to pass a precomputed kernel matrix to
        the estimator methods instead of samples.
        A callable should accept two arguments and the keyword arguments passed
        to this object as kernel_params, and should return a floating point
        number.

    kernel_params : dict or None
        Additional parameters for the kernel function.

    solver : str
        Algorithm used during the fit.

    solver_params : dict or None
        Additional parameters for the solver.

    Attributes
    ----------
    dual_coef_ : array of shape (n_samples) or (n_samples, n_targets)
        Representation of weight vectors in kernel space.

    X_fit_ : array of shape (n_samples, n_features)
        Training data. If kernel == "precomputed" this is instead
        a precomputed kernel array of shape (n_samples, n_samples).

    n_features_in_ : int
        Number of features (or number of samples if kernel == "precomputed")
        used during the fit.

    dtype_ : str
        Dtype of input data.

    Examples
    --------
    >>> from himalaya.ridge import KernelRidge
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 10, 5, 3
    >>> X = np.random.randn(n_samples, n_features)
    >>> Y = np.random.randn(n_samples, n_targets)
    >>> clf = KernelRidge()
    >>> clf.fit(X, Y)
    KernelRidge()
    """

    def __init__(self, alpha=1, kernel="linear", kernel_params=None,
                 solver="eigenvalues", solver_params=None):
        self.alpha = alpha
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.solver = solver
        self.solver_params = solver_params

    def fit(self, X, y=None, sample_weight=None):
        """Fit kernel ridge regression model

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
        accept_sparse = False if self.kernels == "precomputed" else ("csr",
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

        solver_params = self.solver_params or {}

        if self.solver == "eigenvalues":
            self.dual_coef_ = solve_kernel_ridge_eigenvalues(
                K, y, alpha=self.alpha)
        elif self.solver == 'conjugate':
            self.dual_coef_ = solve_kernel_ridge_conjugate_gradient(
                K, y, alpha=self.alpha, **solver_params)
        elif self.solver == 'gradient':
            self.dual_coef_ = solve_kernel_ridge_gradient_descent(
                K, y, alpha=self.alpha, **solver_params)
        else:
            raise ValueError("Unknown solver=%r." % self.solver)

        if ravel:
            self.dual_coef_ = self.dual_coef_[:, 0]

        self.X_fit_ = X
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict using the kernel ridge model

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
        accept_sparse = False if self.kernels == "precomputed" else ("csr",
                                                                     "csc")
        X = check_array(X, dtype=self.dtype_, accept_sparse=accept_sparse,
                        ndim=2)
        K = self._get_kernel(X, self.X_fit_)
        assert X.shape[1] == self.n_features_in_
        Y_hat = K @ self.dual_coef_
        return Y_hat

    def _get_kernel(self, X, Y=None):
        backend = get_backend()
        kernel_params = self.kernel_params or {}
        kernel = pairwise_kernels(X, Y, metric=self.kernel, **kernel_params)
        return backend.asarray(kernel)

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"
