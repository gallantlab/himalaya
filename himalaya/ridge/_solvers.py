import numbers
import warnings

from ..backend import get_backend
from ..utils import _batch_or_skip


def solve_ridge_svd(X, Y, alpha=1., method="svd", fit_intercept=False,
                    negative_eigenvalues="zeros", n_targets_batch=None,
                    warn=True):
    """Solve ridge regression using SVD decomposition.

    Solve the ridge regression::

        b* = argmin_B ||X @ b - Y||^2 + alpha ||b||^2

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Input features.
    Y : array of shape (n_samples, n_targets)
        Target data.
    alpha : float, or array of shape (n_targets, )
        Regularization parameter.
    method : str in {"svd"}
        Method used to diagonalize the input feature matrix.
    fit_intercept : boolean
        Whether to fit an intercept.
        If False, X and Y must be zero-mean over samples.
    negative_eigenvalues : str in {"nan", "error", "zeros"}
        If the decomposition leads to negative eigenvalues (wrongly emerging
        from float32 errors):
        - "error" raises an error.
        - "zeros" replaces them with zeros.
        - "nan" returns nans if the regularization does not compensate
        twice the smallest negative value, else it ignores the problem.
    n_targets_batch : int or None
        Size of the batch for over targets during cross-validation.
        Used for memory reasons. If None, uses all n_targets at once.
    warn : bool
        If True, warn if the number of samples is smaller than the number of
        features.

    Returns
    -------
    weights : array of shape (n_features, n_targets)
        Ridge coefficients.
    intercept : array of shape (n_targets,)
        Intercept. Only returned when fit_intercept is True.
    """
    backend = get_backend()
    if isinstance(alpha, numbers.Number) or alpha.ndim == 0:
        alpha = backend.ones_like(Y, shape=(1, )) * alpha

    X, Y, alpha = backend.check_arrays(X, Y, alpha)

    n_samples, n_features = X.shape
    if n_samples < n_features and warn:
        warnings.warn(
            "Solving ridge is slower than solving kernel ridge when n_samples "
            f"< n_features (here {n_samples} < {n_features}). "
            "Using a linear kernel in himalaya.kernel_ridge.KernelRidge or "
            "himalaya.kernel_ridge.solve_kernel_ridge_eigenvalues would be "
            "faster. Use warn=False to silence this warning.", UserWarning)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")

    X_offset, Y_offset = None, None
    if fit_intercept:
        X_offset = X.mean(0)
        Y_offset = Y.mean(0)
        X = X - X_offset
        Y = Y - Y_offset

    if method == "svd":
        # SVD: X = U @ np.diag(eigenvalues) @ Vt
        U, eigenvalues, Vt = backend.svd(X, full_matrices=False)
    else:
        raise ValueError("Unknown method=%r." % (method, ))

    inverse = eigenvalues[:, None] / (alpha[None] + eigenvalues[:, None] ** 2)

    # negative eigenvalues can emerge from incorrect kernels, or from float32
    if eigenvalues[0] < 0:
        if negative_eigenvalues == "nan":
            if alpha < -eigenvalues[0] * 2:
                return backend.ones_like(Y) * backend.asarray(
                    backend.nan, dtype=Y.dtype)
            else:
                pass

        elif negative_eigenvalues == "zeros":
            eigenvalues[eigenvalues < 0] = 0

        elif negative_eigenvalues == "error":
            raise RuntimeError(
                "Negative eigenvalues. Make sure the kernel is positive "
                "semi-definite, increase the regularization alpha, or use"
                "another solver.")
        else:
            raise ValueError("Unknown negative_eigenvalues=%r." %
                             (negative_eigenvalues, ))

    n_samples, n_features = X.shape
    n_samples, n_targets = Y.shape
    weights = backend.zeros_like(X, shape=(n_features, n_targets),
                                 device="cpu")
    if n_targets_batch is None:
        n_targets_batch = n_targets

    for start in range(0, n_targets, n_targets_batch):
        batch = slice(start, start + n_targets_batch)

        iUT = _batch_or_skip(inverse, batch, 1)[:, None, :] * U.T[:, :, None]
        iUT = backend.transpose(iUT, (2, 0, 1))
        # iUT.shape = (1 or n_targets_batch, n_samples, n_samples)

        if Y.shape[0] < Y.shape[1]:
            weights_batch = ((Vt.T @ iUT) @ Y.T[batch, :, None])[:, :, 0].T
        else:
            weights_batch = Vt.T @ (iUT @ Y.T[batch, :, None])[:, :, 0].T
        weights[:, batch] = backend.to_cpu(weights_batch)

    if fit_intercept:
        intercept = backend.to_cpu(
            Y_offset) - backend.to_cpu(X_offset) @ weights
        return weights, intercept
    else:
        return weights


#: Dictionary with all ridge solvers
RIDGE_SOLVERS = {"svd": solve_ridge_svd}
