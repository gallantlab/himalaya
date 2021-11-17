import numbers

from ..backend import get_backend
from ..utils import compute_lipschitz_constants
from ._kernels import KernelCenterer


def _weighted_kernel_ridge_gradient(Ks, Y, dual_weights, exp_deltas, alpha=1.,
                                    double_K=False, return_objective=False):
    """Compute gradient of dual weights over a multi-kernel ridge regression.

    Parameters
    ----------
    Ks : array of shape (n_kernels, n_samples, n_samples)
        Input kernels.
    Y : array of shape (n_samples, n_targets)
        Target data.
    dual_weights : array of shape (n_samples, n_targets)
        Kernel ridge coefficients.
    exp_deltas : array of shape (n_kernels, ) or (n_kernels, n_targets)
        Kernel weights.
    alpha : float, or array of shape (n_targets, )
        Regularization parameter.
    double_K : bool
        If True, multiply the gradient by the kernel to obtain the true
        gradients, which are less well conditionned.
    return_objective : bool
        If True, returns the objective function as well as the gradient.

    Returns
    -------
    dual_weight_gradient : array of shape (n_samples, n_targets)
        Gradients of `dual_weights`.
    objective : array of shape (n_targets, )
        Objective function.
    """
    backend = get_backend()

    if exp_deltas.ndim == 1:
        exp_deltas = exp_deltas[:, None]

    Ks_times_dualweights = backend.stack(
        [backend.matmul(K, dual_weights).T for K in Ks], 2)

    predictions = backend.matmul(Ks_times_dualweights,
                                 exp_deltas.T[..., None])[..., 0].T
    residual = predictions - Y
    dampened_residual = residual + dual_weights * alpha

    if double_K:
        Ks_times_dampened_residual = backend.stack(
            [backend.matmul(K, dampened_residual).T for K in Ks], 2)
        dual_weight_gradient = backend.matmul(
            Ks_times_dampened_residual, exp_deltas.T[..., None])[..., 0].T
    else:
        dual_weight_gradient = dampened_residual

    if return_objective:
        objective = backend.norm(residual, axis=0) ** 2
        objective += alpha * (predictions * dual_weights).sum(0)

        return dual_weight_gradient, objective
    else:
        return dual_weight_gradient


def solve_weighted_kernel_ridge_gradient_descent(
        Ks, Y, deltas, alpha=1., fit_intercept=False, step_sizes=None,
        lipschitz_Ks=None, initial_dual_weights=None, max_iter=100, tol=1e-3,
        double_K=False, random_state=None, debug=False):
    """Solve weighted kernel ridge regression using gradient descent.

    Solve the kernel ridge regression::

        w* = argmin_w ||K @ w - Y||^2 + alpha (w.T @ K @ w)

    where the kernel K is a weighted sum of multiple kernels::

        K = sum_i exp(deltas[i]) Ks[i]

    Parameters
    ----------
    Ks : array of shape (n_kernels, n_samples, n_samples)
        Input kernels.
    Y : array of shape (n_samples, n_targets)
        Target data.
    deltas : array of shape (n_kernels, ) or (n_kernels, n_targets)
        Kernel weights.
    alpha : float, or array of shape (n_targets, )
        Regularization parameter.
    fit_intercept : boolean
        Whether to fit an intercept. If False, Ks should be centered
        (see KernelCenterer), and Y must be zero-mean over samples.
    step_sizes : float, or array of shape (n_targets), or None
        Step sizes.
        If None, computes a step size based on the Lipschitz constants.
    lipschitz_Ks : float, or array of shape (n_kernels), or None:
        Lipschitz constant.
        Used only if `step_sizes` is None.
        If None, Lipschitz constants are estimated with power iteration on Ks.
    initial_dual_weights : array of shape (n_samples, n_targets)
        Initial kernel ridge coefficients.
    max_iter : int
        Maximum number of gradient step.
    tol : float > 0 or None
        Tolerance for the stopping criterion.
    double_K : bool
        If True, multiply the gradient by the kernel to obtain the true
        gradients, which are less well conditionned.
    random_state : int, or None
        Random generator seed. Use an int for deterministic search.
    debug : bool
        If True, check some intermediate computations.

    Returns
    -------
    dual_weights : array of shape (n_samples, n_targets)
        Kernel ridge coefficients.
    intercept : array of shape (n_targets,)
        Intercept. Only returned when fit_intercept is True.
    """
    backend = get_backend()
    n_targets = Y.shape[1]

    if deltas.ndim == 1:
        deltas = deltas[:, None]
    if isinstance(alpha, numbers.Number) or alpha.ndim == 0:
        alpha = backend.ones_like(Y, shape=(1, )) * alpha

    Ks, Y, deltas, alpha, step_sizes, lipschitz_Ks, initial_dual_weights = \
        backend.check_arrays(Ks, Y, deltas, alpha, step_sizes, lipschitz_Ks,
                             initial_dual_weights)
    exp_deltas = backend.exp(deltas)

    if fit_intercept:
        Ks, Y, Ks_rows, Y_offset = _helper_intercept(Ks, Y)

    if initial_dual_weights is None:
        dual_weights = backend.zeros_like(Y)
    else:
        dual_weights = backend.copy(initial_dual_weights)

    if step_sizes is None:
        if lipschitz_Ks is None:
            lipschitz_Ks = compute_lipschitz_constants(
                Ks, random_state=random_state)
        if not double_K:
            lipschitz_Ks = backend.sqrt(lipschitz_Ks)

        total_lip = backend.matmul(lipschitz_Ks[None, :],
                                   exp_deltas)[0] + alpha
        step_sizes = 1. / total_lip

        if debug:
            assert not backend.any(backend.isnan(step_sizes))

    if isinstance(step_sizes, numbers.Number) or step_sizes.ndim == 0:
        step_sizes = backend.ones_like(Y, shape=(1, )) * step_sizes

    #######################
    # Gradient descent loop
    converged = backend.zeros_like(Y, dtype=backend.bool, shape=(n_targets))
    for i in range(max_iter):
        grads = _weighted_kernel_ridge_gradient(Ks, Y[:, ~converged],
                                                dual_weights[:, ~converged],
                                                exp_deltas=exp_deltas,
                                                alpha=alpha, double_K=double_K)
        update = step_sizes * grads
        dual_weights[:, ~converged] -= update

        ##########################
        # remove converged targets
        if tol is not None:
            relative_update = backend.abs(update / dual_weights[:, ~converged])
            relative_update[dual_weights[:, ~converged] == 0] = 0
            just_converged = backend.max(relative_update, 0) < tol

            if exp_deltas.shape[1] == just_converged.shape[0]:
                exp_deltas = exp_deltas[:, ~just_converged]
            if alpha.shape[0] == just_converged.shape[0]:
                alpha = alpha[~just_converged]
            if step_sizes.shape[0] == just_converged.shape[0]:
                step_sizes = step_sizes[~just_converged]
            converged[~converged] = just_converged

            if backend.all(converged):
                break

    if fit_intercept:
        intercept = Y_offset
        intercept -= ((Ks_rows @ dual_weights) * backend.exp(deltas)).sum(0)
        return dual_weights, intercept
    else:
        return dual_weights


def solve_weighted_kernel_ridge_conjugate_gradient(Ks, Y, deltas, alpha=1.,
                                                   fit_intercept=False,
                                                   initial_dual_weights=None,
                                                   max_iter=100, tol=1e-4,
                                                   random_state=None):
    """Solve weighted kernel ridge regression using conjugate gradient.

    Solve the kernel ridge regression::

        w* = argmin_w ||K @ w - Y||^2 + alpha (w.T @ K @ w)

    where the kernel K is a weighted sum of multiple kernels::

        K = sum_i exp(deltas[i]) Ks[i]

    Parameters
    ----------
    Ks : array of shape (n_kernels, n_samples, n_samples)
        Input kernels.
    Y : torch.Tensor of shape (n_samples, n_targets)
        Target data.
    deltas : array of shape (n_kernels, ) or (n_kernels, n_targets)
        Kernel weights.
    alpha : float, or array of shape (n_targets, )
        Regularization parameter.
    fit_intercept : boolean
        Whether to fit an intercept. If False, Ks should be centered
        (see KernelCenterer), and Y must be zero-mean over samples.
    initial_dual_weights : array of shape (n_samples, n_targets)
        Initial kernel ridge coefficients.
    max_iter : int
        Maximum number of conjugate gradient step.
    tol : float > 0 or None
        Tolerance for the stopping criterion.
    random_state : int, or None
        Random generator seed. Not used.

    Returns
    -------
    dual_weights : array of shape (n_samples, n_targets)
        Kernel ridge coefficients.
    intercept : array of shape (n_targets,)
        Intercept. Only returned when fit_intercept is True.
    """
    backend = get_backend()
    n_targets = Y.shape[1]

    if deltas.ndim == 1:
        deltas = deltas[:, None]
    if isinstance(alpha, numbers.Number) or alpha.ndim == 0:
        alpha = backend.ones_like(Y, shape=(1, )) * alpha

    Ks, Y, deltas, alpha, initial_dual_weights = backend.check_arrays(
        Ks, Y, deltas, alpha, initial_dual_weights)
    exp_deltas = backend.exp(deltas)

    if fit_intercept:
        Ks, Y, Ks_rows, Y_offset = _helper_intercept(Ks, Y)

    if initial_dual_weights is None:
        dual_weights = backend.zeros_like(Y)
    else:
        dual_weights = backend.copy(initial_dual_weights)

    # compute initial residual
    r = _weighted_kernel_ridge_gradient(Ks, Y, dual_weights,
                                        exp_deltas=exp_deltas, alpha=alpha,
                                        double_K=False)
    r *= -1
    p = backend.copy(r)
    new_squared_residual_norm = backend.norm(r, axis=0) ** 2

    #########################
    # Conjugate gradient loop
    converged = backend.zeros_like(Y, dtype=backend.bool, shape=(n_targets))
    for i in range(max_iter):
        Ks_x_p = backend.matmul(Ks, p)
        tmp = backend.matmul(backend.transpose(Ks_x_p, (2, 1, 0)),
                             backend.transpose(exp_deltas,
                                               (1, 0))[:, :, None])[..., 0]
        K_x_p_plus_reg = backend.transpose(tmp, (1, 0)) + alpha * p

        squared_residual_norm = new_squared_residual_norm
        squared_p_A_norm = backend.matmul(
            backend.transpose(p, (1, 0))[:, None],
            backend.transpose(K_x_p_plus_reg, (1, 0))[..., None])[:, 0, 0]

        squared_p_A_norm[squared_p_A_norm == 0] = 1
        alpha_step = squared_residual_norm / squared_p_A_norm

        update = alpha_step * p
        dual_weights[:, ~converged] += update

        r -= alpha_step * K_x_p_plus_reg

        new_squared_residual_norm = backend.norm(r, axis=0) ** 2

        beta = (new_squared_residual_norm) / (squared_residual_norm)
        p = r + beta * p

        ##########################
        # remove converged targets
        if tol is not None:
            relative_update = backend.abs(update / dual_weights[:, ~converged])
            relative_update[dual_weights[:, ~converged] == 0] = 0
            just_converged = backend.max(relative_update, 0) < tol

            r = r[:, ~just_converged]
            p = p[:, ~just_converged]
            new_squared_residual_norm = \
                new_squared_residual_norm[~just_converged]
            if exp_deltas.shape[1] == just_converged.shape[0]:
                exp_deltas = exp_deltas[:, ~just_converged]
            if alpha.shape[0] == just_converged.shape[0]:
                alpha = alpha[~just_converged]
            converged[~converged] = just_converged
            if backend.all(converged):
                break

    if fit_intercept:
        intercept = Y_offset
        intercept -= ((Ks_rows @ dual_weights) * backend.exp(deltas)).sum(0)
        return dual_weights, intercept
    else:
        return dual_weights


def solve_weighted_kernel_ridge_neumann_series(Ks, Y, deltas, alpha=1.,
                                               fit_intercept=False,
                                               max_iter=10, factor=0.0001,
                                               tol=None, random_state=None,
                                               debug=False):
    """Solve weighted kernel ridge regression using Neumann series.

    Solve the kernel ridge regression::

        w* = argmin_w ||K @ w - Y||^2 + alpha (w.T @ K @ w)

    where the kernel K is a weighted sum of multiple kernels::

        K = sum_i exp(deltas[i]) Ks[i]

    The Neumann series approximate the invert of K as K^-1 = sum_j (Id - K)^j.
    It is a poor approximation, so this solver should NOT be used to solve
    ridge. It is however useful during hyper-parameter gradient descent, as
    we do not need a good precision of the results, but merely the direction
    of the gradient.

    See [Lorraine, Vicol, & Duvenaud (2019). Optimizing Millions of
    Hyperparameters by Implicit Differentiation. arXiv:1911.02590].

    Parameters
    ----------
    Ks : array of shape (n_kernels, n_samples, n_samples)
        Input kernels.
    Y : torch.Tensor of shape (n_samples, n_targets)
        Target data.
    deltas : array of shape (n_kernels, ) or (n_kernels, n_targets)
        Kernel weights.
    alpha : float, or array of shape (n_targets, )
        Regularization parameter.
    fit_intercept : boolean
        Whether to fit an intercept. If False, Ks should be centered
        (see KernelCenterer), and Y must be zero-mean over samples.
    max_iter : int
        Number of terms in the Neumann series.
    factor : float, or array of shape (n_targets, )
        Factor used to allow convergence of the series. We actually invert
        (factor * K) instead of K, then multiply the result by factor.
    tol : None
        Not used.
    random_state : int, or None
        Random generator seed. Not used.
    debug : bool
        If True, check some intermediate computations.

    Returns
    -------
    dual_weights : array of shape (n_samples, n_targets)
        Kernel ridge coefficients.
    intercept : array of shape (n_targets,)
        Intercept. Only returned when fit_intercept is True.
    """
    backend = get_backend()

    if deltas.ndim == 1:
        deltas = deltas[:, None]
    if isinstance(alpha, numbers.Number) or alpha.ndim == 0:
        alpha = backend.ones_like(Y, shape=(1, )) * alpha
    if isinstance(factor, numbers.Number) or factor.ndim == 0:
        factor = backend.ones_like(Y, shape=(1, )) * factor

    Ks, Y, deltas, alpha, factor = backend.check_arrays(
        Ks, Y, deltas, alpha, factor)
    exp_deltas = backend.exp(deltas)

    if fit_intercept:
        Ks, Y, Ks_rows, Y_offset = _helper_intercept(Ks, Y)

    # product accumulator: product = (id_minus_K ** ii) @ Ys
    product = Y
    # sum accumulator: dual_weights = sum_ii product
    dual_weights = backend.zeros_like(Y)
    for ii in range(max_iter):
        product = (
            product * (1 - factor[None, :] * alpha[None, :]) -
            factor[None, :] * backend.sum(
                exp_deltas[:, None, :] * backend.matmul(Ks, product), axis=0))
        dual_weights += product

    dual_weights *= factor[None, :]

    if debug:
        assert not backend.any(backend.isinf(dual_weights))
        assert not backend.any(backend.isnan(dual_weights))

    if fit_intercept:
        intercept = Y_offset
        intercept -= ((Ks_rows @ dual_weights) * backend.exp(deltas)).sum(0)
        return dual_weights, intercept
    else:
        return dual_weights


def _helper_intercept(Ks, Y):
    """Transform Ks and Y if we fit an intercept."""
    backend = get_backend()

    # unfortunate, but we need to preserve the input. XXX add a copy parameter
    Ks = backend.copy(Ks)
    centerer = KernelCenterer()
    Ks_rows = backend.zeros_like(Ks, shape=(Ks.shape[0], Ks.shape[1]))
    for ii in range(Ks.shape[0]):
        Ks[ii] = centerer.fit_transform(Ks[ii])
        Ks_rows[ii] = centerer.K_fit_rows_

    Y = backend.copy(Y)
    Y_offset = Y.mean(0)
    Y -= Y_offset
    return Ks, Y, Ks_rows, Y_offset


#: Dictionary with all weighted kernel ridge solvers.
WEIGHTED_KERNEL_RIDGE_SOLVERS = {
    "neumann_series": solve_weighted_kernel_ridge_neumann_series,
    "conjugate_gradient": solve_weighted_kernel_ridge_conjugate_gradient,
    "gradient_descent": solve_weighted_kernel_ridge_gradient_descent,
}
###############################################################################


def solve_kernel_ridge_conjugate_gradient(K, Y, alpha=1., fit_intercept=False,
                                          initial_dual_weights=None,
                                          max_iter=100, tol=1e-3,
                                          random_state=None):
    """Solve kernel ridge regression using conjugate gradient.

    Solve the kernel ridge regression::

        w* = argmin_w ||K @ w - Y||^2 + alpha (w.T @ K @ w)

    Parameters
    ----------
    K : array of shape (n_samples, n_samples)
        Input kernel.
    Y : array of shape (n_samples, n_targets)
        Target data.
    alpha : float, or array of shape (n_targets, )
        Regularization parameter.
    fit_intercept : boolean
        Whether to fit an intercept. If False, K should be centered
        (see KernelCenterer), and Y must be zero-mean over samples.
    initial_dual_weights : array of shape (n_samples, n_targets)
        Initial kernel ridge coefficients.
    max_iter : int
        Maximum number of conjugate gradient step.
    tol : float > 0 or None
        Tolerance for the stopping criterion.
    random_state : int, or None
        Random generator seed. Not used.

    Returns
    -------
    dual_weights : array of shape (n_samples, n_targets)
        Kernel ridge coefficients.
    intercept : array of shape (n_targets,)
        Intercept. Only returned when fit_intercept is True.
    """
    backend = get_backend()
    deltas = backend.zeros_like(K, shape=(1, ))
    return solve_weighted_kernel_ridge_conjugate_gradient(
        K[None], Y=Y, deltas=deltas, alpha=alpha, fit_intercept=fit_intercept,
        initial_dual_weights=initial_dual_weights, max_iter=max_iter, tol=tol)


def solve_kernel_ridge_gradient_descent(K, Y, alpha=1., fit_intercept=False,
                                        step_sizes=None, lipschitz_Ks=None,
                                        initial_dual_weights=None,
                                        max_iter=100, tol=1e-3, double_K=False,
                                        random_state=None, debug=False):
    """Solve kernel ridge regression using conjugate gradient.

    Solve the kernel ridge regression

        w* = argmin_w ||K @ w - Y||^2 + alpha (w.T @ K @ w)

    Parameters
    ----------
    K : array of shape (n_samples, n_samples)
        Input kernel.
    Y : array of shape (n_samples, n_targets)
        Target data.
    alpha : float, or array of shape (n_targets, )
        Regularization parameter.
    fit_intercept : boolean
        Whether to fit an intercept. If False, K should be centered
        (see KernelCenterer), and Y must be zero-mean over samples.
    step_sizes : float, or array of shape (n_targets), or None
        Step sizes.
        If None, computes a step size based on the Lipschitz constants.
    lipschitz_Ks : float, or array of shape (n_kernels), or None:
        Lipschitz constant.
        Used only if `step_sizes` is None.
        If None, Lipschitz constants are estimated with power iteration on Ks.
    initial_dual_weights : array of shape (n_samples, n_targets)
        Initial kernel ridge coefficients.
    max_iter : int
        Maximum number of gradient step.
    tol : float > 0 or None
        Tolerance for the stopping criterion.
    double_K : bool
        If True, multiply the gradient by the kernel to obtain the true
        gradients, which are less well conditionned.
    random_state : int, or None
        Random generator seed. Use an int for deterministic search.
    debug : bool
        If True, check some intermediate computations.

    Returns
    -------
    dual_weights : array of shape (n_samples, n_targets)
        Kernel ridge coefficients.
    intercept : array of shape (n_targets,)
        Intercept. Only returned when fit_intercept is True.
    """
    backend = get_backend()
    deltas = backend.zeros_like(K, shape=(1, ))
    return solve_weighted_kernel_ridge_gradient_descent(
        K[None], Y=Y, deltas=deltas, alpha=alpha, step_sizes=step_sizes,
        lipschitz_Ks=lipschitz_Ks, initial_dual_weights=initial_dual_weights,
        max_iter=max_iter, tol=tol, double_K=double_K,
        random_state=random_state, debug=debug, fit_intercept=fit_intercept)


def solve_kernel_ridge_eigenvalues(K, Y, alpha=1., method="eigh",
                                   fit_intercept=False,
                                   negative_eigenvalues="zeros",
                                   random_state=None):
    """Solve kernel ridge regression using eigenvalues decomposition.

    Solve the kernel ridge regression::

        w* = argmin_w ||K @ w - Y||^2 + alpha (w.T @ K @ w)

    Parameters
    ----------
    K : array of shape (n_samples, n_samples)
        Input kernel.
    Y : array of shape (n_samples, n_targets)
        Target data.
    alpha : float, or array of shape (n_targets, )
        Regularization parameter.
    method : str in {"eigh", "svd"}
        Method used to diagonalize the kernel.
    fit_intercept : boolean
        Whether to fit an intercept. If False, K should be centered
        (see KernelCenterer), and Y must be zero-mean over samples.
    negative_eigenvalues : str in {"nan", "error", "zeros"}
        If the decomposition leads to negative eigenvalues (wrongly emerging
        from float32 errors):
        - "error" raises an error.
        - "zeros" remplaces them with zeros.
        - "nan" returns nans if the regularization does not compensate
        twice the smallest negative value, else it ignores the problem.
    random_state : int, or None
        Random generator seed. Not used.

    Returns
    -------
    dual_weights : array of shape (n_samples, n_targets)
        Kernel ridge coefficients.
    intercept : array of shape (n_targets,)
        Intercept. Only returned when fit_intercept is True.
    """
    backend = get_backend()
    if isinstance(alpha, numbers.Number) or alpha.ndim == 0:
        alpha = backend.ones_like(Y, shape=(1, )) * alpha

    K, Y, alpha = backend.check_arrays(K, Y, alpha)

    centerer, Y_offset = None, None
    if fit_intercept:
        centerer = KernelCenterer()
        K = centerer.fit_transform(K)
        Y_offset = Y.mean(0)
        Y = Y - Y_offset

    if method == "eigh":
        # diagonalization: K = V @ np.diag(eigenvalues) @ V.T
        eigenvalues, V = backend.eigh(K)
        # match SVD notations: K = U @ np.diag(eigenvalues) @ Vt
        U = V
        Vt = V.T
    elif method == "svd":
        # SVD: K = U @ np.diag(eigenvalues) @ Vt
        U, eigenvalues, Vt = backend.svd(K)
    else:
        raise ValueError("Unknown method=%r." % (method, ))

    inverse = 1 / (alpha[None] + eigenvalues[:, None])

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

    iUT = inverse[:, None, :] * U.T[:, :, None]
    iUT = backend.transpose(iUT, (2, 0, 1))
    # Vt.T.shape = (n_samples, n_samples)
    # iUT.shape = (1 or n_targets, n_samples, n_samples)
    # Y.T.shape = (n_targets, n_samples)
    # dual_weights = Vt.T @ iUT @ Y.T (batching over n_targets)

    if Y.shape[0] < Y.shape[1]:
        dual_weights = ((Vt.T @ iUT) @ Y.T[:, :, None])[:, :, 0].T
    else:
        dual_weights = Vt.T @ (iUT @ Y.T[:, :, None])[:, :, 0].T

    if fit_intercept:
        intercept = Y_offset - centerer.K_fit_rows_ @ dual_weights
        return dual_weights, intercept
    else:
        return dual_weights


#: Dictionary with all kernel ridge solvers
KERNEL_RIDGE_SOLVERS = {
    "eigenvalues": solve_kernel_ridge_eigenvalues,
    "conjugate_gradient": solve_kernel_ridge_conjugate_gradient,
    "gradient_descent": solve_kernel_ridge_gradient_descent,
}
