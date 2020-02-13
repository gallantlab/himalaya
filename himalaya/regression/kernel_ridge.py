from ..backend import get_current_backend
from ..utils import compute_lipschitz_constants


def multi_kernel_ridge_gradient(Ks, Y, dual_weights, gammas, alpha=1.,
                                double_K=False, return_objective=False):
    """Compute gradient of dual weights over a multi-kernel ridge regression

    Parameters
    ----------
    Ks : array of shape (n_kernels, n_samples, n_samples)
        Input kernels for each feature space.
    Y : array of shape (n_samples, n_targets)
        Target data.
    dual_weights : array of shape (n_samples, n_targets)
        Kernel Ridge coefficients for each feature space.
    gammas : array of shape (n_kernels, ) or (n_kernels, n_targets)
        Kernel weights for each feature space. Should sum to 1 over kernels.
    alpha : float
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
    objective : float
        Objective function.
    """
    backend = get_current_backend()

    if gammas.ndim == 1:
        gammas = gammas[:, None]

    Ks_times_dualweights = backend.stack(
        [backend.matmul(K, dual_weights).T for K in Ks], 2)

    predictions = backend.matmul(Ks_times_dualweights,
                                 gammas.T[..., None])[..., 0].T
    residual = predictions - Y
    dampened_residual = residual + dual_weights * alpha

    if double_K:
        Ks_times_dampened_residual = backend.stack(
            [backend.matmul(K, dampened_residual).T for K in Ks], 2)
        dual_weight_gradient = backend.matmul(Ks_times_dampened_residual,
                                              gammas.T[..., None])[..., 0].T
    else:
        dual_weight_gradient = dampened_residual

    if return_objective:
        objective = backend.norm(residual, axis=0) ** 2
        objective += alpha * (predictions * dual_weights).sum(0)

        return dual_weight_gradient, objective
    else:
        return dual_weight_gradient


def solve_multi_kernel_ridge_gradient_descent(Ks, Y, gammas, alpha=1.,
                                              step_sizes=None,
                                              lipschitz_Ks=None,
                                              initial_dual_weights=None,
                                              max_iter=100, tol=1e-3,
                                              double_K=False, debug=False):
    """Solve the multi-Kernel ridge regression using gradient descent.

    Parameters
    ----------
    Ks : iterable with elements of shape (n_samples, n_samples)
        Input kernels for each feature space.
    Y : array of shape (n_samples, n_targets)
        Target data.
    gammas : array of shape (n_kernels, ) or (n_kernels, n_targets)
        Kernel weights for each feature space. Should sum to 1 over kernels.
    alpha : float
        Regularization parameter.
    step_sizes : float, or array of shape (n_targets), or None
        Step sizes.
        If None, computes a step size based on the Lipschitz constants.
    lipschitz_Ks : float, or array of shape (n_kernels), or None:
        Lipschitz constant for each feature space.
        Used only if `step_sizes` is None.
        If None, Lipschitz constants are estimated with power iteration on Ks.
    initial_dual_weights : array of shape (n_samples, n_targets)
        Initial kernel Ridge coefficients.
    max_iter : int
        Maximum number of gradient step.
    tol : float > 0 or None
        Tolerance for the stopping criterion.
    double_K : bool
        If True, multiply the gradient by the kernel to obtain the true
        gradients, which are less well conditionned.
    debug : bool
        If True, check some intermediate computations.

    Returns
    -------
    dual_weights : array of shape (n_samples, n_targets)
        Kernel Ridge coefficients.
    """
    backend = get_current_backend()

    if gammas.ndim == 1:
        gammas = gammas[:, None]

    if step_sizes is None:
        if lipschitz_Ks is None:
            lipschitz_Ks = compute_lipschitz_constants(Ks)
            if not double_K:
                lipschitz_Ks = backend.sqrt(lipschitz_Ks)

        total_lip = backend.matmul(lipschitz_Ks[None, :], gammas)[0] + alpha
        step_sizes = 1. / total_lip

        if debug:
            assert not backend.any(backend.isnan(step_sizes))

    if initial_dual_weights is None:
        dual_weights = backend.zeros_like(Y)
    else:
        dual_weights = initial_dual_weights.clone()

    #######################
    # Gradient descent loop
    for i in range(max_iter):
        grads = multi_kernel_ridge_gradient(Ks, Y, dual_weights, gammas,
                                            alpha=alpha, double_K=double_K)
        update = step_sizes * grads
        dual_weights -= update

        if tol is not None:
            max_update = backend.max(backend.abs(update / dual_weights))
            if max_update < tol:
                break

    return dual_weights
