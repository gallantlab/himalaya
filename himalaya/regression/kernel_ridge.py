from ..backend import get_current_backend


def multi_kernel_ridge_gradient(Ks, Ys, dual_weights, gammas, alpha=1.,
                                double_K=False, return_objective=False):
    """Compute gradient of dual weights over a multi-kernel ridge regression

    Parameters
    ----------
    Ks : array of shape (n_kernels, n_samples, n_samples)
        Input kernels for each feature space.
    Ys : array of shape (n_samples, n_targets)
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

    Ks_times_dualweights = backend.stack(
        [backend.matmul(K, dual_weights).T for K in Ks], 2)

    Ktilde_times_dual_weights = backend.matmul(Ks_times_dualweights,
                                               gammas.T[..., None])[..., 0].T
    residual = Ktilde_times_dual_weights - Ys
    dampened_residual = residual + dual_weights * alpha

    if double_K:
        Ks_times_dampened_residual = backend.stack(
            [backend.matmul(K, dampened_residual).T for K in Ks], 2)
        dual_weight_gradient = backend.matmul(Ks_times_dampened_residual,
                                              gammas.T[..., None])[..., 0].T
    else:
        dual_weight_gradient = dampened_residual

    if return_objective:

        objective = (
            backend.norm(residual, axis=0) ** 2 +
            alpha * backend.matmul(Ktilde_times_dual_weights.T[:, None],
                                   dual_weights.T[:, :, None])[:, 0, 0])

        return dual_weight_gradient, objective
    else:
        return dual_weight_gradient
