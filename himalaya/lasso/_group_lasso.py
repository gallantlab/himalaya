import warnings

import numpy as np

from ..utils import compute_lipschitz_constants
from ..progress_bar import bar
from ..backend import get_backend


def solve_group_lasso(X, Y, groups=None, l21_reg=0.05, l1_reg=0.05,
                      max_iter=100, tol=1e-5, momentum=True, progress_bar=True,
                      debug=False):
    backend = get_backend()

    n_samples, n_features = X.shape
    n_samples, n_targets = Y.shape
    lipschitz = compute_lipschitz_constants(X[None], kernelize="XTX")[0]

    if groups is None:
        groups = backend.zeros((n_features))
    groups = backend.asarray(groups)[:]
    groups = [groups == u for u in backend.unique(groups) if u >= 0]

    coef = backend.zeros_like(X, shape=(n_features, n_targets))

    l1_reg = l1_reg * n_samples / lipschitz  # as in scikit-learn
    l21_reg = l21_reg * n_samples / lipschitz

    use_l1_reg = backend.any(l1_reg > 0)
    use_l21_reg = backend.any(l21_reg > 0)

    def loss(ww):
        error_sq = 0.5 * ((X @ ww - Y) ** 2).sum(0)

        if use_l1_reg:
            error_sq += l1_reg * backend.abs(ww).sum(0)

        if use_l21_reg:
            for group in groups:
                error_sq += l21_reg * backend.sqrt((ww[group] ** 2).sum(0))

        return error_sq

    def grad(ww, mask=None):
        if mask is None:
            return X.T @ (X @ ww - Y)
        else:
            return X.T @ (X @ ww - Y[:, mask])

    def prox(ww):
        if use_l1_reg:
            ww = _l1_prox(ww, l1_reg)
        if use_l21_reg:
            ww = _l21_prox(ww, l21_reg, groups)
        return ww

    coef = _fista(loss, grad, prox, step_size=1. / lipschitz, x0=coef,
                  max_iter=max_iter, momentum=momentum, tol=tol,
                  progress_bar=progress_bar, debug=debug)

    return coef


def _l1_prox(ww, reg):
    backend = get_backend()
    return backend.sign(ww) * backend.clip(backend.abs(ww) - reg, 0, None)


def _sqrt_l2_prox(ww, reg):
    """The proximal operator for reg*||w||_2 (not squared)."""
    backend = get_backend()

    norm_ww = backend.norm(ww, axis=0)
    mask = norm_ww == 0

    ww[:, mask] = 0
    ww[:, ~mask] = backend.clip(1 - reg / norm_ww[~mask], 0,
                                None)[None] * ww[:, ~mask]
    return ww


def _l21_prox(ww, reg, groups):
    backend = get_backend()

    ww = backend.copy(ww)
    for group in groups:
        ww[group, :] = _sqrt_l2_prox(ww[group, :], reg)

    return ww


###############################################################################
# fista algorithm


def _fista(f_loss, f_grad, f_prox, step_size, x0, max_iter, momentum=False,
           tol=1e-7, progress_bar=True, debug=False):
    """Proximal Gradient Descent (PGD) and Accelerated PDG.

    This reduces to ISTA and FISTA when the loss function is the l2 loss and
    the proximal operator is the soft-thresholding.

    Parameters
    ----------
    f_loss : callable
        ...
    f_grad : callable
        Gradient of the objective function.
    f_prox : callable
        Proximal operator.
    step_size : float
        Step size of each update.
    x0 : array
        Initial point of the optimization.
    max_iter : int
        Maximum number of iterations.
    momentum : bool
        If True, use FISTA instead of ISTA.
    tol : float
        Tolerance for the stopping criterion.
    progress_bar : bool
        ...
    debug : bool
        ...

    Returns
    -------
    x_hat : array
        The final point after optimization
    """
    backend = get_backend()

    if debug:
        losses = [f_loss(x0)]

    tk = 1.0
    x_hat = backend.copy(x0)

    # auxiliary variables
    x_hat_aux = backend.copy(x_hat)
    grad = backend.zeros_like(x_hat)
    diff = backend.zeros_like(x_hat)

    # not converged targets
    mask = backend.ones_like(x0, dtype=backend.bool, shape=(x0.shape[1]))

    for ii in bar(range(max_iter), 'fista', use_it=progress_bar):
        grad = f_grad(x_hat_aux, mask)
        x_hat_aux -= step_size * grad
        x_hat_aux = f_prox(x_hat_aux)

        diff = x_hat_aux - x_hat[:, mask]
        x_hat[:, mask] = x_hat_aux

        if momentum:
            tk_new = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
            x_hat_aux += (tk - 1) / tk_new * diff
            tk = tk_new

        if debug:
            losses.append(f_loss(x_hat))

        if tol is not None:
            criterion = (backend.norm(diff, axis=0) /
                         (backend.norm(x_hat[:, mask], axis=0) + 1e-16))
            just_converged = criterion <= tol

            if backend.any(just_converged):
                diff = diff[:, ~just_converged]
                grad = grad[:, ~just_converged]
                x_hat_aux = x_hat_aux[:, ~just_converged]
                mask[mask] = ~just_converged

                if backend.all(just_converged):
                    break
    else:
        warnings.warn("FISTA did not converge.", RuntimeWarning)

    if debug:
        return x_hat, backend.stack(losses)

    return x_hat
