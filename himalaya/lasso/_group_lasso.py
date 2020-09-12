import warnings
import itertools

import numpy as np
from sklearn.model_selection import check_cv

from ..utils import compute_lipschitz_constants
from ..progress_bar import bar, ProgressBar
from ..backend import get_backend
from ..scoring import r2_score


def solve_sparse_group_lasso_cv(X, Y, groups=None, l21_regs=[0.05],
                                l1_regs=[0.05], cv=5, max_iter=300, tol=1e-4,
                                momentum=True, n_targets_batch=None,
                                progress_bar=True):
    """Solves the sparse group Lasso, selecting hyperparameters over
    cross-validation.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Input features.
    Y : array of shape (n_samples, n_targets)
        Target data.
    groups : array of shape (n_features, ) or None
        Encoding of the group of each feature. If None, all features are
        gathered in one group, and the problem is equivalent to the Lasso.
    l21_regs : array of shape (n_l21_regs, )
        All the Group Lasso regularization parameter tested.
    l1_regs : array of shape (n_l1_regs, )
        All the Lasso regularization parameter tested.
    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.
    max_iter : int > 0
        Maximum number of iterations.
    tol : float > 0
        Tolerance of the stopping criterion.
    momentum : bool
        If True, use acceleration in the projected gradient descent.
    n_targets_batch : int or None
        Size of the batch for computing predictions. Used for memory reasons.
        If None, uses all n_targets at once.
    progress_bar : bool
        If True, display a progress bar over batches and iterations.

    Returns
    -------
    coef : array of shape (n_features, n_targets)
        Coefficient of the linear model. Always on CPU.
    best_l21_reg : array of shape (n_targets, )
        Best hyperparameter per target.
    best_l1_reg : array of shape (n_targets, )
        Best hyperparameter per target.
    all_cv_scores : array of shape (n_l21_regs * n_l1_regs, n_targets)
        Cross-validation scores of all tested hyperparameters.
    """
    backend = get_backend()

    cv = check_cv(cv)
    n_splits = cv.get_n_splits()

    X, Y, l21_regs, l1_regs = backend.check_arrays(X, Y, l21_regs, l1_regs)
    n_samples, n_features = X.shape
    n_samples, n_targets = Y.shape
    if n_targets_batch is None:
        n_targets_batch = n_targets

    l21_regs = backend.asarray_like(backend.atleast_1d(l21_regs), ref=Y)
    l1_regs = backend.asarray_like(backend.atleast_1d(l1_regs), ref=Y)

    # sort in decreasing order, since it is faster with warm starting
    decreasing_l21_regs = backend.flip(backend.sort(l21_regs), 0)
    if not backend.all(decreasing_l21_regs == l21_regs):
        l21_regs = decreasing_l21_regs
        warnings.warn("l21_regs has been reordered in decreasing order.")

    # sort in decreasing order, since it is faster with warm starting
    decreasing_l1_regs = backend.flip(backend.sort(l1_regs), 0)
    if not backend.all(decreasing_l1_regs == l1_regs):
        l1_regs = decreasing_l1_regs
        warnings.warn("l1_regs has been reordered in decreasing order.")

    n_l21_regs = l21_regs.shape[0]
    n_l1_regs = l1_regs.shape[0]
    n_batches = len(range(0, n_targets, n_targets_batch))

    if progress_bar:
        progress_bar = ProgressBar(
            f"grid search cv over {n_l21_regs * n_l1_regs} parameters",
            max_value=n_l21_regs * n_l1_regs * n_splits * n_batches)

    coef = None
    all_cv_scores = backend.zeros_like(
        X, shape=(n_l21_regs * n_l1_regs, n_targets))
    best_l21_reg = backend.zeros_like(X, shape=(n_targets, ))
    best_l1_reg = backend.zeros_like(X, shape=(n_targets, ))

    for kk, (train, val) in enumerate(cv.split(Y)):
        if hasattr(Y, "device"):
            val = backend.asarray(val, device=Y.device)
            train = backend.asarray(train, device=Y.device)

        lipschitz_train = compute_lipschitz_constants(X[train][None],
                                                      kernelize="XTX")[0]

        for start in range(0, n_targets, n_targets_batch):
            batch = slice(start, start + n_targets_batch)

            # reset coef, to avoid leaking info from split to split
            coef = None
            for ii, (l21_reg,
                     l1_reg) in enumerate(itertools.product(l21_regs,
                                                            l1_regs)):

                coef = solve_sparse_group_lasso(
                    X[train], Y[train][:, batch], groups=groups,
                    l21_reg=l21_reg, l1_reg=l1_reg, max_iter=max_iter, tol=tol,
                    momentum=momentum, initial_coef=coef,
                    lipschitz=lipschitz_train, n_targets_batch=n_targets_batch,
                    progress_bar=False)

                Y_val_pred = X[val] @ coef
                scores = r2_score(Y[val][:, batch], Y_val_pred)

                all_cv_scores[ii, batch] += scores

                if progress_bar:
                    progress_bar.update_with_increment_value(1)

    # select best hyperparameter configuration
    all_cv_scores /= n_splits
    argmax = backend.argmax(all_cv_scores, 0)
    config = backend.asarray(
        list(
            itertools.product(backend.to_numpy(l21_regs),
                              backend.to_numpy(l1_regs))))
    best_config = config[argmax]
    best_l21_reg = best_config[:, 0]
    best_l1_reg = best_config[:, 1]

    # refit
    coef_cpu = backend.zeros_like(X, shape=(n_features, n_targets),
                                  device="cpu")
    for start in range(0, n_targets, n_targets_batch):
        batch = slice(start, start + n_targets_batch)
        coef = solve_sparse_group_lasso(
            X, Y[:, batch], groups=groups, l21_reg=best_l21_reg[batch],
            l1_reg=best_l1_reg[batch], max_iter=max_iter, tol=tol,
            momentum=momentum, initial_coef=None, lipschitz=None,
            n_targets_batch=None, progress_bar=False)
        coef_cpu[:, batch] = backend.to_cpu(coef)
    return coef_cpu, best_l21_reg, best_l1_reg, all_cv_scores


def solve_sparse_group_lasso(X, Y, groups=None, l21_reg=0.05, l1_reg=0.05,
                             max_iter=300, tol=1e-4, momentum=True,
                             initial_coef=None, lipschitz=None,
                             n_targets_batch=None, progress_bar=True,
                             debug=False):
    """Solves the sparse group Lasso.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Input features.
    Y : array of shape (n_samples, n_targets)
        Target data.
    groups : array of shape (n_features, ) or None
        Encoding of the group of each feature. If None, all features are
        gathered in one group, and the problem is equivalent to the Lasso.
    l21_reg : float, or array of shape (n_targets, )
        Regularization parameter for the group Lasso.
    l1_reg : float, or array of shape (n_targets, )
        Regularization parameter for the Lasso.
    max_iter : int > 0
        Maximum number of iterations.
    tol : float > 0
        Tolerance of the stopping criterion.
    momentum : bool
        If True, use acceleration in the projected gradient descent.
    initial_coef : None, or array of shape (n_features, n_targets)
        Initial value for the projected gradient descent.
    lipschitz : float or None
        Lipschitz constant of the gradient. If None, it will be estimated from
        X using power iterations.
    n_targets_batch : int or None
        Size of the batch for computing predictions. Used for memory reasons.
        If None, uses all n_targets at once.
    progress_bar : bool
        If True, display a progress bar over batches and iterations.
    debug : bool
        If True, compute the objective function at each iteration, and return
        the list of results (of the last batch) along with the final
        coefficients.

    Returns
    -------
    coef : array of shape (n_features, n_targets)
        Coefficient of the linear model.
    """
    backend = get_backend()

    n_samples, n_features = X.shape
    n_samples, n_targets = Y.shape
    if n_targets_batch is None:
        n_targets_batch = n_targets

    X, Y = backend.check_arrays(X, Y)

    if lipschitz is None:
        lipschitz = compute_lipschitz_constants(X[None], kernelize="XTX")[0]
    else:
        lipschitz = backend.asarray_like(lipschitz, ref=X)

    if groups is None:
        groups = backend.zeros((n_features))
    groups = backend.asarray(groups)[:]
    groups = [groups == u for u in backend.unique(groups) if u >= 0]

    if initial_coef is None:
        coef = backend.zeros_like(X, shape=(n_features, n_targets))
    else:
        coef = backend.asarray_like(initial_coef, ref=X)

    l1_reg = l1_reg * n_samples  # as in scikit-learn
    l21_reg = l21_reg * n_samples

    l1_reg = backend.asarray_like(backend.atleast_1d(l1_reg), ref=Y)
    if l1_reg.shape[0] == 1:
        l1_reg = backend.ones_like(Y, shape=(n_targets, )) * l1_reg[0]
    l21_reg = backend.asarray_like(backend.atleast_1d(l21_reg), ref=Y)
    if l21_reg.shape[0] == 1:
        l21_reg = backend.ones_like(Y, shape=(n_targets, )) * l21_reg[0]

    use_l1_reg = any(l1_reg > 0)
    use_l21_reg = any(l21_reg > 0)

    use_it = progress_bar and n_targets > n_targets_batch
    for bb, start in enumerate(
            bar(range(0, n_targets, n_targets_batch), title="Group Lasso",
                use_it=use_it)):
        batch = slice(start, start + n_targets_batch)

        def loss(ww):
            """Objective function"""
            error_sq = 0.5 * ((X @ ww - Y[:, batch]) ** 2).sum(0)

            if use_l1_reg:
                error_sq += l1_reg[batch] * backend.abs(ww).sum(0)

            if use_l21_reg:
                for group in groups:
                    error_sq += l21_reg[batch] * backend.sqrt(
                        (ww[group] ** 2).sum(0))

            return error_sq

        def grad(ww, mask=slice(0, None)):
            """Gradient of the objective function"""
            return X.T @ (X @ ww - Y[:, batch][:, mask])

        def prox(ww, mask=slice(0, None)):
            """Proximal operator for the projected gradient descent"""
            if use_l1_reg:
                ww = _l1_prox(ww, l1_reg[batch][mask] / lipschitz)
            if use_l21_reg:
                ww = _l21_prox(ww, l21_reg[batch][mask] / lipschitz, groups)
            return ww

        tmp = _proximal_gradient_descent(
            loss, grad, prox, step_size=1. / lipschitz, x0=coef[:, batch],
            max_iter=max_iter, momentum=momentum, tol=tol,
            progress_bar=progress_bar and n_targets <= n_targets_batch,
            debug=debug)
        if debug:
            tmp, losses = tmp
        coef[:, batch] = tmp

    if debug:
        return coef, losses
    else:
        return coef


def _l1_prox(ww, reg):
    """The proximal operator for reg * ||ww||_1."""
    backend = get_backend()
    return backend.sign(ww) * backend.clip(backend.abs(ww) - reg, 0, None)


def _sqrt_l2_prox(ww, reg):
    """The proximal operator for reg * ||ww||_2 (not squared)."""
    backend = get_backend()

    norm_ww = backend.norm(ww, axis=0)
    mask = norm_ww == 0

    ww[:, mask] = 0
    ww[:, ~mask] = backend.clip(1 - reg[~mask] / norm_ww[~mask], 0,
                                None)[None] * ww[:, ~mask]
    return ww


def _l21_prox(ww, reg, groups):
    """The proximal operator for reg * ||ww||_21 for all groups"""
    backend = get_backend()

    ww = backend.copy(ww)
    for group in groups:
        ww[group, :] = _sqrt_l2_prox(ww[group, :], reg)

    return ww


###############################################################################
# fista algorithm


def _proximal_gradient_descent(f_loss, f_grad, f_prox, step_size, x0, max_iter,
                               momentum=True, tol=1e-7, progress_bar=True,
                               debug=False):
    """Proximal Gradient Descent (PGD) and Accelerated PDG.

    This reduces to ISTA and FISTA when the loss function is the l2 loss and
    the proximal operator is the soft-thresholding.

    Parameters
    ----------
    f_loss : callable
        Objective function. Only used when debug = True.
    f_grad : callable
        Gradient of the objective function.
    f_prox : callable
        Proximal operator.
    step_size : float
        Step size of each update.
    x0 : array of shape (n_features, n_targets)
        Initial point of the optimization.
    max_iter : int
        Maximum number of iterations.
    momentum : bool
        If True, use FISTA instead of ISTA.
    tol : float or None
        Tolerance for the stopping criterion.
    progress_bar : bool
        If True, display a progress bar over iterations.
    debug : bool
        If True, compute the objective function at each iteration, and return
        the list of results along with the final point x_hat.

    Returns
    -------
    x_hat : array of shape (n_features, n_targets)
        The final point after optimization.
    losses : array of shape (n_iter_ + 1, n_targets)
        Returned only if debug = True. Objective function at each iteration.
    """
    backend = get_backend()

    if debug:
        losses = [f_loss(x0)]

    t_new = 1.0
    x_hat = backend.copy(x0)

    # auxiliary variables
    x_old = backend.copy(x_hat)
    x_aux = backend.copy(x_hat)
    diff = backend.zeros_like(x_hat)

    # not converged targets
    mask = backend.ones_like(x0, dtype=backend.bool, shape=(x0.shape[1]))

    if progress_bar:
        progress_bar = ProgressBar(title='fista', max_value=max_iter)
    for ii in range(max_iter):
        # apply gradient and prox, from x_aux
        x_old[:] = x_hat[:, mask]
        x_hat[:, mask] = x_aux - step_size * f_grad(x_aux, mask)
        x_hat[:, mask] = f_prox(x_hat[:, mask], mask)
        diff = x_hat[:, mask] - x_old

        # apply momentum, from x_hat
        x_aux[:] = x_hat[:, mask]
        if momentum:
            t_old = t_new
            t_new = (1 + np.sqrt(1 + 4 * t_old * t_old)) / 2
            x_aux += (t_old - 1.) / t_new * diff

        if debug:
            losses.append(f_loss(x_hat))

        if tol is not None:
            criterion = (backend.norm(diff, axis=0) /
                         (backend.norm(x_hat[:, mask], axis=0) + 1e-16))
            just_converged = criterion <= tol

            if backend.any(just_converged):
                x_aux = x_aux[:, ~just_converged]
                x_old = x_old[:, ~just_converged]
                mask[mask] = ~just_converged

                if backend.all(just_converged):
                    break

        if progress_bar:
            progress_bar.update_with_increment_value(1)
    else:
        warnings.warn("FISTA did not converge.", RuntimeWarning)

    if progress_bar:
        progress_bar.close()

    if debug:
        return x_hat, backend.stack(losses)

    return x_hat
