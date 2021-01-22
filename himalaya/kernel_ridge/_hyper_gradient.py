import numbers

from sklearn.model_selection import check_cv

from ..backend import get_backend
from ..progress_bar import ProgressBar
from ..utils import compute_lipschitz_constants
from ..scoring import l2_neg_loss

from ._solvers import solve_weighted_kernel_ridge_conjugate_gradient
from ._solvers import solve_weighted_kernel_ridge_gradient_descent
from ._solvers import solve_weighted_kernel_ridge_neumann_series
from ._random_search import solve_multiple_kernel_ridge_random_search


def solve_multiple_kernel_ridge_hyper_gradient(
        Ks, Y, score_func=l2_neg_loss, cv=5, fit_intercept=False,
        return_weights=None, Xs=None, initial_deltas=0, max_iter=10, tol=1e-2,
        max_iter_inner_dual=1, max_iter_inner_hyper=1, cg_tol=1e-3,
        n_targets_batch=None, hyper_gradient_method="conjugate_gradient",
        kernel_ridge_method="gradient_descent", random_state=None,
        progress_bar=True, Y_in_cpu=False):
    """Solve bilinear kernel ridge regression with cross-validation.

    The hyper-parameters deltas correspond to::

        log(kernel_weights / ridge_regularization)

    Parameters
    ----------
    Ks : array of shape (n_kernels, n_samples, n_samples)
        Training kernel for each feature space.
    Y : array of shape (n_samples, n_targets)
        Training target data.
    score_func : callable
        Function used to compute the score of predictions.
    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.
    fit_intercept : boolean
        Whether to fit an intercept. If False, Ks should be centered
        (see KernelCenterer), and Y must be zero-mean over samples.
        Only available if return_weights == 'dual'.
    return_weights : None, 'primal', or 'dual'
        Whether to refit on the entire dataset and return the weights.
    Xs : array of shape (n_kernels, n_samples, n_features) or None
        Necessary if return_weights == 'primal'.
    initial_deltas : str, float, array of shape (n_kernels, n_targets)
        Initial log kernel weights for each target.
        If a float, initialize the deltas with this value.
        If a str, initialize the deltas with different strategies:
        - 'ridgecv' : fit a RidgeCV model over the average kernel.
    max_iter : int
        Maximum number of iteration for the outer loop.
    tol : float > 0, or None
        Tolerance for the stopping criterion.
    max_iter_inner_dual : int
        Maximum number of iterations for the dual weights conjugate gradient.
    max_iter_inner_hyper :
        Maximum number of iterations for the deltas gradient descent.
    cg_tol : float, or array of shape (max_iter)
        Tolerance for the conjugate gradients.
    n_targets_batch : int or None
        Size of the batch for computing predictions. Used for memory reasons.
        If None, uses all n_targets at once.
    hyper_gradient_method : str, "conjugate_gradient", "neumann", "direct"
        Method to compute the hypergradient.
    kernel_ridge_method : str, "conjugate_gradient" or "gradient_descent"
        Algorithm used for the inner step.
    random_state : int, or None
        Random generator seed. Use an int for deterministic search.
    progress_bar : bool
        If True, display a progress bar over batches and iterations.
    Y_in_cpu : bool
        If True, keep the target values ``Y`` in CPU memory (slower).

    Returns
    -------
    deltas : array of shape (n_kernels, n_targets)
        Best log kernel weights for each target.
    refit_weights : array or None
        Refit regression weights on the entire dataset, using selected best
        hyperparameters. Refit weights will always be on CPU memory.
        If return_weights == 'primal', shape is (n_features, n_targets),
        if return_weights == 'dual', shape is (n_samples, n_targets),
        else, None.
    cv_scores : array of shape (max_iter * max_iter_inner_hyper, n_targets)
        Cross-validation scores per iteration, averaged over splits.
        Cross-validation scores will always be on CPU memory.
    intercept : array of shape (n_targets,)
        Intercept. Only returned when fit_intercept is True.
    """
    backend = get_backend()

    n_samples, n_targets = Y.shape
    if n_targets_batch is None:
        n_targets_batch = n_targets

    Ks = backend.asarray(Ks)
    dtype = Ks.dtype
    device = getattr(Ks, "device", None)
    Y = backend.asarray(Y, dtype=dtype, device="cpu" if Y_in_cpu else device)

    if fit_intercept:
        raise NotImplementedError('Coming soon.')

    cv = check_cv(cv)
    n_splits = cv.get_n_splits()
    for train, val in cv.split(Y):
        if len(val) == 0 or len(train) == 0:
            raise ValueError("Empty train or validation set. "
                             "Check that `cv` is correctly defined.")

    deltas = _init_multiple_kernel_ridge(Ks, Y, initial_deltas, cv,
                                         n_targets_batch=n_targets_batch,
                                         Y_in_cpu=Y_in_cpu)

    if return_weights == 'primal':
        if Xs is None:
            raise ValueError("Xs is needed to compute the primal weights.")
        n_features = sum(X.shape[1] for X in Xs)
        refit_weights = backend.zeros_like(Ks, shape=(n_features, n_targets),
                                           device="cpu")

    elif return_weights == 'dual':
        refit_weights = backend.zeros_like(Ks, shape=(n_samples, n_targets),
                                           device="cpu")
    elif return_weights is None:
        refit_weights = None
    else:
        raise ValueError("Unknown parameter return_weights=%r." %
                         (return_weights, ))

    lipschitz_constants = None
    name = "hypergradient_" + hyper_gradient_method
    if kernel_ridge_method == "conjugate_gradient":
        inner_function = solve_weighted_kernel_ridge_conjugate_gradient
    elif kernel_ridge_method == "gradient_descent":
        inner_function = solve_weighted_kernel_ridge_gradient_descent

        # precompute the lipschitz constants
        lipschitz_constants = []
        for train, _ in cv.split(Y):
            train = backend.to_gpu(train, device=device)
            Ks_train = Ks[:, train[:, None], train]
            lipschitz_constants.append(
                compute_lipschitz_constants(Ks_train,
                                            random_state=random_state))
    else:
        raise ValueError("Unknown parameter kernel_ridge_method=%r." %
                         (kernel_ridge_method, ))

    if isinstance(cg_tol, (int, float)):
        cg_tol = backend.full_like(Y, shape=(max_iter, ), fill_value=cg_tol)

    alpha = 1.0

    cv_scores = backend.zeros_like(
        Ks, shape=(max_iter * max_iter_inner_hyper, n_targets), device='cpu')

    batch_iterates = range(0, n_targets, n_targets_batch)
    bar = None
    if progress_bar:
        bar = ProgressBar(title=name, max_value=len(batch_iterates) * max_iter)
    for bb, start in enumerate(batch_iterates):
        batch = slice(start, start + n_targets_batch)
        Y_batch = backend.to_gpu(Y[:, batch], device=device)

        previous_solutions = [None] * n_splits
        step_sizes = [None] * n_splits
        dual_weights_cv = [None] * n_splits

        for ii in range(max_iter):
            if progress_bar:
                bar.update(bb * max_iter + ii)

            ##########################
            # updates the dual weights

            # First pass needs more iterations to have something reasonable.
            # We also use conjugate gradient as it converges faster.
            if ii == 0:
                max_iter_inner_dual_ = 50
                cg_tol_ = min(1e-2, cg_tol[ii])
                inner_function_ = solve_weighted_kernel_ridge_conjugate_gradient  # noqa
            else:
                max_iter_inner_dual_ = max_iter_inner_dual
                cg_tol_ = cg_tol[ii]
                inner_function_ = inner_function

            for kk, (train, val) in enumerate(cv.split(Y)):
                train = backend.to_gpu(train, device=device)
                Ks_train = Ks[:, train[:, None], train]
                Y_train = Y_batch[train]

                if kernel_ridge_method == "gradient_descent" and ii != 0:
                    kwargs = dict(lipschitz_Ks=lipschitz_constants[kk])
                else:
                    kwargs = dict()

                dual_weights_cv[kk] = inner_function_(
                    Ks_train, Y_train, deltas[:, batch],
                    initial_dual_weights=dual_weights_cv[kk], alpha=alpha,
                    max_iter=max_iter_inner_dual_, tol=cg_tol_, **kwargs)

            ###################
            # update the deltas
            deltas_old = backend.copy(deltas[:, batch])
            for jj in range(max_iter_inner_hyper):

                gradients = backend.zeros_like(deltas[:, batch])
                scores = backend.zeros_like(
                    Ks, shape=(n_splits, deltas[:, batch].shape[1]))
                for kk, (train, val) in enumerate(cv.split(Y)):
                    val = backend.to_gpu(val, device=device)
                    train = backend.to_gpu(train, device=device)

                    Ks_val = Ks[:, val[:, None], train]
                    Ks_train = Ks[:, train[:, None], train]
                    Y_val = Y_batch[val]

                    (gradients_kk, step_sizes[kk], predictions,
                     previous_solutions[kk]) = _compute_delta_gradient(
                         Ks_val=Ks_val, Y_val=Y_val, deltas=deltas[:, batch],
                         dual_weights=dual_weights_cv[kk], Ks_train=Ks_train,
                         tol=cg_tol[ii], random_state=random_state,
                         hyper_gradient_method=hyper_gradient_method,
                         previous_solution=previous_solutions[kk])

                    gradients += gradients_kk * Y_val.shape[0] / n_samples

                    scores[kk] = score_func(Y_val, predictions)

                it = ii * max_iter_inner_hyper + jj
                cv_scores[it, batch] = backend.to_cpu(scores.mean(0))

                # update deltas, using the minimum step size over splits
                step_size = backend.min(backend.stack(step_sizes), axis=0)
                deltas[:, batch] -= gradients * step_size[None, :]
                assert not backend.any(
                    backend.isinf(backend.exp(deltas[:, batch])))

            ####################
            # stopping criterion
            if tol is not None:
                if backend.max(
                        backend.abs(deltas_old - deltas[:, batch])) < tol:
                    cv_scores = cv_scores[:(ii + 1) * max_iter_inner_hyper]
                    break

        ##########################################
        # refit dual weights on the entire dataset
        if return_weights in ["primal", "dual"]:
            dual_weights = solve_weighted_kernel_ridge_conjugate_gradient(
                Ks, Y_batch, deltas[:, batch], initial_dual_weights=None,
                alpha=alpha, max_iter=100, tol=1e-4)
            if return_weights == 'primal':
                # multiply by g and not np.sqrt(g), as we then want to use
                # the primal weights on the unscaled features Xs, and not
                # on the scaled features (np.sqrt(g) * Xs)
                X = None
                for tt in range(refit_weights[:, batch].shape[1]):
                    X = backend.concatenate([
                        t * g for t, g in zip(
                            Xs, backend.exp(deltas[:, batch][:, tt]))
                    ], 1)
                    refit_weights[:, batch][:, tt] = backend.to_cpu(
                        backend.matmul(X.T, dual_weights[:, tt]))
                del X

            elif return_weights == 'dual':
                refit_weights[:, batch] = backend.to_cpu(dual_weights)

            del dual_weights

    if progress_bar:
        bar.update(bar.max_value)

    results = [deltas, refit_weights, cv_scores]
    if fit_intercept:
        results.append(intercept)
    return results


#: Dictionary with all multiple kernel ridge solvers
MULTIPLE_KERNEL_RIDGE_SOLVERS = {
    "random_search": solve_multiple_kernel_ridge_random_search,
    "hyper_gradient": solve_multiple_kernel_ridge_hyper_gradient,
}


def _init_multiple_kernel_ridge(Ks, Y, initial_deltas, cv, **ridgecv_kwargs):
    """Initialize deltas, i.e. log kernel weights.

    Parameters
    ----------
    Ks : array of shape (n_kernels, n_samples, n_samples)
        Training kernel for each feature space.
    Y : array of shape (n_samples, n_targets)
        Training target data.
    initial_deltas : str, float, array of shape (n_kernels, n_targets)
        Initial log kernel weights for each target.
        If a float, initialize the deltas with this value.
        If a str, initialize the deltas with different strategies:
        - 'ridgecv' : fit a RidgeCV model over the average kernel
    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.
    ridgecv_kwargs : dict
        Other parameters if initial_deltas == 'ridgecv'.

    Returns
    -------
    deltas : array of shape (n_kernels, n_targets)
        Initial deltas.
    """
    backend = get_backend()

    n_kernels = Ks.shape[0]
    n_targets = Y.shape[1]

    if initial_deltas is None:
        initial_deltas = 0

    if ((isinstance(initial_deltas, str) and (initial_deltas == 'ridgecv'))):
        alphas = backend.logspace(-10, 20, 30)
        gammas = backend.full_like(Y, shape=n_kernels,
                                   fill_value=1. / n_kernels)[None]
        deltas, _, _ = solve_multiple_kernel_ridge_random_search(
            Ks, Y, n_iter=gammas, alphas=alphas, cv=cv, n_alphas_batch=5,
            return_weights=None, progress_bar=False, **ridgecv_kwargs)

    elif isinstance(initial_deltas, numbers.Number):
        deltas = backend.full_like(Ks, shape=(n_kernels, n_targets),
                                   fill_value=initial_deltas)

    else:
        deltas = backend.copy(backend.asarray_like(initial_deltas, ref=Ks))

    Ks, deltas = backend.check_arrays(Ks, deltas)
    return deltas


def _compute_delta_loss(Ks_val, Y_val, deltas, dual_weights):
    """Compute the validation loss.

    Parameters
    ----------
    Ks_val : array of shape (n_kernels, n_samples_val, n_samples_train)
        Cross-kernels between training set and validation set.
    Y_val : array of shape (n_samples_val, n_targets)
        Target data on the validation set.
    deltas : array of shape (n_kernels, n_targets)
        Log of the kernel weights.
    dual_weights : array of shape (n_samples_train, n_targets)
        Kernel ridge weights.

    Returns
    -------
    loss : array of shape (n_targets)
        L2 validation loss.
    """
    backend = get_backend()
    exp_delta = backend.exp(deltas)
    chi_val = backend.matmul(Ks_val, dual_weights)
    exp_delta_chi_val = exp_delta[:, None, :] * chi_val
    predictions = exp_delta_chi_val.sum(0)
    residuals = predictions - Y_val
    loss = 0.5 * backend.norm(residuals, axis=0) ** 2
    return loss


def _compute_delta_gradient(Ks_val, Y_val, deltas, dual_weights, Ks_train=None,
                            tol=None, previous_solution=None,
                            hyper_gradient_method='conjugate_gradient',
                            random_state=None):
    """Compute the gradient over deltas on the validation dataset.

    Parameters
    ----------
    Ks_val : array of shape (n_kernels, n_samples_val, n_samples_train)
        Cross-kernels between training set and validation set.
    Y_val : array of shape (n_samples_val, n_targets)
        Target data on the validation set.
    deltas : array of shape (n_kernels, n_targets)
        Log of the kernel weights.
    dual_weights : array of shape (n_samples_train, n_targets)
        Kernel ridge weights.
    Ks_train : array of shape (n_kernels, n_samples_train, n_samples_train)
        Kernels on the training set.
        Not required if hyper_gradient_method = "direct".
    tol : float
        Tolerance for the conjugate method.
        Required if hyper_gradient_method = "conjugate_gradient".
    previous_solution
        Speed up hyper_gradient_method = "conjugate_gradient" by warm starting
        with the previous solution.
    hyper_gradient_method : str, in {"conjugate_gradient", "neumann", "direct"}
        Method used to compute the hyper gradient.
    random_state : int, or None
        Random generator seed. Use an int for deterministic search.

    Returns
    -------
    gradient : array of shape (n_kernels, n_targets)
        Gradient over deltas.
    step_size : array of shape (n_targets)
        Step size computed based on the direct gradient's Lipschitz constant.
    predictions : array of shape (n_samples_val, n_targets)
        Predictions on the validation set.
    solution : array of shape (n_samples_train, n_targets) or None
        Solution of the inverse Hessian in the indirect gradient.
    """
    backend = get_backend()

    # prepare quantities
    exp_delta = backend.exp(deltas)
    chi_val = backend.matmul(Ks_val, dual_weights)
    exp_delta_chi_val = exp_delta[:, None, :] * chi_val
    predictions = exp_delta_chi_val.sum(0)
    assert predictions.shape == Y_val.shape

    # direct gradient
    residuals = predictions - Y_val
    direct_gradient = (residuals[None] * exp_delta_chi_val).sum(1)
    assert direct_gradient.shape == deltas.shape

    # estimate a step size
    XTXs = _compute_deltas_hessian(exp_delta_chi_val, residuals)
    # (these lipschitz constants only correspond to the direct gradient)
    lipschitz_1 = compute_lipschitz_constants(XTXs, "X",
                                              random_state=random_state)
    step_size = 1. / (lipschitz_1 + 1e-15)

    if hyper_gradient_method == 'direct':
        gradient = direct_gradient
        solution = None
    else:
        # compute the indirect gradient by inverting the Hessian

        # compute nabla_g_1
        tmp = backend.matmul(backend.transpose(Ks_val, (2, 0, 1)), residuals)
        tmp = backend.transpose(tmp, (2, 0, 1))
        nabla_g_1 = backend.matmul(
            tmp,
            backend.transpose(exp_delta, (1, 0))[:, :, None])
        nabla_g_1 = backend.transpose(nabla_g_1[:, :, 0], (1, 0))
        assert nabla_g_1.shape == dual_weights.shape

        # solve linear system (sum_i gamma[i]*K[i] + 1) @ X = nabla_g_1
        alpha = 1
        assert Ks_train is not None
        if hyper_gradient_method == 'conjugate_gradient':
            assert tol is not None
            solution = solve_weighted_kernel_ridge_conjugate_gradient(
                Ks=Ks_train, Y=nabla_g_1, deltas=deltas,
                initial_dual_weights=previous_solution, max_iter=100, tol=tol,
                alpha=alpha)
        elif hyper_gradient_method == 'neumann':
            solution = solve_weighted_kernel_ridge_neumann_series(
                Ks=Ks_train, Y=nabla_g_1, deltas=deltas, max_iter=5,
                factor=0.00001, alpha=alpha)
        else:
            raise ValueError("Unknown parameter hyper_gradient_method=%r." %
                             (hyper_gradient_method, ))

        # finish the indirect gradient
        chi_train = backend.matmul(Ks_train, dual_weights)
        exp_delta_chi_train = exp_delta[:, None, :] * chi_train
        indirect_gradient = (exp_delta_chi_train * solution[None, :, :]).sum(1)
        assert indirect_gradient.shape == deltas.shape

        gradient = direct_gradient - indirect_gradient
        assert not backend.any(backend.isinf(gradient))

    return gradient, step_size, predictions, solution


def _compute_deltas_hessian(exp_delta_chi, residuals):
    """Compute the hessian of the direct gradient.

    The direct gradient correponds to a linear problem:
        argmin_delta ||exp(delta) @ chi - Y||^2
    where chi = Ks @ dual_weights.

    The Hessian is not just `chi.T @ chi` because of the exponential
    parametrization of deltas.

    Parameters
    ----------
    exp_delta_chi : array of shape (n_kernels, n_samples, n_targets)
        Precomputation of exp(delta) * (Ks @ dual_weights).
    residuals : array of shape (n_samples, n_targets)
        Difference between prediction and target on the validation split.

    Returns
    -------
    XTXs : array of shape (n_targets, n_kernels, n_kernels)
        Hessian of the direct gradient.
    """
    backend = get_backend()

    XTXs = backend.matmul(backend.transpose(exp_delta_chi, (2, 0, 1)),
                          backend.transpose(exp_delta_chi, (2, 1, 0)))
    XTbs = backend.matmul(backend.transpose(exp_delta_chi, (2, 0, 1)),
                          backend.transpose(residuals,
                                            (1, 0))[:, :, None])[:, :, 0]
    diagonal_view = backend.diagonal_view(XTXs, axis1=1, axis2=2)
    diagonal_view += XTbs
    return XTXs
