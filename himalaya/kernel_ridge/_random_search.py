import warnings
import numbers

from sklearn.model_selection import check_cv

from ..backend import get_backend
from ..progress_bar import bar
from ..scoring import l2_neg_loss
from ..validation import check_random_state
from ._solvers import _helper_intercept
from ._kernels import KernelCenterer


def solve_multiple_kernel_ridge_random_search(
        Ks, Y, n_iter=100, concentration=[0.1, 1.0], alphas=1.0,
        score_func=l2_neg_loss, cv=5, fit_intercept=False, return_weights=None,
        Xs=None, local_alpha=True, jitter_alphas=False, random_state=None,
        n_targets_batch=None, n_targets_batch_refit=None, n_alphas_batch=None,
        progress_bar=True, Ks_in_cpu=False, conservative=False, Y_in_cpu=False,
        diagonalize_method="eigh", return_alphas=False):
    """Solve multiple kernel ridge regression using random search.

    Parameters
    ----------
    Ks : array of shape (n_kernels, n_samples, n_samples)
        Input kernels.
    Y : array of shape (n_samples, n_targets)
        Target data.
    n_iter : int, or array of shape (n_iter, n_kernels)
        Number of kernel weights combination to search.
        If an array is given, the solver uses it as the list of kernel weights
        to try, instead of sampling from a Dirichlet distribution.
    concentration : float, or list of float
        Concentration parameters of the Dirichlet distribution.
        If a list, iteratively cycle through the list.
        Not used if n_iter is an array.
    alphas : float or array of shape (n_alphas, )
        Range of ridge regularization parameter.
    score_func : callable
        Function used to compute the score of predictions versus Y.
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
    local_alpha : bool
        If True, alphas are selected per target, else shared over all targets.
    jitter_alphas : bool
        If True, alphas range is slightly jittered for each gamma.
    random_state : int, or None
        Random generator seed. Use an int for deterministic search.
    n_targets_batch : int or None
        Size of the batch for over targets during cross-validation.
        Used for memory reasons. If None, uses all n_targets at once.
    n_targets_batch_refit : int or None
        Size of the batch for over targets during refit.
        Used for memory reasons. If None, uses all n_targets at once.
    n_alphas_batch : int or None
        Size of the batch for over alphas. Used for memory reasons.
        If None, uses all n_alphas at once.
    progress_bar : bool
        If True, display a progress bar over gammas.
    Ks_in_cpu : bool
        If True, keep Ks in CPU memory to limit GPU memory (slower).
        This feature is not available through the scikit-learn API.
    conservative : bool
        If True, when selecting the hyperparameter alpha, take the largest one
        that is less than one standard deviation away from the best.
        If False, take the best.
    Y_in_cpu : bool
        If True, keep the target values ``Y`` in CPU memory (slower).
    diagonalize_method : str in {"eigh", "svd"}
        Method used to diagonalize the kernel.
    return_alphas : bool
        If True, return the best alpha value for each target.

    Returns
    -------
    deltas : array of shape (n_kernels, n_targets)
        Best log kernel weights for each target.
    refit_weights : array or None
        Refit regression weights on the entire dataset, using selected best
        hyperparameters. Refit weights are always stored on CPU memory.
        If return_weights == 'primal', shape is (n_features, n_targets),
        if return_weights == 'dual', shape is (n_samples, n_targets),
        else, None.
    cv_scores : array of shape (n_iter, n_targets)
        Cross-validation scores per iteration, averaged over splits, for the
        best alpha. Cross-validation scores will always be on CPU memory.
    best_alphas : array of shape (n_targets, )
        Best alpha value per target. Only returned if return_alphas is True.
    intercept : array of shape (n_targets,)
        Intercept. Only returned when fit_intercept is True.
    """
    backend = get_backend()
    if isinstance(n_iter, int):
        gammas = generate_dirichlet_samples(n_samples=n_iter,
                                            n_kernels=len(Ks),
                                            concentration=concentration,
                                            random_state=random_state)
        gammas[0] = 1 / len(Ks)
    elif n_iter.ndim == 2:
        gammas = n_iter
        assert gammas.shape[1] == Ks.shape[0]
    else:
        raise ValueError("Unknown parameter n_iter=%r." % (n_iter, ))

    if isinstance(alphas, numbers.Number) or alphas.ndim == 0:
        alphas = backend.ones_like(Y, shape=(1, )) * alphas

    dtype = Ks.dtype
    gammas = backend.asarray(gammas, dtype=dtype)
    device = getattr(gammas, "device", None)
    gammas, alphas, Xs = backend.check_arrays(gammas, alphas, Xs)
    Y = backend.asarray(Y, dtype=dtype, device="cpu" if Y_in_cpu else device)
    Ks = backend.asarray(Ks, dtype=dtype,
                         device="cpu" if Ks_in_cpu else device)

    if fit_intercept:
        if return_weights == 'dual':
            Ks, Y, Ks_rows, Y_offset = _helper_intercept(Ks, Y)
        elif return_weights == 'primal':
            raise NotImplementedError(
                'Computing the intercept with return_weights="primal" is not'
                'implemented. Use fit_intercept=False or'
                'return_weights="dual".')
        else:
            raise ValueError(f'Cannot compute the intercept if return_weights'
                             f'is equal to {return_weights}.')

    n_samples, n_targets = Y.shape
    if n_targets_batch is None:
        n_targets_batch = n_targets
    if n_targets_batch_refit is None:
        n_targets_batch_refit = n_targets_batch
    if n_alphas_batch is None:
        n_alphas_batch = len(alphas)

    cv = check_cv(cv)
    n_splits = cv.get_n_splits()
    n_kernels = len(Ks)
    for train, val in cv.split(Y):
        if len(val) == 0 or len(train) == 0:
            raise ValueError("Empty train or validation set. "
                             "Check that `cv` is correctly defined.")

    if jitter_alphas:
        random_generator = check_random_state(random_state)
        given_alphas = backend.copy(alphas)

    best_gammas = backend.full_like(gammas, fill_value=1.0 / n_kernels,
                                    shape=(n_kernels, n_targets))
    best_alphas = backend.ones_like(gammas, shape=n_targets)
    cv_scores = backend.zeros_like(gammas, shape=(len(gammas), n_targets),
                                   device="cpu")
    current_best_scores = backend.full_like(gammas, fill_value=-backend.inf,
                                            shape=n_targets)

    # initialize refit ridge weights
    if return_weights == 'primal':
        if Xs is None:
            raise ValueError("Xs is needed to compute the primal weights.")
        n_features = sum(X.shape[1] for X in Xs)
        refit_weights = backend.zeros_like(gammas,
                                           shape=(n_features, n_targets),
                                           device="cpu")

    elif return_weights == 'dual':
        refit_weights = backend.zeros_like(gammas,
                                           shape=(n_samples, n_targets),
                                           device="cpu")
    elif return_weights is None:
        refit_weights = None
    else:
        raise ValueError("Unknown parameter return_weights=%r." %
                         (return_weights, ))

    for ii, gamma in enumerate(
            bar(gammas, '%d random sampling with cv' % len(gammas),
                use_it=progress_bar)):

        if Ks_in_cpu:
            K = (backend.to_cpu(gamma[:, None, None]) * Ks).sum(0)
            K = backend.to_gpu(K, device=device)
        else:
            K = (gamma[:, None, None] * Ks).sum(0)

        if jitter_alphas:
            noise = backend.asarray_like(random_generator.rand(), alphas)
            alphas = given_alphas * (10 ** (noise - 0.5))

        scores = backend.zeros_like(gammas,
                                    shape=(n_splits, len(alphas), n_targets))
        for jj, (train, test) in enumerate(cv.split(K)):
            train = backend.to_gpu(train, device=device)
            test = backend.to_gpu(test, device=device)
            Ktrain, Ktest = K[train[:, None], train], K[test[:, None], train]
            if fit_intercept:
                centerer = KernelCenterer()
                Ktrain = centerer.fit_transform(Ktrain)
                Ktest = centerer.transform(Ktest)

            for matrix, alpha_batch in _decompose_kernel_ridge(
                    Ktrain=Ktrain, alphas=alphas, Ktest=Ktest,
                    negative_eigenvalues="nan", n_alphas_batch=n_alphas_batch,
                    method=diagonalize_method):
                # n_alphas_batch, n_samples_test, n_samples_train = \
                # matrix.shape

                for start in range(0, n_targets, n_targets_batch):
                    batch = slice(start, start + n_targets_batch)
                    Ybatch = backend.to_gpu(Y[:, batch], device=device)
                    Ytrain, Ytest = Ybatch[train], Ybatch[test]
                    del Ybatch
                    if fit_intercept:
                        Ytrain_mean = Ytrain.mean(0)
                        Ytrain = Ytrain - Ytrain_mean
                        Ytest = Ytest - Ytrain_mean

                    predictions = backend.matmul(matrix, Ytrain)
                    # n_alphas_batch, n_samples_test, n_targets_batch = \
                    # predictions.shape

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        scores[jj, alpha_batch, batch] = score_func(
                            Ytest, predictions)
                        # n_alphas_batch, n_targets_batch = score.shape

                # make small alphas impossible to select
                too_small_alphas = backend.isnan(matrix[:, 0, 0])
                scores[jj, alpha_batch, :][too_small_alphas] = -1e5

                del matrix, predictions
            del train, test

        # select best alphas
        alphas_argmax, cv_scores_ii = _select_best_alphas(
            scores, alphas, local_alpha, conservative)
        cv_scores[ii, :] = backend.to_cpu(cv_scores_ii)

        # update best_gammas and best_alphas
        mask = cv_scores_ii > current_best_scores
        current_best_scores[mask] = cv_scores_ii[mask]
        best_gammas[:, mask] = gamma[:, None]
        best_alphas[mask] = alphas[alphas_argmax[mask]]

        # compute primal or dual weights on the entire dataset (nocv)
        if return_weights is not None:
            update_indices = backend.flatnonzero(mask)
            if Y_in_cpu:
                update_indices = backend.to_cpu(update_indices)
            if len(update_indices) > 0:

                # refit weights only for alphas used by at least one target
                used_alphas = backend.unique(best_alphas[mask])
                dual_weights = backend.zeros_like(
                    K, shape=(n_samples, len(update_indices)), device="cpu")
                for matrix, alpha_batch in _decompose_kernel_ridge(
                        K, used_alphas, Ktest=None,
                        negative_eigenvalues="zeros",
                        n_alphas_batch=min(len(used_alphas), n_alphas_batch),
                        method=diagonalize_method):

                    for start in range(0, len(update_indices),
                                       n_targets_batch_refit):
                        batch = slice(start, start + n_targets_batch_refit)

                        weights = backend.matmul(
                            matrix,
                            backend.to_gpu(Y[:, update_indices[batch]],
                                           device=device))
                        # used_n_alphas_batch, n_samples, n_targets_batch = \
                        # weights.shape

                        # select alphas corresponding to best cv_score
                        alphas_indices = backend.searchsorted(
                            used_alphas, best_alphas[mask][batch])
                        # mask targets whose selected alphas are outside the
                        # alpha batch
                        mask2 = backend.isin(
                            alphas_indices,
                            backend.arange(len(used_alphas))[alpha_batch])
                        # get indices in alpha_batch
                        alphas_indices = backend.searchsorted(
                            backend.arange(len(used_alphas))[alpha_batch],
                            alphas_indices[mask2])
                        # update corresponding weights
                        tmp = weights[alphas_indices, :,
                                      backend.arange(weights.shape[2])[mask2]]
                        dual_weights[:, batch][:, backend.to_cpu(mask2)] = \
                            backend.to_cpu(tmp).T
                        del weights, alphas_indices, mask2
                    del matrix

                if return_weights == 'primal':
                    # multiply by g and not np.sqrt(g), as we then want to use
                    # the primal weights on the unscaled features Xs, and not
                    # on the scaled features (np.sqrt(g) * Xs)
                    X = backend.concatenate([t * g for t, g in zip(Xs, gamma)],
                                            1)
                    primal_weights = backend.to_cpu(X.T) @ dual_weights
                    refit_weights[:, backend.to_cpu(mask)] = primal_weights
                    del X, primal_weights

                elif return_weights == 'dual':
                    refit_weights[:, backend.to_cpu(mask)] = dual_weights

                del dual_weights
            del update_indices
        del K, mask

    deltas = backend.log(best_gammas / best_alphas[None, :])
    if return_weights == 'dual':
        refit_weights *= backend.to_cpu(best_alphas)

    if fit_intercept:
        if return_weights == 'dual':
            intercept = backend.to_cpu(Y_offset)
            intercept -= ((backend.to_cpu(Ks_rows) @ refit_weights) *
                          backend.to_cpu(backend.exp(deltas))).sum(0)
        else:
            return NotImplementedError()

    results = [deltas, refit_weights, cv_scores]
    if return_alphas:
        results.append(best_alphas)
    if fit_intercept:
        results.append(intercept)
    return results


def _select_best_alphas(scores, alphas, local_alpha, conservative):
    """Helper to select the best alphas

    Parameters
    ----------
    scores : array of shape (n_splits, n_alphas, n_targets)
        Cross validation scores.
    alphas :array of shape (n_alphas, )
        Regularization parameters.
    local_alpha : bool
        If True, alphas are selected per target, else shared over all targets.
    conservative : bool
        If True, when selecting the hyperparameter alpha, take the largest one
        that is less than one standard deviation away from the best.
        If False, take the best.

    Returns
    -------
    alphas_argmax : array of shape (n_targets, )
        Indices of the best alphas.
    best_scores_mean : arrya of shape (n_targets, )
        Scores, averaged over splits, and maximized over alphas.
    """
    backend = get_backend()

    # average scores over splits
    scores_mean = backend.mean_float64(scores, axis=0)
    # add epsilon slope to select larger alphas if scores are equal
    scores_mean += (backend.log(alphas) * 1e-10)[:, None]

    # compute the max over alphas
    axis = 0
    if local_alpha:
        alphas_argmax = backend.argmax(scores_mean, axis)

        if conservative:
            # take a conservative alpha, the largest that beats the best
            # score minus one standard deviation
            scores_std = scores.std(0)
            to_beat = backend.apply_argmax(scores_mean - scores_std,
                                           alphas_argmax, axis)
            does_beat = backend.asarray(scores_mean > to_beat, dtype="float32")
            # add bias toward large alphas (scaled to be << 1)
            does_beat += backend.log(alphas)[:, None] * 1e-4
            alphas_argmax = backend.argmax(does_beat, axis)
    else:
        if conservative:
            raise NotImplementedError()
        else:
            alphas_argmax = backend.argmax(scores_mean.mean(1))
            alphas_argmax = backend.full_like(alphas_argmax,
                                              shape=scores_mean.shape[1],
                                              fill_value=alphas_argmax)
    best_scores_mean = backend.apply_argmax(scores_mean, alphas_argmax, axis)

    return alphas_argmax, best_scores_mean


def generate_dirichlet_samples(n_samples, n_kernels, concentration=[.1, 1.],
                               random_state=None):
    """Generate samples from a Dirichlet distribution.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_kernels : int
        Number of dimension of the distribution.
    concentration : float, or list of float
        Concentration parameters of the Dirichlet distribution.
        A value of 1 corresponds to uniform sampling over the simplex.
        A value of infinity corresponds to equal weights.
        If a list, samples cycle through the list.
    random_state : int, or None
        Random generator seed. Use an int for deterministic samples.

    Returns
    -------
    gammas : array of shape (n_samples, n_kernels)
        Dirichlet samples.
    """
    import numpy as np
    random_generator = check_random_state(random_state)

    concentration = np.atleast_1d(concentration)
    n_concentrations = len(concentration)
    n_samples_per_concentration = int(
        np.ceil(n_samples / float(n_concentrations)))

    # generate the gammas
    gammas = []
    for conc in concentration:
        if conc == np.inf:
            gamma = np.full(n_kernels, fill_value=1. / n_kernels)[None]
            gamma = np.tile(gamma, (n_samples_per_concentration, 1))
        else:
            gamma = random_generator.dirichlet([conc] * n_kernels,
                                               n_samples_per_concentration)
        gammas.append(gamma)
    gammas = np.vstack(gammas)

    # reorder the gammas to alternate between concentrations:
    # [a0, a1, a2, a0, a1, a2] instead of [a0, a0, a1, a1, a2, a2]
    gammas = gammas.reshape(n_concentrations, n_samples_per_concentration,
                            n_kernels)
    gammas = np.swapaxes(gammas, 0, 1)
    gammas = gammas.reshape(n_concentrations * n_samples_per_concentration,
                            n_kernels)

    # remove extra gammas
    gammas = gammas[:n_samples]

    # cast to current backend
    backend = get_backend()
    gammas = backend.asarray(gammas)

    return gammas


def _decompose_kernel_ridge(Ktrain, alphas, Ktest=None, n_alphas_batch=None,
                            method="eigh", negative_eigenvalues="zeros"):
    """Precompute resolution matrices for kernel ridge predictions.

    To compute the prediction::

        Ytest_hat = Ktest @ (Ktrain + alphas * Id)^-1 @ Ytrain

    this function precomputes::

        matrices = Ktest @ (Ktrain + alphas * Id)^-1

    or just::

        matrices = (Ktrain + alphas * Id)^-1

    if Ktest is None.

    Parameters
    ----------
    Ktrain : array of shape (n_samples_train, n_samples_train)
        Training kernel for one feature space.
    alphas : float, or array of shape (n_alphas, )
        Range of ridge regularization parameter.
    Ktest : array of shape (n_samples_test, n_samples_train)
        Testing kernel for one feature space.
    n_alphas_batch : int or None
        If not None, returns a generator over batches of alphas.
    method : str in {"eigh", "svd"}
        Method used to diagonalize the kernel.
    negative_eigenvalues : str in {"nan", "error", "zeros"}
        If the decomposition leads to negative eigenvalues (wrongly emerging
        from float32 errors):
            - "error" raises an error.
            - "zeros" remplaces them with zeros.
            - "nan" returns nans if the regularization does not compensate
                twice the smallest negative value, else it ignores the problem.

    Returns
    -------
    matrices : array of shape (n_alphas, n_samples_train, n_samples_train) or \
        (n_alphas, n_samples_test, n_samples_train) if Ktest is not None
        Precomputed resolution matrices.
    alpha_batch : slice
        Slice of the batch of alphas.
    """
    backend = get_backend()

    use_alpha_batch = n_alphas_batch is not None
    if n_alphas_batch is None:
        n_alphas_batch = len(alphas)

    if method == "eigh":
        # diagonalization: K = V @ np.diag(eigenvalues) @ V.T
        eigenvalues, V = backend.eigh(Ktrain)
        # match SVD notations: K = U @ np.diag(eigenvalues) @ Vt
        U = V
        Vt = V.T
    elif method == "svd":
        # SVD: K = U @ np.diag(eigenvalues) @ Vt
        U, eigenvalues, Vt = backend.svd(Ktrain)
    else:
        raise ValueError("Unknown method=%r." % (method, ))

    if Ktest is not None:
        Ktest_V = backend.matmul(Ktest, Vt.T)

    for start in range(0, len(alphas), n_alphas_batch):
        batch = slice(start, start + n_alphas_batch)

        ev_weighting = (alphas[batch, None] + eigenvalues) ** -1

        # negative eigenvalues can emerge from incorrect kernels,
        # or from float32
        if eigenvalues[0] < 0:
            if negative_eigenvalues == "nan":
                ev_weighting[alphas[batch] < -eigenvalues[0] *
                             2, :] = backend.asarray(backend.nan,
                                                     dtype=ev_weighting.dtype)

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

        if Ktest is not None:
            matrices = backend.matmul(Ktest_V, ev_weighting[:, :, None] * U.T)
        else:
            matrices = backend.matmul(Vt.T, ev_weighting[:, :, None] * U.T)

        if use_alpha_batch:
            yield matrices, batch
        else:
            return matrices, batch

        del matrices


def solve_kernel_ridge_cv_eigenvalues(K, Y, alphas=1.0, score_func=l2_neg_loss,
                                      cv=5, fit_intercept=False,
                                      local_alpha=True, n_targets_batch=None,
                                      n_targets_batch_refit=None,
                                      n_alphas_batch=None, conservative=False,
                                      Y_in_cpu=False):
    """Solve kernel ridge regression with a grid search over alphas.

    Parameters
    ----------
    K : array of shape (n_samples, n_samples)
        Input kernel.
    Y : array of shape (n_samples, n_targets)
        Target data.
    alphas : float or array of shape (n_alphas, )
        Range of ridge regularization parameter.
    score_func : callable
        Function used to compute the score of predictions versus Y.
    cv : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.
    fit_intercept : boolean
        Whether to fit an intercept. If False, K should be centered
        (see KernelCenterer), and Y must be zero-mean over samples.
    local_alpha : bool
        If True, alphas are selected per target, else shared over all targets.
    n_targets_batch : int or None
        Size of the batch for over targets during cross-validation.
        Used for memory reasons. If None, uses all n_targets at once.
    n_targets_batch_refit : int or None
        Size of the batch for over targets during refit.
        Used for memory reasons. If None, uses all n_targets at once.
    n_alphas_batch : int or None
        Size of the batch for over alphas. Used for memory reasons.
        If None, uses all n_alphas at once.
    conservative : bool
        If True, when selecting the hyperparameter alpha, take the largest one
        that is less than one standard deviation away from the best.
        If False, take the best.
    Y_in_cpu : bool
        If True, keep the target values ``Y`` in CPU memory (slower).

    Returns
    -------
    best_alphas : array of shape (n_targets, )
        Selected best hyperparameter alphas.
    dual_weights : array of shape (n_samples, n_targets)
        Kernel ridge coefficients refit on the entire dataset, using selected
        best hyperparameters alpha. Always stored on CPU memory.
    cv_scores : array of shape (n_targets, )
        Cross-validation scores averaged over splits, for the best alpha.
    """
    backend = get_backend()

    n_iter = backend.ones_like(K, shape=(1, 1))
    fixed_params = dict(return_weights="dual", Xs=None, progress_bar=False,
                        concentration=None, jitter_alphas=False,
                        random_state=None, n_iter=n_iter)

    copied_params = dict(alphas=alphas, score_func=score_func, cv=cv,
                         local_alpha=local_alpha, fit_intercept=fit_intercept,
                         n_targets_batch=n_targets_batch,
                         n_targets_batch_refit=n_targets_batch_refit,
                         n_alphas_batch=n_alphas_batch,
                         conservative=conservative, Y_in_cpu=Y_in_cpu)

    tmp = solve_multiple_kernel_ridge_random_search(K[None], Y,
                                                    **copied_params,
                                                    **fixed_params)
    if fit_intercept:
        deltas, dual_weights, cv_scores, intercept = tmp
        best_alphas = backend.exp(-deltas[0])
        dual_weights /= backend.to_cpu(best_alphas)
        return best_alphas, dual_weights, cv_scores, intercept
    else:
        deltas, dual_weights, cv_scores = tmp
        best_alphas = backend.exp(-deltas[0])
        dual_weights /= backend.to_cpu(best_alphas)
        return best_alphas, dual_weights, cv_scores
