from sklearn.model_selection import check_cv

from ..backend import get_current_backend
from ..progress_bar import bar
from ..scoring import l2_neg_loss


def solve_multiple_kernel_ridge_random_search(
        Ks, Y, gammas, alphas, score_func=l2_neg_loss, cv_splitter=10,
        compute_weights=None, Xs=None, local_alpha=True, jitter_alphas=False,
        random_state=None, n_targets_batch=None, n_targets_batch_refit=None,
        n_alphas_batch=None):
    """Solve multiple kernel ridge regression using random search.

    Parameters
    ----------
    Ks : array of shape (n_kernels, n_samples, n_samples)
        Input kernels for each feature space.
    Y : array of shape (n_samples, n_targets)
        Target data.
    gammas : array of shape (n_gammas, n_kernels)
        Kernel weights for each feature space.
    alphas : float or array of shape (n_alphas, )
        Range of ridge regularization parameter.
    score_func : callable
        Function used to compute the score of predictions versus Y.
    cv_splitter : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.
    compute_weights : None, 'primal', or 'dual'
        Whether to refit on the entire dataset and return the weights.
    Xs : array of shape (n_kernels, n_samples, n_features) or None
        Necessary if compute_weights == 'primal'.
    local_alpha : bool
        If True, alphas are selected per voxel, else globally over all voxels.
    jitter_alphas : bool
        If True, alphas range is slightly gittered for each gamma.
    random_state : int, np.random.RandomState, or None
        Random generator state, used only if jitter_alphas is True.
    n_targets_batch : int or None
        Size of the batch for over targets during cross-validation.
        Used for memory reasons. If None, uses all n_targets at once.
    n_targets_batch_refit : int or None
        Size of the batch for over targets during refit.
        Used for memory reasons. If None, uses all n_targets at once.
    n_alphas_batch : int or None
        Size of the batch for over alphas. Used for memory reasons.
        If None, uses all n_alphas at once.

    Returns
    -------
    all_scores_mean : array of shape (n_gammas, n_targets)
        Scores averaged over splits, for best alpha.
    best_gammas : array of shape (n_kernels, n_targets)
        Best kernel weights.
    best_alphas : array of shape (n_targets)
        Best ridge regularization.
    refit_weights : array or None
        Refit regression weights on the entire dataset, using selected best
        hyperparameters.
        If compute_weights == 'primal', shape is (n_features, n_targets),
        if compute_weights == 'dual', shape is (n_samples, n_targets),
        else, None.
    """
    backend = get_current_backend()

    n_samples, n_targets = Y.shape
    if n_targets_batch is None:
        n_targets_batch = n_targets
    if n_targets_batch_refit is None:
        n_targets_batch_refit = n_targets_batch
    if n_alphas_batch is None:
        n_alphas_batch = len(alphas)

    cv_splitter = check_cv(cv_splitter)
    n_splits = cv_splitter.get_n_splits()
    n_kernels = len(Ks)

    if jitter_alphas:
        random_generator = backend.check_random_state(random_state)
        given_alphas = alphas.clone()

    best_gammas = backend.full_like(Ks, fill_value=1.0 / n_kernels,
                                    shape=(n_kernels, n_targets))
    best_alphas = backend.ones_like(Ks, shape=n_targets)
    all_scores_mean = backend.zeros_like(Ks, shape=(len(gammas), n_targets))
    current_best_scores = backend.full_like(Ks, fill_value=-backend.inf,
                                            shape=n_targets)

    if compute_weights == 'primal':
        if Xs is None:
            raise ValueError("Xs is needed to compute the primal weights.")
        n_features = sum(X.shape[1] for X in Xs)
        refit_weights = backend.zeros_like(Ks, shape=(n_features, n_targets))

    elif compute_weights == 'dual':
        refit_weights = backend.zeros_like(Ks, shape=(n_samples, n_targets))
    elif compute_weights is None:
        refit_weights = None
    else:
        raise ValueError("Unknown parameter compute_weights=%r." %
                         (compute_weights, ))

    for ii, gamma in enumerate(
            bar(gammas, '%d random sampling with cv' % len(gammas))):

        K = (gamma[:, None, None] * Ks).sum(0)

        if jitter_alphas:
            alphas = given_alphas * (10 ** (random_generator.rand() - 0.5))

        scores = backend.zeros_like(Y,
                                    shape=(n_splits, len(alphas), n_targets))
        for jj, (train, test) in enumerate(cv_splitter.split(K)):
            if hasattr(K, "device"):
                train = backend.asarray(train, device=K.device)
                test = backend.asarray(test, device=K.device)

            for matrix, alpha_batch in _decompose_kernel_ridge(
                    Ktrain=K[train[:, None], train], alphas=alphas,
                    Ktest=K[test[:, None], train],
                    n_alphas_batch=n_alphas_batch):
                # n_alphas_batch, n_samples_test, n_samples_train = \
                # matrix.shape

                for start in range(0, n_targets, n_targets_batch):
                    batch = slice(start, start + n_targets_batch)

                    predictions = backend.matmul(matrix, Y[train, batch])
                    # n_alphas_batch, n_samples_test, n_targets_batch = \
                    # predictions.shape

                    scores[jj, alpha_batch, batch] = score_func(
                        Y[test, batch], predictions)
                    # n_alphas_batch, n_targets_batch = score.shape

                # make small alphas impossible to select
                too_small_alphas = backend.isnan(matrix[:, 0, 0])
                scores[jj, alpha_batch, :][too_small_alphas] = -1e5

                del matrix, predictions
            del train, test

        # average scores over splits
        scores_mean = backend.mean_float64(scores, axis=0)
        # add epsilon slope to select larger alphas if scores are equal
        scores_mean += (backend.log10(alphas) * 1e-10)[:, None]

        # compute the max over alphas
        axis = 0
        if local_alpha:
            alphas_argmax = backend.argmax(scores_mean, axis)
        else:
            alphas_argmax = backend.argmax(scores_mean.mean(1))
            alphas_argmax = backend.full_like(Ks, shape=scores_mean.shape[1],
                                              dtype=backend.int32,
                                              fill_value=alphas_argmax)
        all_scores_mean[ii, :] = backend.apply_argmax(scores_mean,
                                                      alphas_argmax, axis)

        # update best_gammas and best_alphas
        mask = all_scores_mean[ii, :] > current_best_scores
        current_best_scores[mask] = all_scores_mean[ii, mask]
        best_gammas[:, mask] = gamma[:, None]
        best_alphas[mask] = alphas[alphas_argmax[mask]]

        # compute primal or dual weights on the entire dataset (nocv)
        if compute_weights is not None:
            update_indices = backend.flatnonzero(mask)
            if len(update_indices) > 0:

                # XXX. refit over selected alphas only
                dual_weights = backend.zeros_like(
                    K, shape=(n_samples, len(update_indices)))
                for matrix, alpha_batch in _decompose_kernel_ridge(
                        K, alphas, Ktest=None, n_alphas_batch=n_alphas_batch):

                    for start in range(0, len(update_indices),
                                       n_targets_batch_refit):
                        batch = slice(start, start + n_targets_batch_refit)
                        weights = backend.matmul(matrix,
                                                 Y[:, update_indices[batch]])
                        # n_alphas_batch, n_samples, n_targets_batch = \
                        # weights.shape

                        # select alphas corresponding to best cv_score
                        alphas_indices = alphas_argmax[update_indices[batch]]
                        # mask targets whose selected alphas are outside the
                        # alpha batch
                        mask2 = backend.isin(
                            alphas_indices,
                            backend.arange(len(alphas))[alpha_batch])
                        # get indices in alpha_batch
                        alphas_indices = backend.searchsorted(
                            backend.arange(len(alphas))[alpha_batch],
                            alphas_indices[mask2])
                        # update corresponding weights
                        dual_weights[:, batch][:, mask2] = (
                            weights[alphas_indices, :,
                                    backend.arange(weights.shape[2])[mask2]]).T
                        del weights
                    del matrix

                if compute_weights == 'primal':
                    # multiply by g and not np.sqrt(g), as we then want to use
                    # the primal weights on the unscaled features Xs, and not
                    # on the scaled features (np.sqrt(g) * Xs)
                    X = backend.concatenate([t * g for t, g in zip(Xs, gamma)],
                                            1)
                    primal_weights = backend.matmul(X.T, dual_weights)
                    refit_weights[:, mask] = primal_weights

                    del X, primal_weights
                elif compute_weights == 'dual':
                    refit_weights[:, mask] = dual_weights

                del dual_weights

        del K

    return all_scores_mean, best_gammas, best_alphas, refit_weights


def _decompose_kernel_ridge(Ktrain, alphas, Ktest=None, n_alphas_batch=None,
                            method="eigh", negative_eigenvalues="nan"):
    """Precompute resolution matrices for kernel ridge predictions

    To compute the prediction:
        Ytest_hat = Ktest @ (Ktrain + alphas * Id)^-1 @ Ytrain
    this function precomputes:
        matrices = Ktest @ (Ktrain + alphas * Id)^-1
    or just:
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
    negative_eigenvalues : str in {"nan", "error"}
        If the decomposition leads to negative eigenvalues (wrongly emerging
        from float32 errors):
            - "error" raises an error.
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
    backend = get_current_backend()

    use_alpha_batch = n_alphas_batch is not None
    if n_alphas_batch is None:
        n_alphas_batch = len(alphas)

    if method == "eigh":
        # diagonalization: K = V @ np.diag(eigenvalues) @ V.T
        eigenvalues, V = backend.eigh(Ktrain)
        U = V
    elif method == "svd":
        # SVD: K = U @ np.diag(eigenvalues) @ V.T
        U, eigenvalues, V = backend.svd(Ktrain)
    else:
        raise ValueError("Unknown method=%r." % (method, ))

    if Ktest is not None:
        Ktest_V = backend.matmul(Ktest, V)

    for start in range(0, len(alphas), n_alphas_batch):
        batch = slice(start, start + n_alphas_batch)

        ev_weighting = (alphas[batch, None] + eigenvalues) ** -1

        # negative eigenvalues can emerge from incorrect kernels,
        # or from float32
        if eigenvalues[0] < 0:
            if negative_eigenvalues == "nan":
                ev_weighting[alphas[batch] < -eigenvalues[0] *
                             2, :] = backend.nan

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
            matrices = backend.matmul(V, ev_weighting[:, :, None] * U.T)

        if use_alpha_batch:
            yield matrices, batch
        else:
            return matrices, batch
