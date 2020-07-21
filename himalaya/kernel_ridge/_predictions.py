from ..backend import get_backend


def predict_weighted_kernel_ridge(Ks, dual_weights, deltas, split=False):
    """
    Compute predictions, typically on a test set.

    Parameters
    ----------
    Ks : array of shape (n_kernels, n_samples_test, n_samples_train)
        Test kernels.
    dual_weights : array of shape (n_samples_train, n_targets)
        Dual weights of the kernel ridge model.
    deltas : array of shape (n_kernels, n_targets)
        Log kernel weights for each target.
    split : bool
        If True, the predictions is split across kernels.

    Returns
    -------
    Y_hat : array of shape (n_samples_test, n_targets) or \
            (n_kernels, n_samples_test, n_targets) (if split is True)
        Predicted values.
    """
    backend = get_backend()

    Ks, dual_weights, deltas = backend.check_arrays(Ks, dual_weights, deltas)
    chi = backend.matmul(Ks, dual_weights)
    split_predictions = backend.exp(deltas[:, None, :]) * chi
    if split:
        Y_hat = split_predictions
    else:
        Y_hat = split_predictions.sum(0)

    return Y_hat


def predict_and_score_weighted_kernel_ridge(Ks, dual_weights, deltas, Y,
                                            score_func, split=False,
                                            n_targets_batch=None):
    """
    Compute predictions, typically on a test set, and compute the score.

    Parameters
    ----------
    Ks : array of shape (n_kernels, n_samples_test, n_samples_train)
        Input kernels.
    dual_weights : array of shape (n_samples_train, n_targets)
        Dual weights of the kernel ridge model.
    deltas : array of shape (n_kernels, n_targets)
        Log kernel weights for each target.
    Y : array of shape (n_samples_test, n_targets)
        Target data.
    score_func : callable
        Function used to compute the score of predictions.
    split : bool
        If True, the predictions is split across kernels.
    n_targets_batch : int or None
        Size of the batch for computing predictions. Used for memory reasons.
        If None, uses all n_targets at once.

    Returns
    -------
    scores : array of shape (n_targets, ) or (n_kernels, n_targets) (if split)
        Prediction score per target.
    """
    backend = get_backend()
    Ks, dual_weights, deltas, Y = backend.check_arrays(Ks, dual_weights,
                                                       deltas, Y)

    n_kernels, n_targets = deltas.shape
    if split:
        scores = backend.zeros_like(Y, shape=(n_kernels, n_targets))
    else:
        scores = backend.zeros_like(Y, shape=(n_targets))

    if n_targets_batch is None:
        n_targets_batch = n_targets
    for start in range(0, n_targets, n_targets_batch):
        batch = slice(start, start + n_targets_batch)
        predictions = predict_weighted_kernel_ridge(Ks, dual_weights[:, batch],
                                                    deltas[:, batch],
                                                    split=split)
        score_batch = score_func(Y[:, batch], predictions)

        if split:
            scores[:, batch] = score_batch
        else:
            scores[batch] = score_batch

    return scores


def primal_weights_kernel_ridge(dual_weights, X_fit):
    """Compute the primal weights for kernel ridge regression.

    Parameters
    ----------
    dual_weights : array of shape (n_samples_fit, n_targets)
        Dual coefficient of the kernel ridge regression.
    X_fit : array of shape (n_samples_fit, n_features)
        Training features.

    Returns
    -------
    primal_weights : array of shape (n_features, n_targets)
        Primal coefficients of the equivalent ridge regression. The
        coefficients are computed on CPU memory, since they can be large.
    """
    backend = get_backend()
    X_fit = backend.to_cpu(X_fit)
    dual_weights = backend.to_cpu(dual_weights)

    return X_fit.T @ dual_weights


def primal_weights_weighted_kernel_ridge(dual_weights, deltas, Xs_fit):
    """Compute the primal weights for weighted kernel ridge regression.

    Parameters
    ----------
    dual_weights : array of shape (n_samples_fit, n_targets)
        Dual coefficient of the kernel ridge regression.
    deltas : array of shape (n_kernels, n_targets)
        Log of kernel weights.
    Xs_fit : list of arrays of shape (n_samples_fit, n_features)
        Training features. The list should have `n_kernels` elements.

    Returns
    -------
    primal_weights : list of arrays of shape (n_features, n_targets)
        Primal coefficients of the equivalent ridge regression. The
        coefficients are computed on CPU memory, since they can be large.
    """
    backend = get_backend()
    dual_weights = backend.to_cpu(dual_weights)

    primal_weights = []
    for X_fit, deltas_i in zip(Xs_fit, deltas):
        X_fit = backend.to_cpu(X_fit)
        exp_deltas_i = backend.to_cpu(backend.exp(deltas_i))
        primal_weights_i = X_fit.T @ dual_weights * exp_deltas_i[None]
        primal_weights.append(primal_weights_i)

    return primal_weights
