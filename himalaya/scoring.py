import warnings

from .backend import get_backend


def l2_neg_loss(y_true, y_pred):
    """L2 negative loss, computed for multiple predictions.

    Parameters
    ----------
    y_true : array or Tensor of shape (n_samples, n_targets)
        Ground truth.
    y_pred : array or Tensor of shape (n_predictions, n_samples, n_targets) or\
            (n_samples, n_targets)
        Predictions.

    Returns
    -------
    l2 : array of shape (n_predictions, n_targets) or (n_targets, )
        L2 negative losses.
    """
    backend = get_backend()
    y_true, y_pred = backend.check_arrays(y_true, y_pred)
    y_pred = _check_finite(y_pred)

    # broadcasting if y_pred has more dimensions
    axis = 1 if len(y_pred.shape) > len(y_true.shape) else 0
    n_samples = y_true.shape[0]

    error = ((y_true - y_pred) ** 2).sum(axis)
    l2 = -error / n_samples

    return l2


def r2_score(y_true, y_pred):
    """R2 score, computed for multiple predictions (e.g. multiple alphas).

    Parameters
    ----------
    y_true : array or Tensor of shape (n_samples, n_targets)
        Ground truth.
    y_pred : array or Tensor of shape (n_predictions, n_samples, n_targets) or\
            (n_samples, n_targets)
        Predictions.

    Returns
    -------
    r2 : array of shape (n_predictions, n_targets) or (n_targets, )
        R2 scores.
    """
    backend = get_backend()
    y_true, y_pred = backend.check_arrays(y_true, y_pred)
    y_pred = _check_finite(y_pred)

    # broadcasting if y_pred has more dimensions
    axis = 1 if len(y_pred.shape) > len(y_true.shape) else 0

    error = ((y_true - y_pred) ** 2).sum(axis)
    var = ((y_true - y_true.mean(0)) ** 2).sum(0)
    r2 = 1. - error / var
    if axis == 0:
        r2[var == 0] = 0
    else:
        r2[:, var == 0] = 0

    return r2


def correlation_score(y_true, y_pred):
    """Correlation score, computed for multiple predictions.

    Parameters
    ----------
    y_true : array or Tensor of shape (n_samples, n_targets)
        Ground truth.
    y_pred : array or Tensor of shape (n_predictions, n_samples, n_targets) or\
            (n_samples, n_targets)
        Predictions.

    Returns
    -------
    correlations : array of shape (n_predictions, n_targets) or (n_targets, )
        Correlation scores.
    """
    backend = get_backend()
    y_true, y_pred = backend.check_arrays(y_true, y_pred)
    y_pred = _check_finite(y_pred)

    # broadcasting if y_pred has more dimensions
    axis = 1 if len(y_pred.shape) > len(y_true.shape) else 0

    product = _zscore(y_true, axis=0) * _zscore(y_pred, axis=axis)
    correlations = product.mean(axis)

    return correlations


def r2_score_split(y_true, y_pred, include_correlation=True):
    """Split variance explained (R2) into individual feature space components.

    When estimating a linear joint model, the predictions of
    each feature space are summed::

        Yhat_joint = Yhat_A + Yhat_B + ... + Yhat_Z

    The joint model R2 can be computed as::

        R2_joint = R2(Yhat_joint, Y)

    This function estimates the contribution of each feature space to the
    joint model R2 such that::

        R2_joint = R2_A + R2_B + ... + R2_Z

    Mathematically, this is achieved by taking into account the correlations
    between predictions (i.e. Yhat_A*Yhat_B,..., Yhat_A*Yhat_Z).
    The function can also returns an estimate that ignores these correlations.

    This function assumes that y_true is zero-mean over samples.

    Parameters
    ----------
    y_true : array of shape (n_samples, n_targets)
        Observed data. Has to be zero-mean over samples.
    y_pred : array of shape (n_kernels, n_samples, n_targets) or \
            (n_samples, n_targets)
        Predictions.
    include_correlation : bool
        Whether to include correlation between feature spaces.
        If True, individual feature space R2 sum is equivalent to the joint
        model R2 (i.e. from `y_pred.sum(0)`).

    Returns
    -------
    r2 : array (n_kernels, n_targets) or (n_targets, )
        Individual feature space R2 scores.
    """
    backend = get_backend()
    y_true, y_pred = backend.check_arrays(y_true, y_pred)
    y_pred = _check_finite(y_pred)

    if backend.any(backend.abs(y_true.mean(0)) > 1e-6):
        warnings.warn(
            'y_true has to be zero-mean over samples to compute '
            'the split r2 scores.', UserWarning)

    sst = (y_true ** 2).sum(0)

    no_split = y_pred.ndim == 2
    if no_split:
        y_pred = y_pred[None]

    n_kernels, n_samples, n_targets = y_pred.shape
    r2 = backend.zeros_like(y_pred[:, 0, :])

    for fsi in range(n_kernels):
        if include_correlation:
            asst = (y_pred[fsi] * y_pred).sum(0)
        else:
            asst = (y_pred[fsi] ** 2)

        inter = y_true * y_pred[fsi]
        r2[fsi, :] = ((2 * inter - asst) / sst).sum(0)
        del inter, asst

    if no_split:
        r2 = r2[0]

    return r2


###############################################################################


def _check_finite(y_pred):
    backend = get_backend()

    is_nan = backend.isnan(y_pred)
    if backend.any(is_nan):
        warnings.warn('nan in y_pred.')
        y_pred[is_nan] = 0

    isinf = backend.isinf(y_pred)
    if backend.any(isinf):
        warnings.warn('inf in y_pred.')
        y_pred[isinf] = 0

    return y_pred


def _zscore(X, axis):
    backend = get_backend()

    Xz = X - X.mean(axis, keepdims=True)
    std = backend.std_float64(Xz, axis, demean=False, keepdims=True)
    std[std == 0] = 1
    Xz /= std
    return Xz
