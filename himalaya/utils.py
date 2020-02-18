from .backend import get_current_backend


def compute_lipschitz_constants(Xs, kernelize="XTX"):
    """Compute Lipschitz constants of gradients of linear regression problems

    Find the largest eigenvalue of X^TX for several X, using power iteration.

    Parameters
    ----------
    Xs : array of shape (n_kernels, n_samples, n_features) or
            (n_kernels, n_samples, n_samples)
        Multiple linear features or kernels.
    kernelize : str in {"XTX", "XXT", "X"}
        Whether to consider X^TX, XX^T, or directly X.

    Returns
    -------
    lipschitz : array of shape (n_kernels)
        Lipschitz constants.
    """
    backend = get_current_backend()

    if kernelize == "XXT":
        XTs = backend.transpose(Xs, (0, 2, 1))
        kernels = backend.matmul(Xs, XTs)
    elif kernelize == "XTX":
        XTs = backend.transpose(Xs, (0, 2, 1))
        kernels = backend.matmul(XTs, Xs)
    elif kernelize == "X":
        kernels = Xs
    else:
        raise ValueError("Unknown parameter kernelize=%r" % (kernelize,))

    ys = backend.randn(*(kernels.shape[:2] + (1, )))
    ys = backend.asarray_like(ys, Xs)
    for i in range(10):
        ys /= backend.norm(ys, axis=1, keepdims=True) + 1e-16
        ys = backend.matmul(kernels, ys)
    evs = backend.norm(ys, axis=1)[:, 0]
    return evs
