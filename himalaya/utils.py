import numpy as np

from .backend import get_backend
from .validation import check_random_state


def compute_lipschitz_constants(Xs, kernelize="XTX", random_state=None):
    """Compute Lipschitz constants of gradients of linear regression problems.

    Find the largest eigenvalue of X^TX for several X, using power iteration.

    Parameters
    ----------
    Xs : array of shape (n_kernels, n_samples, n_features) or \
            (n_kernels, n_samples, n_samples)
        Multiple linear features or kernels.
    kernelize : str in {"XTX", "XXT", "X"}
        Whether to consider X^TX, XX^T, or directly X.
    random_state : int, or None
        Random generator seed. Use an int for deterministic search.

    Returns
    -------
    lipschitz : array of shape (n_kernels)
        Lipschitz constants.
    """
    backend = get_backend()

    if kernelize == "XXT":
        XTs = backend.transpose(Xs, (0, 2, 1))
        kernels = backend.matmul(Xs, XTs)
        del XTs
    elif kernelize == "XTX":
        XTs = backend.transpose(Xs, (0, 2, 1))
        kernels = backend.matmul(XTs, Xs)
        del XTs
    elif kernelize == "X":
        kernels = Xs
    else:
        raise ValueError("Unknown parameter kernelize=%r" % (kernelize, ))

    # check the random state
    random_generator = check_random_state(random_state)
    ys = random_generator.randn(*(kernels.shape[:2] + (1, )))

    ys = backend.asarray_like(ys, Xs)
    for i in range(10):
        ys /= backend.norm(ys, axis=1, keepdims=True) + 1e-16
        ys = backend.matmul(kernels, ys)
    evs = backend.norm(ys, axis=1)[:, 0]
    return evs


def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    """Test array equality, casting all arrays to numpy."""
    backend = get_backend()
    x = backend.to_numpy(x)
    y = backend.to_numpy(y)
    return np.testing.assert_array_almost_equal(x, y, decimal=decimal,
                                                err_msg=err_msg,
                                                verbose=verbose)
