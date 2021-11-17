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


def generate_multikernel_dataset(n_kernels=4, n_targets=500,
                                 n_samples_train=1000, n_samples_test=400,
                                 noise=0.1, kernel_weights_true=None,
                                 n_features_list=None, random_state=None):
    """Utility to generate datasets for the gallery of examples."""
    from .kernel_ridge import generate_dirichlet_samples
    backend = get_backend()

    # Create a few kernel weights if not given.
    if kernel_weights_true is None:
        kernel_weights_true = generate_dirichlet_samples(
            n_targets, n_kernels, concentration=[.3],
            random_state=random_state)
        kernel_weights_true = backend.to_numpy(kernel_weights_true)

    if n_features_list is None:
        n_features_list = np.full(n_kernels, fill_value=1000)

    # Then, generate a random dataset, using the arbitrary scalings.
    Xs_train, Xs_test = [], []
    Y_train, Y_test = None, None
    for ii in range(n_kernels):
        n_features = n_features_list[ii]

        X_train = backend.randn(n_samples_train, n_features)
        X_test = backend.randn(n_samples_test, n_features)
        X_train -= X_train.mean(0)
        X_test -= X_test.mean(0)
        Xs_train.append(X_train)
        Xs_test.append(X_test)

        weights = backend.randn(n_features, n_targets) / n_features
        weights *= backend.asarray_like(kernel_weights_true[:, ii],
                                        ref=weights) ** 0.5

        if ii == 0:
            Y_train = X_train @ weights
            Y_test = X_test @ weights
        else:
            Y_train += X_train @ weights
            Y_test += X_test @ weights

    std = Y_train.std(0)[None]
    Y_train /= std
    Y_test /= std

    Y_train += backend.randn(n_samples_train, n_targets) * noise
    Y_test += backend.randn(n_samples_test, n_targets) * noise
    Y_train -= Y_train.mean(0)
    Y_test -= Y_test.mean(0)

    # Concatenate the feature spaces.
    X_train = backend.asarray(backend.concatenate(Xs_train, 1),
                              dtype="float32")
    X_test = backend.asarray(backend.concatenate(Xs_test, 1), dtype="float32")

    return (X_train, X_test, Y_train, Y_test, kernel_weights_true,
            n_features_list)
