import numbers

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

    # Auto-adjust precision for torch_mps backend due to float32 conversion
    if backend.name == "torch_mps" and decimal > 4:
        import warnings
        warnings.warn(
            f"Reducing precision from decimal={decimal} to decimal=4 for "
            "torch_mps backend due to float32 conversion limitations",
            UserWarning
        )
        decimal = 4

    x = backend.to_numpy(x)
    y = backend.to_numpy(y)
    return np.testing.assert_array_almost_equal(x, y, decimal=decimal,
                                                err_msg=err_msg,
                                                verbose=verbose)


def generate_multikernel_dataset(n_kernels=4, n_targets=500,
                                 n_samples_train=1000, n_samples_test=400,
                                 noise=0.1, kernel_weights=None,
                                 n_features_list=None, random_state=None):
    """Utility to generate datasets for the gallery of examples.

    Parameters
    ----------
    n_kernels : int
        Number of kernels.
    n_targets : int
        Number of targets.
    n_samples_train : int
        Number of samples in the training set.
    n_samples_test : int
        Number of sample in the testing set.
    noise : float > 0
        Scale of the Gaussian white noise added to the targets.
    kernel_weights : array of shape (n_targets, n_kernels) or None
        Kernel weights used in the prediction of the targets.
        If None, generate random kernel weights from a Dirichlet distribution.
    n_features_list : list of int of length (n_kernels, ) or None
        Number of features in each kernel. If None, use 1000 features for each.
    random_state : int, or None
        Random generator seed use to generate the true kernel weights.

    Returns
    -------
    X_train : array of shape (n_samples_train, n_features)
        Training features.
    X_test : array of shape (n_samples_test, n_features)
        Testing features.
    Y_train : array of shape (n_samples_train, n_targets)
        Training targets.
    Y_test : array of shape (n_samples_test, n_targets)
        Testing targets.
    kernel_weights : array of shape (n_targets, n_kernels)
        Kernel weights in the prediction of the targets.
    n_features_list : list of int of length (n_kernels, )
        Number of features in each kernel.
    """
    from .kernel_ridge import generate_dirichlet_samples
    backend = get_backend()

    # Create a few kernel weights if not given.
    if kernel_weights is None:
        kernel_weights = generate_dirichlet_samples(n_targets, n_kernels,
                                                    concentration=[.3],
                                                    random_state=random_state)
        kernel_weights = backend.to_numpy(kernel_weights)

    if n_features_list is None:
        n_features_list = np.full(n_kernels, fill_value=1000)

    rng = check_random_state(random_state)

    # Then, generate a random dataset, using the arbitrary scalings.
    Xs_train, Xs_test = [], []
    Y_train, Y_test = None, None
    for ii in range(n_kernels):
        n_features = n_features_list[ii]

        X_train = rng.randn(n_samples_train, n_features)
        X_test = rng.randn(n_samples_test, n_features)
        X_train -= X_train.mean(0)
        X_test -= X_test.mean(0)
        Xs_train.append(X_train)
        Xs_test.append(X_test)

        weights = rng.randn(n_features, n_targets) / n_features
        weights *= kernel_weights[:, ii] ** 0.5

        if ii == 0:
            Y_train = X_train @ weights
            Y_test = X_test @ weights
        else:
            Y_train += X_train @ weights
            Y_test += X_test @ weights

    std = Y_train.std(0)[None]
    Y_train /= std
    Y_test /= std

    Y_train += rng.randn(n_samples_train, n_targets) * noise
    Y_test += rng.randn(n_samples_test, n_targets) * noise
    Y_train -= Y_train.mean(0)
    Y_test -= Y_test.mean(0)

    # Concatenate the feature spaces.
    X_train = backend.asarray(np.concatenate(Xs_train, 1), dtype="float32")
    X_test = backend.asarray(np.concatenate(Xs_test, 1), dtype="float32")
    Y_train = backend.asarray(Y_train, dtype="float32")
    Y_test = backend.asarray(Y_test, dtype="float32")
    kernel_weights = backend.asarray(kernel_weights, dtype="float32")

    return X_train, X_test, Y_train, Y_test, kernel_weights, n_features_list


def _batch_or_skip(array, batch, axis):
    """Apply a batch on given axis, or skip if the dimension is equal to 1."""
    skip = (array is None or isinstance(array, numbers.Number)
            or array.ndim == 0 or array.shape[axis] == 1)  # noqa
    if skip:
        return array
    else:
        # Not general but works with slices in `batch`.
        if axis == 0:
            return array[batch]
        elif axis == 1:
            return array[:, batch]
        else:
            raise NotImplementedError()


def skip_torch_mps_precision_checks(backend, estimator, check, 
                                   precision_sensitive_checks=None):
    """Skip sklearn checks that fail due to torch_mps float32 precision limitations.
    
    This utility function provides a centralized way to handle precision-sensitive
    sklearn estimator checks when using the torch_mps backend, which uses float32
    precision that can cause small numerical differences exceeding sklearn's strict
    tolerance requirements.
    
    Parameters
    ----------
    backend : object
        The current backend instance
    estimator : object
        The estimator being tested
    check : callable
        The sklearn check function (partial with check name)
    precision_sensitive_checks : dict, optional
        Dictionary mapping estimator names to lists of check names that should be
        skipped due to precision issues. If None, uses default set.
        
    Returns
    -------
    bool
        True if the check should be skipped, False otherwise
        
    Examples
    --------
    >>> # In test function:
    >>> if skip_torch_mps_precision_checks(backend, estimator, check):
    ...     pytest.skip("torch_mps precision limitation")
    >>> check(estimator)
    """
    # Default precision-sensitive checks if not provided
    if precision_sensitive_checks is None:
        precision_sensitive_checks = {
            'KernelRidge_': [
                'check_methods_subset_invariance',
                'check_sample_weight_equivalence_on_dense_data', 
                'check_sample_weight_equivalence_on_sparse_data'
            ],
            'KernelRidgeCV_': [
                'check_methods_subset_invariance'
            ],
            'Kernelizer_': [
                'check_methods_subset_invariance'
            ]
        }
    
    # Only apply to torch_mps backend
    # Use getattr for safety in case backend doesn't have name attribute
    if getattr(backend, 'name', None) != "torch_mps":
        return False
        
    # Check if we have a callable check with a function name
    if not hasattr(check, 'func'):
        return False
        
    check_name = check.func.__name__
    estimator_name = estimator.__class__.__name__
    
    # Check if this estimator/check combination should be skipped
    if estimator_name in precision_sensitive_checks:
        return check_name in precision_sensitive_checks[estimator_name]
        
    return False
