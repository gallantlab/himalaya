import numbers

import numpy as np

try:
    from scipy.sparse import issparse
except ImportError:

    def issparse(X):
        return False


from .backend import get_backend


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def check_array(array, accept_sparse=False, dtype=["float32", "float64"],
                copy=False, force_all_finite=True, ndim=2,
                ensure_min_samples=1, ensure_min_features=1):
    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    dtype : str, list of str or None
        Data type of result. If None, the dtype of the input is preserved.
        If dtype is a list of str, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accept both np.inf and np.nan in array.
        - 'allow-nan': accept only np.nan values in array. Values cannot
          be infinite.

        For object dtyped data, only np.nan is checked and not np.inf.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    ndim : int, list of int, or None (default=2)
        If not None, list the accepted number of dimension of the array.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    Returns
    -------
    array_converted : object
        The converted and validated array.
    """
    backend = get_backend()

    ########################
    # decide to change dtype

    dtype_input = _get_string_dtype(array)

    if dtype is None:
        change_dtype = False
    elif isinstance(dtype, str):
        if dtype not in ("float32", "float64"):
            raise ValueError("Unknown dtype=%r" % dtype)
        change_dtype = dtype_input != dtype
    elif isinstance(dtype, (list, tuple)):
        change_dtype = dtype_input not in dtype
        if change_dtype:
            dtype = dtype[0]
    else:
        raise ValueError("Unknown dtype=%r" % dtype)

    #############
    # sparse case
    if issparse(array):
        changed_format = False

        if isinstance(accept_sparse, str):
            accept_sparse = [accept_sparse]
        if accept_sparse is False:
            raise TypeError('A sparse matrix was passed, but dense '
                            'data is required. Use X.toarray() to '
                            'convert to a dense numpy array.')
        elif isinstance(accept_sparse, (list, tuple)):
            if len(accept_sparse) == 0:
                raise ValueError("When providing 'accept_sparse' "
                                 "as a tuple or list, it must contain at "
                                 "least one string value.")
            # ensure correct sparse format
            if array.format not in accept_sparse:
                array = array.asformat(accept_sparse[0])
                changed_format = True
        elif accept_sparse is not True:
            raise ValueError("Parameter 'accept_sparse' should be a string, "
                             "boolean or list of strings. You provided "
                             "'accept_sparse={}'.".format(accept_sparse))

        if change_dtype:
            array = array.astype(dtype)
        elif copy and not changed_format:
            array = array.copy()

        _assert_all_finite(array.data, force_all_finite)

    ############
    # dense case
    else:
        if ndim is not None and array.ndim not in np.atleast_1d(ndim):
            raise ValueError("Found array with ndim %d, expected %d." %
                             (array.ndim, ndim))

        if change_dtype:
            array = backend.asarray(array, dtype=dtype)
        elif copy:
            array = backend.copy(array)

        _assert_all_finite(array, force_all_finite)

    #####################
    # check minimum sizes
    if ensure_min_samples > 0:
        n_samples = array.shape[0]
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required." %
                             (n_samples, array.shape, ensure_min_samples))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required." %
                             (n_features, array.shape, ensure_min_features))

    return array


def _assert_all_finite(X, force_all_finite):
    backend = get_backend()

    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         '. Got {!r} instead'.format(force_all_finite))

    if not force_all_finite:
        return
    if backend.any(backend.isinf(X)):
        raise ValueError("Input contains infinity values")
    if force_all_finite != 'allow_nan' and backend.any(backend.isnan(X)):
        raise ValueError("Input contains NaNs values")


def _get_string_dtype(array):
    dtype = getattr(array, "dtype", None)
    if dtype is None:
        return None

    if hasattr(dtype, "name"):
        return dtype.name
    else:
        return NotImplementedError()
