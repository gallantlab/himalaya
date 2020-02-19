import numpy as np
import scipy.linalg


def apply_argmax(array, argmax, axis):
    """Apply precomputed argmax indices in multi dimension arrays

    array[np.argmax(array)] works fine in dimension 1, but not in higher ones.
    This function extends it to higher dimensions.

    Example
    -------
    >>> import numpy as np
    >>> array = np.random.randn(10, 4, 8)
    >>> argmax = np.argmax(array, axis=1)
    >>> max_ = apply_argmax(array, argmax, axis=1)
    >>> assert np.all(max_ == np.max(array, axis=1))
    """
    argmax = np.expand_dims(argmax, axis=axis)
    max_ = np.take_along_axis(array, argmax, axis=axis)
    return np.take(max_, 0, axis=axis)


def std_float64(array, axis=None, demean=True, keepdims=False):
    """Compute the standard deviation of X with double precision,
    and cast back the result to original dtype.
    """
    return array.std(axis, dtype=np.float64,
                     keepdims=keepdims).astype(array.dtype, copy=False)


def mean_float64(array, axis=None, keepdims=False):
    """Compute the mean of X with double precision,
    and cast back the result to original dtype.
    """
    return array.mean(axis, dtype=np.float64,
                      keepdims=keepdims).astype(array.dtype, copy=False)


###############################################################################

argmax = np.argmax
assert_allclose = np.testing.assert_allclose
max = np.max
abs = np.abs
randn = np.random.randn
rand = np.random.rand
matmul = np.matmul
transpose = np.transpose
stack = np.stack
concatenate = np.concatenate
sum = np.sum
zeros = np.zeros
sqrt = np.sqrt
any = np.any
all = np.all
nan = np.nan
inf = np.inf
isnan = np.isnan
isinf = np.isinf
logspace = np.logspace
eye = np.eye
copy = np.copy
bool = np.bool
float32 = np.float32
float64 = np.float64
int32 = np.int32
asarray = np.asarray
eigh = scipy.linalg.eigh
svd = scipy.linalg.svd
norm = scipy.linalg.norm
log10 = np.log10
arange = np.arange
flatnonzero = np.flatnonzero
isin = np.isin
searchsorted = np.searchsorted
sqrt = np.sqrt
unique = np.unique


def to_numpy(array):
    return array


def zeros_like(array, shape=None, dtype=None):
    """Add a shape parameter in zeros_like."""
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype
    return np.zeros(shape, dtype=dtype)


def ones_like(array, shape=None, dtype=None):
    """Add a shape parameter in ones_like."""
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype
    return np.ones(shape, dtype=dtype)


def full_like(array, fill_value, shape=None, dtype=None):
    """Add a shape parameter in full_like."""
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype
    return np.full(shape, fill_value, dtype=dtype)


def asarray_like(x, ref):
    return np.asarray(x, dtype=ref.dtype)


def check_arrays(*all_inputs):
    """Change all inputs into arrays (or list of arrays) using the same
    precision as the first one. Some arrays can be None.
    """
    all_arrays = []
    all_arrays.append(np.asarray(all_inputs[0]))
    for tensor in all_inputs[1:]:
        if tensor is None:
            pass
        elif isinstance(tensor, list):
            tensor = [
                np.asarray(tt, dtype=all_arrays[0].dtype) for tt in tensor
            ]
        else:
            tensor = np.asarray(tensor, dtype=all_arrays[0].dtype)
        all_arrays.append(tensor)
    return all_arrays
