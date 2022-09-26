"""The "cupy" GPU backend, based on CuPy.

To use this backend, call ``himalaya.backend.set_backend("cupy")``.
"""
try:
    import cupy
except ImportError as error:
    import sys
    if "pytest" in sys.modules:  # if run through pytest
        import pytest
        pytest.skip("Cupy not installed.")
    raise ImportError("Cupy not installed.") from error

from ._utils import warn_if_not_float32

###############################################################################


def apply_argmax(array, argmax, axis):
    """Apply precomputed argmax indices in multi dimension arrays

    array[np.argmax(array)] works fine in dimension 1, but not in higher ones.
    This function extends it to higher dimensions.

    Examples
    --------
    >>> import cupy
    >>> array = cupy.random.randn(10, 4, 8)
    >>> argmax = cupy.argmax(array, axis=1)
    >>> max_ = apply_argmax(array, argmax, axis=1)
    >>> assert cupy.all(max_ == cupy.max(array, axis=1))
    """
    argmax = cupy.expand_dims(argmax, axis=axis)
    max_ = cupy.take_along_axis(array, argmax, axis=axis)
    return cupy.take(max_, 0, axis=axis)


def std_float64(array, axis=None, demean=True, keepdims=False):
    """Compute the standard deviation of X with double precision,
    and cast back the result to original dtype.
    """
    return array.std(axis, dtype=cupy.float64,
                     keepdims=keepdims).astype(array.dtype, copy=False)


def mean_float64(array, axis=None, keepdims=False):
    """Compute the mean of X with double precision,
    and cast back the result to original dtype.
    """
    return array.mean(axis, dtype=cupy.float64,
                      keepdims=keepdims).astype(array.dtype, copy=False)


###############################################################################

name = "cupy"
argmax = cupy.argmax
max = cupy.max
min = cupy.min
abs = cupy.abs
randn = cupy.random.randn
rand = cupy.random.rand
matmul = cupy.matmul
transpose = cupy.transpose
stack = cupy.stack
concatenate = cupy.concatenate
sum = cupy.sum
sqrt = cupy.sqrt
any = cupy.any
all = cupy.all
nan = cupy.nan
inf = cupy.inf
isnan = cupy.isnan
isinf = cupy.isinf
logspace = cupy.logspace
copy = cupy.copy
bool = cupy.bool_
float32 = cupy.float32
float64 = cupy.float64
int32 = cupy.int32
eigh = cupy.linalg.eigh
norm = cupy.linalg.norm
log = cupy.log
exp = cupy.exp
arange = cupy.arange
flatnonzero = cupy.flatnonzero
unique = cupy.unique
einsum = cupy.einsum
tanh = cupy.tanh
power = cupy.power
prod = cupy.prod
zeros = cupy.zeros
sign = cupy.sign
clip = cupy.clip
sort = cupy.sort
flip = cupy.flip
atleast_1d = cupy.atleast_1d
finfo = cupy.finfo
eye = cupy.eye


def diagonal_view(array, axis1=0, axis2=1):
    """Return a view of the array diagonal."""
    return cupy.diagonal(array, 0, axis1=axis1, axis2=axis2)


def to_numpy(array):
    return cupy.asnumpy(array)


def isin(x, y):
    import numpy as np  # XXX
    np_result = np.isin(to_numpy(x), to_numpy(y))
    return asarray(np_result, dtype=cupy.bool)


def searchsorted(x, y):
    import numpy as np  # XXX
    np_result = np.searchsorted(to_numpy(x), to_numpy(y))
    return asarray(np_result, dtype=cupy.int64)


def zeros_like(array, shape=None, dtype=None, device=None):
    """Add a shape parameter in zeros_like."""
    xp = cupy.get_array_module(array)
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype
    if device == "cpu":
        import numpy as xp
    return xp.zeros(shape, dtype=dtype)


def ones_like(array, shape=None, dtype=None, device=None):
    """Add a shape parameter in ones_like."""
    xp = cupy.get_array_module(array)
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype
    if device == "cpu":
        import numpy as xp
    return xp.ones(shape, dtype=dtype)


def full_like(array, fill_value, shape=None, dtype=None, device=None):
    """Add a shape parameter in full_like."""
    xp = cupy.get_array_module(array)
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype
    if device == "cpu":
        import numpy as xp
    return xp.full(shape, fill_value, dtype=dtype)


def to_cpu(array):
    return cupy.asnumpy(array)


def to_gpu(array, device=None):
    return cupy.asarray(array)


def is_in_gpu(array):
    return getattr(array, "device", None) is not None


def asarray(a, dtype=None, order=None, device=None):
    if device == "cpu":
        import numpy as np
        return np.asarray(cupy.asnumpy(a), dtype, order)
    else:
        return cupy.asarray(a, dtype, order)


def asarray_like(x, ref):
    xp = cupy.get_array_module(ref)
    return xp.asarray(x, dtype=ref.dtype)


def check_arrays(*all_inputs):
    """Change all inputs into arrays (or list of arrays) using the same
    precision as the first one. Some arrays can be None.
    """
    all_arrays = []
    all_arrays.append(asarray(all_inputs[0]))
    dtype = all_arrays[0].dtype
    warn_if_not_float32(dtype)
    for tensor in all_inputs[1:]:
        if tensor is None:
            pass
        elif isinstance(tensor, list):
            tensor = [asarray(tt, dtype=dtype) for tt in tensor]
        else:
            tensor = asarray(tensor, dtype=dtype)
        all_arrays.append(tensor)
    return all_arrays


def svd(X, full_matrices=True):
    if X.ndim == 2:
        return cupy.linalg.svd(X, full_matrices=full_matrices)
    elif X.ndim == 3:
        UsV_list = [
            cupy.linalg.svd(Xi, full_matrices=full_matrices) for Xi in X
        ]
        return map(cupy.stack, zip(*UsV_list))
    else:
        raise NotImplementedError()
