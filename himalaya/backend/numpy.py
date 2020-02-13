import numpy as np


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


###############################################################################

argmax = np.argmax
assert_allclose = np.testing.assert_allclose
max = np.max
abs = np.abs
randn = np.random.randn
rand = np.random.rand
matmul = np.matmul
norm = np.linalg.norm
transpose = np.transpose
stack = np.stack
concatenate = np.concatenate
sum = np.sum
zeros = np.zeros
sqrt = np.sqrt
any = np.any
all = np.all
isnan = np.isnan
isinf = np.isinf
logspace = np.logspace
eye = np.eye
copy = np.copy
bool = np.bool


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
