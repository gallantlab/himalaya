from functools import partial

import torch


def apply_argmax(array, argmax, axis):
    """Apply precomputed argmax indices in multi dimension arrays

    array[np.argmax(array)] works fine in dimension 1, but not in higher ones.
    This function tackles this issue.

    Example
    -------
    >>> import torch
    >>> array = torch.randn(10, 4, 8)
    >>> argmax = torch.argmax(array, axis=1)
    >>> max_ = apply_argmax(array, argmax, axis=1)
    >>> assert torch.all(max_ == torch.max(array, axis=1).values)
    """
    argmax = argmax.unsqueeze(dim=axis)
    max_ = torch.gather(array, dim=axis, index=argmax)
    return torch.squeeze(max_, dim=axis)


def std_float64(X, axis=None, demean=True, keepdims=False):
    """Compute the standard deviation of X with double precision,
    and cast back the result to original dtype.
    """
    X_64 = torch.as_tensor(X, dtype=torch.float64)
    X_std = (X_64 ** 2).sum(dim=axis, dtype=torch.float64)
    if demean:
        X_std -= X_64.sum(axis, dtype=torch.float64) ** 2 / X.shape[axis]
    X_std = X_std ** .5
    X_std /= (X.shape[axis] ** .5)

    X_std = torch.as_tensor(X_std, dtype=X.dtype, device=X.device)
    if keepdims:
        X_std = X_std.unsqueeze(dim=axis)

    return X_std


###############################################################################

argmax = torch.argmax
assert_allclose = torch.testing.assert_allclose
randn = torch.randn
rand = torch.rand
matmul = torch.matmul
stack = torch.stack
zeros = torch.zeros
abs = torch.abs
sum = torch.sum
sqrt = torch.sqrt
any = torch.any
all = torch.all
nan = torch.tensor(float('nan'))
inf = torch.tensor(float('inf'))
isnan = torch.isnan
isinf = torch.isinf
logspace = torch.logspace
eye = torch.eye
concatenate = torch.cat
bool = torch.bool
float32 = torch.float32
float64 = torch.float64
eigh = partial(torch.symeig, eigenvectors=True)
svd = torch.svd


def asarray(x, dtype=None):
    if dtype is None:
        if isinstance(x, torch.Tensor):
            dtype = x.dtype
        if hasattr(x.dtype, "name"):
            dtype = x.dtype.name
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if isinstance(x, torch.Tensor):
        device = x.device
    else:
        device = None
    return torch.as_tensor(x, dtype=dtype, device=device)


def norm(x, ord=None, axis=None, keepdims=False):
    return torch.norm(x, p=ord, dim=axis, keepdim=keepdims)


def copy(x):
    return x.clone()


def transpose(a, axes=None):
    if axes is None:
        return a.t()
    else:
        return a.permute(*axes)


def max(*args, **kwargs):
    res = torch.max(*args, **kwargs)
    if isinstance(res, torch.Tensor):
        return res
    else:
        return res.values


def zeros_like(array, shape=None, dtype=None):
    """Add a shape parameter in zeros_like."""
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype
    return torch.zeros(shape, dtype=dtype, device=array.device,
                       layout=array.layout)


def ones_like(array, shape=None, dtype=None):
    """Add a shape parameter in ones_like."""
    if shape is None:
        shape = array.shape
    if dtype is None:
        dtype = array.dtype
    return torch.ones(shape, dtype=dtype, device=array.device,
                      layout=array.layout)
