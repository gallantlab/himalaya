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
isnan = torch.isnan
isinf = torch.isinf
logspace = torch.logspace
eye = torch.eye
concatenate = torch.cat
bool = torch.bool


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
