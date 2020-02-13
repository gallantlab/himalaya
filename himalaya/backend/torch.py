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


argmax = torch.argmax
assert_array_equal = torch.testing.assert_allclose
randn = torch.randn


def max(*args, **kwargs):
    return torch.max(*args, **kwargs).values
