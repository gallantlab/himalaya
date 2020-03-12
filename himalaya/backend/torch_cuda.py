"""Used for testing on GPU."""

from .torch import *  # noqa
import torch

try:
    torch.arange(1).cuda()
except AssertionError as error:
    try:
        import pytest
        pytest.skip("Torch not compiled with CUDA enabled.")
    except ImportError:
        pass
    raise AssertionError("Torch not compiled with CUDA enabled.") from error

###############################################################################


def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs).cuda()


def rand(*args, **kwargs):
    return torch.rand(*args, **kwargs).cuda()


def asarray(x, dtype=None, device="cuda"):
    if dtype is None:
        if isinstance(x, torch.Tensor):
            dtype = x.dtype
        if hasattr(x, "dtype") and hasattr(x.dtype, "name"):
            dtype = x.dtype.name
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if device is None and isinstance(x, torch.Tensor):
        device = x.device
    return torch.as_tensor(x, dtype=dtype, device=device)
