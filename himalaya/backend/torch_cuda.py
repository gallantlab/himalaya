"""Used for testing on GPU."""

from .torch import *  # noqa
import torch

try:
    torch.arange(1).cuda()
except AssertionError as error:
    import sys
    if "pytest" in sys.modules:  # if run through pytest
        import pytest
        pytest.skip("PyTorch not compiled with CUDA enabled.")
    raise AssertionError("PyTorch not compiled with CUDA enabled.") from error

from ._utils import _dtype_to_str

###############################################################################

name = "torch_cuda"


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
    if dtype is not None:
        dtype = _dtype_to_str(dtype)
        dtype = getattr(torch, dtype)
    if device is None:
        if isinstance(x, torch.Tensor):
            device = x.device
        else:
            device = "cuda"
    try:
        tensor = torch.as_tensor(x, dtype=dtype, device=device)
    except Exception:
        import numpy as np
        array = np.asarray(x, dtype=_dtype_to_str(dtype))
        tensor = torch.as_tensor(array, dtype=dtype, device=device)
    return tensor


def zeros(shape, dtype="float32", device="cuda"):
    if isinstance(shape, int):
        shape = (shape, )
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return torch.zeros(shape, dtype=dtype, device=device)


def to_cpu(array):
    return array.cpu()


def to_gpu(array, device="cuda"):
    return asarray(array, device=device)
