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
