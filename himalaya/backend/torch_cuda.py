"""Used for testing on GPU."""
import torch
from .torch import *  # noqa


def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs).cuda()


def rand(*args, **kwargs):
    return torch.rand(*args, **kwargs).cuda()
