"""The "torch_mps" GPU backend, based on PyTorch.

To use this backend, call ``himalaya.backend.set_backend("torch_mps")``.
"""
from .torch import *  # noqa
import torch
import warnings

if not torch.backends.mps.is_available():
    import sys
    if "pytest" in sys.modules:  # if run through pytest
        import pytest
        pytest.skip("PyTorch with MPS is not available.")
    raise RuntimeError("PyTorch with MPS is not available.")

from ._utils import _dtype_to_str

###############################################################################

name = "torch_mps"


def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs).to("mps")


def rand(*args, **kwargs):
    return torch.rand(*args, **kwargs).to("mps")


def asarray(x, dtype=None, device="mps"):
    if dtype is None:
        if isinstance(x, torch.Tensor):
            dtype = x.dtype
        if hasattr(x, "dtype") and hasattr(x.dtype, "name"):
            dtype = x.dtype.name
    if dtype is not None:
        dtype = _dtype_to_str(dtype)
        dtype = _check_dtype_torch_mps(dtype)
        dtype = getattr(torch, dtype)
    if device is None:
        if isinstance(x, torch.Tensor):
            device = x.device
        else:
            device = "mps"
    try:
        tensor = torch.as_tensor(x, dtype=dtype, device=device)
    except Exception:
        import numpy as np
        array = np.asarray(x, dtype=_dtype_to_str(dtype))
        tensor = torch.as_tensor(array, dtype=dtype, device=device)
    return tensor


_already_warned = [False]


def _check_dtype_torch_mps(dtype):
    """Warn that X will be cast from float64 to float32 and return the correct dtype"""
    if _dtype_to_str(dtype) == "float64":
        if not _already_warned[0]:  # avoid warning multiple times
            warnings.warn(
                f"GPU backend torch_mps requires single "
                f"precision floats (float32), got input in {dtype}. "
                "Data will be automatically cast to float32", UserWarning)
            _already_warned[0] = True
        return "float32"
    return dtype


def check_arrays(*all_inputs):
    """Change all inputs into Tensors (or list of Tensors) using the same
    precision and device as the first one. Some tensors can be None. float64 tensors
    are automatically cast to float32 due to the requirement of torch MPS backend.
    """
    all_tensors = []
    all_tensors.append(asarray(all_inputs[0]))
    dtype = all_tensors[0].dtype
    dtype = _check_dtype_torch_mps(dtype)
    device = all_tensors[0].device
    for tensor in all_inputs[1:]:
        if tensor is None:
            pass
        elif isinstance(tensor, list):
            tensor = [asarray(tt, dtype=dtype, device=device) for tt in tensor]
        else:
            tensor = asarray(tensor, dtype=dtype, device=device)
        all_tensors.append(tensor)
    return all_tensors


def zeros(shape, dtype="float32", device="mps"):
    if isinstance(shape, int):
        shape = (shape, )
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return torch.zeros(shape, dtype=dtype, device=device)


def to_cpu(array):
    return array.cpu()


def to_gpu(array, device="mps"):
    return asarray(array, device=device)


def std_float64(X, axis=None, demean=True, keepdims=False):
    """Compute the standard deviation of X with double precision on CPU,
    then cast back the result to original dtype on the original device.
    """
    X_64 = torch.as_tensor(X.to("cpu"), dtype=torch.float64, device="cpu")
    X_std = (X_64 ** 2).sum(dim=axis, dtype=torch.float64)
    if demean:
        X_std -= X_64.sum(axis, dtype=torch.float64) ** 2 / X.shape[axis]
    X_std = X_std ** .5
    X_std /= (X.shape[axis] ** .5)

    X_std = torch.as_tensor(X_std, dtype=torch.float32, device=X.device)
    if keepdims:
        X_std = X_std.unsqueeze(dim=axis)

    return X_std


def mean_float64(X, axis=None, keepdims=False):
    """Compute the mean of X with double precision on CPU,
    then cast back the result to original dtype on the original device.
    """
    X_mean = X.to("cpu").sum(axis, dtype=torch.float64) / X.shape[axis]

    X_mean = torch.as_tensor(X_mean, dtype=X.dtype, device=X.device)
    if keepdims:
        X_mean = X_mean.unsqueeze(dim=axis)
    return X_mean


def is_in_gpu(array):
    """Check if array is on MPS device."""
    return array.device.type == "mps"


def eigh(input):
    """Compute eigendecomposition on CPU and move results back to MPS.

    PyTorch's MPS backend doesn't fully support torch.linalg.eigh, so we
    perform the computation on CPU and move the results back to MPS device.
    """
    try:
        # Move to CPU, compute eigendecomposition, then move back to MPS
        input_cpu = input.cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(input_cpu)

        # Move results back to original device (MPS)
        eigenvalues = eigenvalues.to(input.device)
        eigenvectors = eigenvectors.to(input.device)

        return eigenvalues, eigenvectors
    except Exception as e:
        msg = (f"The eigenvalues decomposition failed on backend {name}. You may"
               " try using `diagonalize_method='svd'`, or `solver_params={"
               "'diagonalize_method': 'svd'}` if called through the class API.")
        raise RuntimeError(
            f"{msg}\nOriginal error:\n{type(e).__name__}: {e}")
