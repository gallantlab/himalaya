"""The "torch" CPU backend, based on PyTorch.

To use this backend, call ``himalaya.backend.set_backend("torch")``.
"""
from functools import partial

try:
    import torch
except ImportError as error:
    import sys
    if "pytest" in sys.modules:  # if run through pytest
        import pytest
        pytest.skip("PyTorch not installed.")
    raise ImportError("PyTorch not installed.") from error

from ._utils import _dtype_to_str
from ._utils import _add_error_message

###############################################################################


def apply_argmax(array, argmax, axis):
    """Apply precomputed argmax indices in multi dimension arrays

    array[np.argmax(array)] works fine in dimension 1, but not in higher ones.
    This function tackles this issue.

    Examples
    --------
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


def mean_float64(X, axis=None, keepdims=False):
    """Compute the mean of X with double precision,
    and cast back the result to original dtype.
    """
    X_mean = X.sum(axis, dtype=torch.float64) / X.shape[axis]

    X_mean = torch.as_tensor(X_mean, dtype=X.dtype, device=X.device)
    if keepdims:
        X_mean = X_mean.unsqueeze(dim=axis)
    return X_mean


###############################################################################

name = "torch"
argmax = torch.argmax
randn = torch.randn
rand = torch.rand
matmul = torch.matmul
stack = torch.stack
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
concatenate = torch.cat
bool = torch.bool
int32 = torch.int32
float32 = torch.float32
float64 = torch.float64
log = torch.log
exp = torch.exp
arange = torch.arange
unique = torch.unique
einsum = torch.einsum
tanh = torch.tanh
power = torch.pow
prod = torch.prod
sign = torch.sign
clip = torch.clamp
finfo = torch.finfo
eye = torch.eye


def atleast_1d(array):
    array = asarray(array)
    if array.ndim == 0:
        array = array[None]
    return array


def flip(array, axis=0):
    return torch.flip(array, dims=[axis])


def sort(array, axis=-1):
    return torch.sort(array, dim=axis).values


def diagonal_view(array, axis1=0, axis2=1):
    """Return a view of the array diagonal."""
    return torch.diagonal(array, 0, dim1=axis1, dim2=axis2)


def to_numpy(array):
    try:
        return array.cpu().numpy()
    except AttributeError:
        return array


def to_cpu(array):
    return array.cpu()


def to_gpu(array, device=None):
    return array


def is_in_gpu(array):
    return array.device.type == "cuda"


def isin(x, y):
    import numpy as np  # XXX
    np_result = np.isin(x.cpu().numpy(), y.cpu().numpy())
    return asarray(np_result, dtype=torch.bool, device=x.device)


def searchsorted(x, y):
    import numpy as np  # XXX
    np_result = np.searchsorted(x.cpu().numpy(), y.cpu().numpy())
    return asarray(np_result, dtype=torch.int64, device=x.device)


def flatnonzero(x):
    return torch.nonzero(torch.flatten(x), as_tuple=True)[0]


def asarray(x, dtype=None, device="cpu"):
    if dtype is None:
        if isinstance(x, torch.Tensor):
            dtype = x.dtype
        if hasattr(x, "dtype") and hasattr(x.dtype, "name"):
            dtype = x.dtype.name
    if dtype is not None:
        dtype = _dtype_to_str(dtype)
        dtype = getattr(torch, dtype)
    if device is None and isinstance(x, torch.Tensor):
        device = x.device

    try:
        tensor = torch.as_tensor(x, dtype=dtype, device=device)
    except Exception:
        import numpy as np
        if torch.is_tensor(x) and x.device != "cpu":
            x = x.cpu()
        array = np.asarray(x, dtype=_dtype_to_str(dtype))
        tensor = torch.as_tensor(array, dtype=dtype, device=device)
    return tensor


def asarray_like(x, ref):
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


def norm(x, ord="fro", axis=None, keepdims=False):
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


def min(*args, **kwargs):
    res = torch.min(*args, **kwargs)
    if isinstance(res, torch.Tensor):
        return res
    else:
        return res.values


def zeros(shape, dtype="float32", device="cpu"):
    if isinstance(shape, int):
        shape = (shape, )
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return torch.zeros(shape, dtype=dtype, device=device)


def zeros_like(array, shape=None, dtype=None, device=None):
    """Add a shape parameter in zeros_like."""
    if shape is None:
        shape = array.shape
    if isinstance(shape, int):
        shape = (shape, )
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if dtype is None:
        dtype = array.dtype
    if device is None:
        device = array.device
    return torch.zeros(shape, dtype=dtype, device=device, layout=array.layout)


def ones_like(array, shape=None, dtype=None, device=None):
    """Add a shape parameter in ones_like."""
    if shape is None:
        shape = array.shape
    if isinstance(shape, int):
        shape = (shape, )
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if dtype is None:
        dtype = array.dtype
    if device is None:
        device = array.device
    return torch.ones(shape, dtype=dtype, device=device, layout=array.layout)


def full_like(array, fill_value, shape=None, dtype=None, device=None):
    """Add a shape parameter in full_like."""
    if shape is None:
        shape = array.shape
    if isinstance(shape, int):
        shape = (shape, )
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if dtype is None:
        dtype = array.dtype
    if device is None:
        device = array.device
    return torch.full(shape, fill_value, dtype=dtype, device=device,
                      layout=array.layout)


def check_arrays(*all_inputs):
    """Change all inputs into Tensors (or list of Tensors) using the same
    precision and device as the first one. Some tensors can be None.
    """
    all_tensors = []
    all_tensors.append(asarray(all_inputs[0]))
    dtype = all_tensors[0].dtype
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


try:
    svd = torch.linalg.svd
except AttributeError:
    # torch.__version__ < 1.8

    def svd(X, full_matrices=True):
        U, s, V = torch.svd(X, some=not full_matrices)
        Vh = V.transpose(-2, -1)
        return U, s, Vh


try:
    eigh = _add_error_message(
        torch.linalg.eigh,
        msg=(f"The eigenvalues decomposition failed on backend {name}. You may"
             " try using `diagonalize_method='svd'`, or `solver_params={"
             "'diagonalize_method': 'svd'}` if called through the class API."))

except AttributeError:
    # torch.__version__ < 1.8
    eigh = partial(torch.symeig, eigenvectors=True)
