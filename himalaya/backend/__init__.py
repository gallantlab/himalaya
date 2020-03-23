import types
import importlib

ALL_BACKENDS = [
    "numpy",
    "cupy",
    "torch",
    "torch_cuda",
]

CURRENT_BACKEND = "numpy"


def set_backend(backend):
    """Set the backend using a global variable, and return the backend module.

    Parameters
    ----------
    backend : str or module
        Name or module of the backend.

    Returns
    -------
    module : python module
        Module of the backend.
    """
    global CURRENT_BACKEND

    if isinstance(backend, types.ModuleType):  # get backend name from module
        backend = backend.name

    if backend not in ALL_BACKENDS:
        raise ValueError("Unknown backend=%r" % (backend, ))

    module = importlib.import_module(__name__ + "." + backend)
    CURRENT_BACKEND = backend
    return module


def get_backend():
    """Get the current backend module.

    Returns
    -------
    module : python module
        Module of the backend.
    """
    module = importlib.import_module(__name__ + "." + CURRENT_BACKEND)
    return module


def _dtype_to_str(dtype):
    """Cast dtype to string, such as "float32", or "float64"."""
    if hasattr(dtype, "name"):  # works for numpy and cupy
        return dtype.name
    elif "float32" in str(dtype):  # works for torch
        return "float32"
    elif "float64" in str(dtype):
        return "float64"
    else:
        raise NotImplementedError()
