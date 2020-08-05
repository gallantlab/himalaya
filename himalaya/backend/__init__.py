import types
import importlib
import warnings

#: List of all available backends.
ALL_BACKENDS = [
    "numpy",
    "cupy",
    "torch",
    "torch_cuda",
]

CURRENT_BACKEND = "numpy"


def set_backend(backend, on_error="raise"):
    """Set the backend using a global variable, and return the backend module.

    Parameters
    ----------
    backend : str or module
        Name or module of the backend.
    on_error : str in {"raise", "warn"}
        Define what is done if the backend fails to be loaded.
        If "warn", this function only warns, and keeps the previous backend.
        If "raise", this function raises on errors.

    Returns
    -------
    module : python module
        Module of the backend.
    """
    global CURRENT_BACKEND

    try:
        if isinstance(backend, types.ModuleType):  # get name from module
            backend = backend.name

        if backend not in ALL_BACKENDS:
            raise ValueError("Unknown backend=%r" % (backend, ))

        module = importlib.import_module(__name__ + "." + backend)
        CURRENT_BACKEND = backend
    except Exception as error:
        if on_error == "raise":
            raise error
        elif on_error == "warn":
            warnings.warn(f"Setting backend to {backend} failed: {str(error)}."
                          f"Falling back to {CURRENT_BACKEND} backend.")
            module = get_backend()
        else:
            raise ValueError('Unknown value on_error=%r' % (on_error, ))

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
