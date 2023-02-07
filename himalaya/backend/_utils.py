import types
import importlib
import warnings
from functools import wraps

ALL_BACKENDS = [
    "numpy",
    "cupy",
    "torch",
    "torch_cuda",
]

CURRENT_BACKEND = "numpy"

MATCHING_CPU_BACKEND = {
    "numpy": "numpy",
    "cupy": "numpy",
    "torch": "torch",
    "torch_cuda": "torch",
}


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

        module = importlib.import_module(__package__ + "." + backend)
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
    module = importlib.import_module(__package__ + "." + CURRENT_BACKEND)
    return module


def _dtype_to_str(dtype):
    """Cast dtype to string, such as "float32", or "float64"."""
    if isinstance(dtype, str):
        return dtype
    elif hasattr(dtype, "name"):  # works for numpy and cupy
        return dtype.name
    elif "torch." in str(dtype):  # works for torch
        return str(dtype)[6:]
    elif dtype is None:
        return None
    else:
        raise NotImplementedError()


def force_cpu_backend(func):
    """Decorator to force the use of a CPU backend."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # skip if the object does not force cpu use
        if not hasattr(args[0], "force_cpu") or not args[0].force_cpu:
            return func(*args, **kwargs)

        # set corresponding cpu backend
        original_backend = get_backend().name
        temp_backend = MATCHING_CPU_BACKEND[original_backend]
        set_backend(temp_backend)

        # run function
        result = func(*args, **kwargs)

        # set back original backend
        set_backend(original_backend)
        return result

    return wrapper


def _add_error_message(func, msg=""):
    """Decorator to add a custom error message to a function."""

    @wraps(func)
    def with_error_message(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"{msg}\nOriginal error:\n{type(e).__name__}: {e}")

    return with_error_message


_already_warned = [False]


def warn_if_not_float32(dtype):
    """Warn if X is not float32."""
    if _already_warned[0]:  # avoid warning multiple times
        return None

    if _dtype_to_str(dtype) != "float32":
        backend = get_backend()
        warnings.warn(
            f"GPU backend {backend.name} is much faster with single "
            f"precision floats (float32), got input in {dtype}. "
            "Consider casting your data to float32.", UserWarning)
        _already_warned[0] = True
