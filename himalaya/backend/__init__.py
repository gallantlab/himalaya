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
        backend = backend.__name__.split('.')[-1]

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
