ALL_BACKENDS = [
    "numpy",
    "torch",
]

CURRENT_BACKEND = "torch"


def get_backend(backend):
    if backend == "numpy":
        from . import numpy as backend
    elif backend == "torch":
        from . import torch as backend

    return backend


def get_current_backend():
    return get_backend(CURRENT_BACKEND)
