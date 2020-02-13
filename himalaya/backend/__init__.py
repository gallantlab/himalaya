ALL_BACKENDS = [
    "numpy",
    "torch",
]

CURRENT_BACKEND = "torch"


def change_backend(backend):
    global CURRENT_BACKEND
    CURRENT_BACKEND = backend
    if backend == "numpy":
        from . import numpy as backend
    elif backend == "torch":
        from . import torch as backend

    return backend


def get_current_backend():
    return change_backend(CURRENT_BACKEND)
