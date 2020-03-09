ALL_BACKENDS = [
    "numpy",
    "torch",
    "torch_cuda",
]

CURRENT_BACKEND = "numpy"


def set_backend(backend):
    global CURRENT_BACKEND
    CURRENT_BACKEND = backend
    return get_current_backend()


def get_current_backend():
    if CURRENT_BACKEND == "numpy":
        from . import numpy as backend
    elif CURRENT_BACKEND == "torch":
        from . import torch as backend
    elif CURRENT_BACKEND == "torch_cuda":
        from . import torch_cuda as backend
    return backend
