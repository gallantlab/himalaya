"""
Fitting a model on GPU
======================

This example demonstrates how to fit a model using GPU computations.

Himalaya implements different computational backends to fit the models:

- "numpy" (CPU) (default)
- "torch" (CPU)
- "torch_cuda" (GPU)
- "cupy" (GPU)

Each backend is only available if you installed the corresponding package with
CUDA enabled. Check the ``pytorch``/``cupy`` documentation for installation
instructions.
"""

###############################################################################
# Create a random dataset
# -----------------------
import numpy as np
n_samples, n_features, n_targets = 10, 20, 4
X = np.random.randn(n_samples, n_features)
Y = np.random.randn(n_samples, n_targets)

###############################################################################
# Change backend
# --------------
# To change the backend, you need to call the function
# ``himalaya.backend.set_backend``. With the option ``on_error="warn"``, the
# function does not raise an error if the new backend fails to be imported, and
# the backend is kept unchanged.

from himalaya.backend import set_backend
backend = set_backend("cupy", on_error="warn")

###############################################################################
# GPU backend
# -----------
# To fit a himalaya model on GPU, you don't need to move the input arrays to
# GPU, the method ``fit`` will do it for you. However, the float precision will
# not be changed.
#
# To make the most of GPU memory and computational speed, you might want to
# change the float precision to float32.
X = X.astype("float32")

from himalaya.kernel_ridge import KernelRidge
model_him = KernelRidge(kernel="linear", alpha=0.1)
model_him.fit(X, Y)

###############################################################################
# The results are stored in GPU memory, using an array object specific to the
# backend used. To use the results in other libraries (for example matplotlib),
# you can create a numpy array using the function ``backend.to_numpy``.
scores = model_him.score(X, Y)
print(scores.__class__)
scores = backend.to_numpy(scores)
print(scores.__class__)
