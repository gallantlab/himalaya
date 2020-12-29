"""
Multiple kernel ridge with scikit-learn API
===========================================
This example demonstrates how to solve multiple kernel ridge regression, using
scikit-learn API.
"""

import numpy as np
import matplotlib.pyplot as plt

from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.kernel_ridge import MultipleKernelRidgeCV
from himalaya.kernel_ridge import Kernelizer
from himalaya.kernel_ridge import ColumnKernelizer

from sklearn.pipeline import make_pipeline

# sphinx_gallery_thumbnail_number = 2
###############################################################################
# In this example, we use the ``torch`` backend.
#
# Torch can perform computations both on CPU and GPU.
# To use the CPU, use the "torch" backend.
# To use GPU, you can either use the "torch" backend and move your data to GPU
# with the ``.cuda`` method, or you can use the "torch_cuda" backend which calls
# this method in ``backend.asarray``.

backend = set_backend("torch_cuda")

###############################################################################
# Generate a random dataset
# -------------------------
# - Xs_train : list of arrays of shape (n_samples_train, n_features)
# - Xs_test : list of arrays of shape (n_samples_test, n_features)
# - Y_train : array of shape (n_samples_train, n_targets)
# - Y_test : array of shape (n_repeat, n_samples_test, n_targets)

n_samples_train = 1000
n_samples_test = 300
n_targets = 1000
n_features_list = [1000, 1000, 500]
feature_names = ["feature space A", "feature space B", "feature space C"]

Xs_train = [
    backend.randn(n_samples_train, n_features)
    for n_features in n_features_list
]
Xs_test = [
    backend.randn(n_samples_test, n_features) for n_features in n_features_list
]
ws = [
    backend.randn(n_features, n_targets) / n_features
    for n_features in n_features_list
]
Y_train = backend.stack([X @ w for X, w in zip(Xs_train, ws)]).sum(0)
Y_test = backend.stack([X @ w for X, w in zip(Xs_test, ws)]).sum(0)

###############################################################################
# Optional: Add some arbitrary scalings per kernel
if True:
    scalings = [0.2, 5, 1]
    Xs_train = [X * scaling for X, scaling in zip(Xs_train, scalings)]
    Xs_test = [X * scaling for X, scaling in zip(Xs_test, scalings)]

###############################################################################
# Concatenate the feature spaces and move to GPU with ``backend.asarray``.
X_train = backend.asarray(backend.concatenate(Xs_train, 1), dtype="float32")
X_test = backend.asarray(backend.concatenate(Xs_test, 1), dtype="float32")

###############################################################################
# We could precompute the kernels by hand on ``Xs_train``, as done in
# ``plot_mkr_random_search.py``. Instead, here we use the
# ``ColumnKernelizer`` to make a ``scikit-learn`` ``Pipeline``.

# Find the start and end of each feature space X in Xs
start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
slices = [
    slice(start, end)
    for start, end in zip(start_and_end[:-1], start_and_end[1:])
]

###############################################################################
# Create a different ``Kernelizer`` for each feature space. Here we use a
# linear kernel for all feature spaces, but ``ColumnKernelizer`` accepts any
# ``Kernelizer``, or ``scikit-learn`` ``Pipeline`` ending with a
# ``Kernelizer``.
kernelizers = [(name, Kernelizer(), slice_)
               for name, slice_ in zip(feature_names, slices)]
column_kernelizer = ColumnKernelizer(kernelizers)

# Note that ``ColumnKernelizer`` has a parameter ``n_jobs`` to parallelize each
# kernelizer, yet such parallelism does not work with GPU arrays.

###############################################################################
# Define the model
# ----------------
#
# The class takes a number of common parameters during initialization, such as
# `kernels` or `solver`. Since the solver parameters might be different
# depending on the solver, they can be passed in the `solver_params` parameter.

###############################################################################
# Here we use the "random_search" solver.
# We can check its specific parameters in the function docstring:
solver_function = MultipleKernelRidgeCV.ALL_SOLVERS["random_search"]
print("Docstring of the function %s:" % solver_function.__name__)
print(solver_function.__doc__)

###############################################################################
# We use 100 iterations to have a reasonably fast example (~40 sec).
# To have a better convergence, we probably need more iterations.
# Note that there is currently no stopping criterion in this method.
n_iter = 100

###############################################################################
# Grid of regularization parameters.
alphas = np.logspace(-10, 10, 41)

###############################################################################
# Batch parameters are used to reduce the necessary GPU memory. A larger value
# will be a bit faster, but the solver might crash if it runs out of memory.
# Optimal values depend on the size of your dataset.
n_targets_batch = 1000
n_alphas_batch = 20
n_targets_batch_refit = 200

solver_params = dict(n_iter=n_iter, alphas=alphas,
                     n_targets_batch=n_targets_batch,
                     n_alphas_batch=n_alphas_batch,
                     n_targets_batch_refit=n_targets_batch_refit,
                     jitter_alphas=True)

model = MultipleKernelRidgeCV(kernels="precomputed", solver="random_search",
                              solver_params=solver_params)

###############################################################################
# Define and fit the pipeline
pipe = make_pipeline(column_kernelizer, model)
pipe.fit(X_train, Y_train)

###############################################################################
# Plot the convergence curve
# --------------------------

# ``cv_scores`` gives the scores for each sampled kernel weights.
# The convergence curve is thus the current maximum for each target.
cv_scores = backend.to_numpy(pipe[1].cv_scores_)
current_max = np.maximum.accumulate(cv_scores, axis=0)
mean_current_max = np.mean(current_max, axis=1)

x_array = np.arange(1, len(mean_current_max) + 1)
plt.plot(x_array, mean_current_max, '-o')
plt.grid("on")
plt.xlabel("Number of kernel weights sampled")
plt.ylabel("L2 negative loss (higher is better)")
plt.title("Convergence curve, averaged over targets")
plt.tight_layout()
plt.show()

###############################################################################
# Compare to ``KernelRidgeCV``
# ----------------------------
# Compare to a baseline ``KernelRidgeCV`` model with all the concatenated features.
# Comparison is performed using the prediction scores on the test set.


###############################################################################
# Fit the baseline model ``KernelRidgeCV``
baseline = KernelRidgeCV(kernel="linear", alphas=alphas)
baseline.fit(X_train, Y_train)

###############################################################################
# Compute scores of both models
scores = pipe.score(X_test, Y_test)
scores = backend.to_numpy(scores)

scores_baseline = baseline.score(X_test, Y_test)
scores_baseline = backend.to_numpy(scores_baseline)

###############################################################################
# Plot histograms
bins = np.linspace(min(scores_baseline.min(), scores.min()),
                   max(scores_baseline.max(), scores.max()), 50)
plt.hist(scores, bins, alpha=0.5, label="MultipleKernelRidgeCV")
plt.hist(scores_baseline, bins, alpha=0.5, label="KernelRidgeCV")
plt.xlabel(r"$R^2$ generalization score")
plt.title("Histogram over targets")
plt.legend()
plt.show()
