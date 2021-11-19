"""
Multiple-kernel ridge with scikit-learn API
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
from himalaya.utils import generate_multikernel_dataset

from sklearn.pipeline import make_pipeline

# sphinx_gallery_thumbnail_number = 2
###############################################################################
# In this example, we use the ``torch_cuda`` backend.
#
# Torch can perform computations both on CPU and GPU. To use CPU, use the
# "torch" backend, to use GPU, use the "torch_cuda" backend.

backend = set_backend("torch_cuda", on_error="warn")

###############################################################################
# Generate a random dataset
# -------------------------
# - X_train : array of shape (n_samples_train, n_features)
# - X_test : array of shape (n_samples_test, n_features)
# - Y_train : array of shape (n_samples_train, n_targets)
# - Y_test : array of shape (n_samples_test, n_targets)

(X_train, X_test, Y_train, Y_test, kernel_weights,
 n_features_list) = generate_multikernel_dataset(n_kernels=3, n_targets=500,
                                                 n_samples_train=1000,
                                                 n_samples_test=300,
                                                 random_state=42)

feature_names = [f"Feature space {ii}" for ii in range(len(n_features_list))]

###############################################################################
# We could precompute the kernels by hand on ``Xs_train``, as done in
# ``plot_mkr_random_search.py``. Instead, here we use the ``ColumnKernelizer``
# to make a ``scikit-learn`` ``Pipeline``.

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
# Compare to a baseline ``KernelRidgeCV`` model with all the concatenated
# features. Comparison is performed using the prediction scores on the test
# set.

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
bins = np.linspace(0, max(scores_baseline.max(), scores.max()), 50)
plt.hist(scores_baseline, bins, alpha=0.7, label="KernelRidgeCV")
plt.hist(scores, bins, alpha=0.7, label="MultipleKernelRidgeCV")
plt.xlabel(r"$R^2$ generalization score")
plt.title("Histogram over targets")
plt.legend()
plt.show()
