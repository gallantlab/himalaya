"""
Multiple-kernel ridge refining
==============================
This example demonstrates how to solve multiple-kernel ridge regression with
hyperparameter random search, then refine the results with hyperparameter
gradient descent.
"""
import numpy as np

from himalaya.backend import set_backend
from himalaya.kernel_ridge import MultipleKernelRidgeCV
from himalaya.kernel_ridge import Kernelizer
from himalaya.kernel_ridge import ColumnKernelizer
from himalaya.utils import generate_multikernel_dataset

from sklearn.pipeline import make_pipeline

###############################################################################
# In this example, we use the ``cupy`` backend (GPU).

backend = set_backend("cupy", on_error="warn")

###############################################################################
# We can display the ``scikit-learn`` pipeline with an HTML diagram.
from sklearn import set_config
set_config(display='diagram')  # requires scikit-learn 0.23

###############################################################################
# Generate a random dataset
# -------------------------
# - X_train : array of shape (n_samples_train, n_features)
# - X_test : array of shape (n_samples_test, n_features)
# - Y_train : array of shape (n_samples_train, n_targets)
# - Y_test : array of shape (n_samples_test, n_targets)

(X_train, X_test, Y_train, Y_test, kernel_weights,
 n_features_list) = generate_multikernel_dataset(n_kernels=4, n_targets=500,
                                                 n_samples_train=1000,
                                                 n_samples_test=300,
                                                 random_state=42)

feature_names = [f"Feature space {ii}" for ii in range(len(n_features_list))]

###############################################################################
# Prepare the pipeline
# --------------------

# Find the start and end of each feature space X in Xs
start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
slices = [
    slice(start, end)
    for start, end in zip(start_and_end[:-1], start_and_end[1:])
]

# Create a different ``Kernelizer`` for each feature space.
kernelizers = [("space %d" % ii, Kernelizer(), slice_)
               for ii, slice_ in enumerate(slices)]
column_kernelizer = ColumnKernelizer(kernelizers)

###############################################################################
# Define the random-search model
# ------------------------------
# We use very few iteration on purpose, to make the random search suboptimal,
# and refine it with hyperparameter gradient descent.

solver_params = dict(n_iter=5, alphas=np.logspace(-10, 10, 41))

model_1 = MultipleKernelRidgeCV(kernels="precomputed", solver="random_search",
                                solver_params=solver_params, random_state=42)
pipe_1 = make_pipeline(column_kernelizer, model_1)

# Fit the model on all targets
pipe_1.fit(X_train, Y_train)

###############################################################################
# Define the gradient-descent model
# ---------------------------------

solver_params = dict(max_iter=100, hyper_gradient_method="direct",
                     max_iter_inner_hyper=10,
                     initial_deltas="here_will_go_the_previous_deltas")

model_2 = MultipleKernelRidgeCV(kernels="precomputed", solver="hyper_gradient",
                                solver_params=solver_params)
pipe_2 = make_pipeline(column_kernelizer, model_2)

###############################################################################
# Use the random-search to initialize the gradient-descent
# --------------------------------------------------------

# We might want to refine only the best predicting targets, since the
# hyperparameter gradient descent is less efficient over many targets.
top = 60  # top 60%
best_cv_scores = backend.to_numpy(pipe_1[-1].cv_scores_.max(0))
mask = best_cv_scores > np.percentile(best_cv_scores, 100 - top)

pipe_2[-1].solver_params['initial_deltas'] = pipe_1[-1].deltas_[:, mask]
pipe_2.fit(X_train, Y_train[:, mask])

###############################################################################
# Compute predictions on a test set
# ---------------------------------
import matplotlib.pyplot as plt

# use the first model for all targets
test_scores_1 = pipe_1.score(X_test, Y_test)

# use the second model for the refined targets
test_scores_2 = backend.copy(test_scores_1)
test_scores_2[mask] = pipe_2.score(X_test, Y_test[:, mask])

test_scores_1 = backend.to_numpy(test_scores_1)
test_scores_2 = backend.to_numpy(test_scores_2)
plt.figure(figsize=(4, 4))
plt.scatter(test_scores_1, test_scores_2, alpha=0.3)
plt.xlim(0, 1)
plt.plot(plt.xlim(), plt.xlim(), color='k', lw=1)
plt.xlabel(r"Base model")
plt.ylabel(r"Refined model")
plt.title("$R^2$ generalization score")
plt.tight_layout()
plt.show()
