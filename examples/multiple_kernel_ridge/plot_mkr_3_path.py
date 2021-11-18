"""
Multiple-kernel ridge path between two kernels
==============================================
This example demonstrates the path of all possible ratios of kernel weights
between two kernels, in a multiple kernel ridge regression model. Over the path
of ratios, the kernels are weighted by the kernel weights, then summed, and a
joint model is fit on the obtained kernel. The explained variance on a test set
is then computed, and decomposed over both kernels.
"""
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from himalaya.backend import set_backend
from himalaya.kernel_ridge import MultipleKernelRidgeCV
from himalaya.kernel_ridge import Kernelizer
from himalaya.kernel_ridge import ColumnKernelizer
from himalaya.progress_bar import bar
from himalaya.utils import generate_multikernel_dataset

from sklearn.pipeline import make_pipeline

###############################################################################
# In this example, we use the ``cupy`` backend.

backend = set_backend("cupy", on_error="warn")

###############################################################################
# We also use the nice display of scikit-learn pipelines.

from sklearn import set_config
set_config(display='diagram')  # requires scikit-learn 0.23

###############################################################################
# Generate a random dataset
# -------------------------
# - X_train : array of shape (n_samples_train, n_features)
# - X_test : array of shape (n_samples_test, n_features)
# - Y_train : array of shape (n_samples_train, n_targets)
# - Y_test : array of shape (n_samples_test, n_targets)

n_targets = 50
kernel_weights_true = np.tile(np.array([0.6, 0.4])[None], (n_targets, 1))

(X_train, X_test, Y_train, Y_test, kernel_weights_true,
 n_features_list) = generate_multikernel_dataset(
     n_kernels=2, n_targets=n_targets, n_samples_train=1000,
     n_samples_test=300, random_state=42, noise=0.3,
     kernel_weights_true=kernel_weights_true)

feature_names = [f"Feature space {ii}" for ii in range(len(n_features_list))]

###############################################################################
# Create a MultipleKernelRidgeCV model, see plot_mkr_sklearn_api.py for more
# details.

# Find the start and end of each feature space X in Xs.
start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
slices = [
    slice(start, end)
    for start, end in zip(start_and_end[:-1], start_and_end[1:])
]

# Create a different ``Kernelizer`` for each feature space.
kernelizers = [(name, Kernelizer(), slice_)
               for name, slice_ in zip(feature_names, slices)]
column_kernelizer = ColumnKernelizer(kernelizers)

# Create a MultipleKernelRidgeCV model.
solver_params = dict(alphas=np.logspace(-5, 5, 41), progress_bar=False)
model = MultipleKernelRidgeCV(kernels="precomputed", solver="random_search",
                              solver_params=solver_params)
pipe = make_pipeline(column_kernelizer, model)
pipe

###############################################################################
# Then, we manually perfom a hyperparameter grid search for the kernel weights.

# Make the score method use `split=True` by default.
model.score = partial(model.score, split=True)

# Define the hyperparameter grid search.
ratios = np.logspace(-4, 4, 41)
candidates = np.array([1 - ratios / (1 + ratios), ratios / (1 + ratios)]).T

# Loop over hyperparameter candidates
split_r2_scores = []
for candidate in bar(candidates, "Hyperparameter candidates"):
    # test one hyperparameter candidate at a time
    pipe[-1].solver_params["n_iter"] = candidate[None]
    pipe.fit(X_train, Y_train)

    # split the R2 score between both kernels
    scores = pipe.score(X_test, Y_test)
    split_r2_scores.append(backend.to_numpy(scores))

# average scores over targets for plotting
split_r2_scores_avg = np.array(split_r2_scores).mean(axis=2)

###############################################################################
# Plot the variance decomposition for all the hyperparameter ratios.
#
# For a ratio of 1e-3, feature space 0 is almost not used. For a ratio of 1e3,
# feature space 1 is almost not used. The best ratio is here around 1, because
# the feature spaces are used with similar scales in the stimulated dataset.

fig, ax = plt.subplots(figsize=(5, 3))
accumulator = np.zeros_like(ratios)
for split in split_r2_scores_avg.T:
    ax.fill_between(ratios, accumulator, accumulator + split, alpha=0.7)
    accumulator += split

ax.set(xscale='log')
ax.set(xlabel=r"Ratio of kernel weight ($\gamma_A / \gamma_B$)")
ax.set(ylabel=r"$R^2$ score (test set)")
ax.set(title=r"$R^2$ score decomposition")
ax.legend(feature_names, loc="upper left")
ax.grid()
plt.show()
