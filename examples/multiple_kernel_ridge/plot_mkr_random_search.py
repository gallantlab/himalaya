"""
Multiple kernel ridge regression
================================
This example demonstrates how to solve multiple kernel ridge regression.
It uses random search and cross validation to select optimal hyperparameters.
"""

import numpy as np
import matplotlib.pyplot as plt

from himalaya.backend import set_backend
from himalaya.kernel_ridge import solve_multiple_kernel_ridge_random_search
from himalaya.kernel_ridge import predict_and_score_weighted_kernel_ridge
from himalaya.scoring import r2_score_split
from himalaya.viz import plot_alphas_diagnostic

# sphinx_gallery_thumbnail_number = 4
###############################################################################
# In this example, we use the ``cupy`` backend, and fit the model on GPU.

backend = set_backend("cupy")

###############################################################################
# Generate a random dataset
# -------------------------
#
# - Xs_train : list of arrays of shape (n_samples_train, n_features)
# - Xs_test : list of arrays of shape (n_samples_test, n_features)
# - Y_train : array of shape (n_samples_train, n_targets)
# - Y_test : array of shape (n_repeat, n_samples_test, n_targets)

n_samples_train = 1000
n_samples_test = 300
n_targets = 1000
n_features_list = [1000, 1000, 500]

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

Y_train -= Y_train.mean(0)
Y_test -= Y_test.mean(0)

###############################################################################
# Precompute the linear kernels
# -----------------------------
# We also cast them to float32.

Ks_train = backend.stack([X_train @ X_train.T for X_train in Xs_train])
Ks_train = backend.asarray(Ks_train, dtype=backend.float32)
Y_train = backend.asarray(Y_train, dtype=backend.float32)

Ks_test = backend.stack(
    [X_test @ X_train.T for X_train, X_test in zip(Xs_train, Xs_test)])
Ks_test = backend.asarray(Ks_test, dtype=backend.float32)
Y_test = backend.asarray(Y_test, dtype=backend.float32)

###############################################################################
# Run the solver, using random search
# -----------------------------------
# This method should work fine for
# small number of kernels (< 20). The larger the number of kenels, the larger
# we need to sample the hyperparameter space (i.e. increasing ``n_iter``).

###############################################################################
# Here we use 100 iterations to have a reasonably fast example (~40 sec).
# To have a better convergence, we probably need more iterations.
# Note that there is currently no stopping criterion in this method.
n_iter = 100

###############################################################################
# Grid of regularization parameters.
alphas = np.logspace(-10, 10, 21)

###############################################################################
# Batch parameters are used to reduce the necessary GPU memory. A larger value
# will be a bit faster, but the solver might crash if it runs out of memory.
# Optimal values depend on the size of your dataset.
n_targets_batch = 1000
n_alphas_batch = 20

###############################################################################
# If ``return_weights == "dual"``, the solver will use more memory.
# To mitigate this, you can reduce ``n_targets_batch`` in the refit
# using ```n_targets_batch_refit``.
# If you don't need the dual weights, use ``return_weights = None``.
return_weights = 'dual'
n_targets_batch_refit = 200

###############################################################################
# Run the solver. For each iteration, it will:
#
# - sample kernel weights from a Dirichlet distribution
# - fit (n_splits * n_alphas * n_targets) ridge models
# - compute the scores on the validation set of each split
# - average the scores over splits
# - take the maximum over alphas
# - (only if you ask for the ridge weights) refit using the best alphas per
#   target and the entire dataset
# - return for each target the log kernel weights leading to the best CV score
#   (and the best weights if necessary)
results = solve_multiple_kernel_ridge_random_search(
    Ks=Ks_train,
    Y=Y_train,
    n_iter=n_iter,
    alphas=alphas,
    n_targets_batch=n_targets_batch,
    return_weights=return_weights,
    n_alphas_batch=n_alphas_batch,
    n_targets_batch_refit=n_targets_batch_refit,
    jitter_alphas=True,
)

###############################################################################
# As we used the ``cupy`` backend, the results are ``cupy`` arrays, which are
# on GPU. Here, we cast the results back to CPU, and to ``numpy`` arrays.
deltas = backend.to_numpy(results[0])
dual_weights = backend.to_numpy(results[1])
cv_scores = backend.to_numpy(results[2])

###############################################################################
# Plot the convergence curve
# --------------------------
#
# ``cv_scores`` gives the scores for each sampled kernel weights.
# The convergence curve is thus the current maximum for each target.

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
# Plot the optimal alphas selected by the solver
# ----------------------------------------------
#
# This plot is helpful to refine the alpha grid if the range is too small or
# too large.

best_alphas = 1. / np.sum(np.exp(deltas), axis=0)
plot_alphas_diagnostic(best_alphas, alphas)
plt.title("Best alphas selected by cross-validation")
plt.show()

###############################################################################
# Compute the predictions on the test set
# ---------------------------------------
# (requires the dual weights)

split = False
scores = predict_and_score_weighted_kernel_ridge(
    Ks_test, dual_weights, deltas, Y_test, split=split,
    n_targets_batch=n_targets_batch, score_func=r2_score_split)
scores = backend.to_numpy(scores)

plt.hist(scores, 50)
plt.xlabel(r"$R^2$ generalization score")
plt.title("Histogram over targets")
plt.show()

###############################################################################
# Compute the split predictions on the test set 
# ---------------------------------------------
# (requires the dual weights)
#
# Here we apply the dual weights on each kernel separately
# (``exp(deltas[i]) * kernel[i]``), and we compute the R\ :sup:`2` scores
# (corrected for correlations) of each prediction.

split = True
scores = predict_and_score_weighted_kernel_ridge(
    Ks_test, dual_weights, deltas, Y_test, split=split,
    n_targets_batch=n_targets_batch, score_func=r2_score_split)
scores = backend.to_numpy(scores)

bins = np.linspace(scores.min(), scores.max(), 50)
for score in scores:
    plt.hist(score, bins, alpha=0.5)
plt.title(r"Histogram of $R^2$ generalization score split between kernels")
plt.legend(["kernel %d" % kk for kk in range(scores.shape[0])])
plt.show()
