"""
Multiple kernel ridge regression
================================
This example demonstrates how to solve multiple kernel ridge regression.
It uses random search and cross validation to select optimal hyperparameters.
"""

import numpy as np
import matplotlib.pyplot as plt

from himalaya.backend import change_backend
from himalaya.ridge import solve_multiple_kernel_ridge_random_search
from himalaya.ridge import generate_dirichlet_samples
from himalaya.ridge import predict_and_score
from himalaya.scoring import l2_neg_loss
from himalaya.scoring import r2_score_split
from himalaya.viz import plot_alphas_diagnostic

print(__doc__)

###############################################################################
# In this example, we use the torch backend, and fit the model on GPU.

backend = change_backend("torch")

###############################################################################
# Generate a random dataset.
#
# Xs_train : list of array of shape (n_samples_train, n_features)
# Xs_test : list of array of shape (n_samples_test, n_features)
# Y_train : array of shape (n_samples_train, n_targets)
# Y_test : array of shape (n_repeat, n_samples_test, n_targets)

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

# Add some arbitrary scalings per kernel
scalings = [0.2, 5, 1]
Xs_train = [X * scaling for X, scaling in zip(Xs_train, scalings)]
Xs_test = [X * scaling for X, scaling in zip(Xs_test, scalings)]

###############################################################################
# Precompute the linear kernels, and cast them to float32.
#
# We also send to GPU memory with `.cuda()`, to do the computation on GPU.

Ks_train = backend.stack([X_train @ X_train.T for X_train in Xs_train])
Ks_train = backend.asarray(Ks_train, dtype=backend.float32).cuda()
Y_train = backend.asarray(Y_train, dtype=backend.float32).cuda()

Ks_test = backend.stack(
    [X_test @ X_train.T for X_train, X_test in zip(Xs_train, Xs_test)])
Ks_test = backend.asarray(Ks_test, dtype=backend.float32).cuda()
Y_test = backend.asarray(Y_test, dtype=backend.float32).cuda()

###############################################################################
# Run the solver, using random search. This method should work fine for
# small number of kernels (< 20). The larger the number of kenels, the larger
# we need to sample the hyperparameter space (i.e. increasing n_iter).

# Here we use 100 iterations to have a reasonably fast example (~40 sec).
# To have a better convergence, we probably need more iterations.
# Note that there is currently no stopping criterion in this method.
n_iter = 100

# Grid of regularization parameters.
alphas = np.logspace(-10, 10, 21)

# Generate the samples in hyperparameter space.
n_kernels = len(Ks_train)
# As n_kernels increases, the Dirichlet's concentration parameters need to be
# smaller to get to the edges of Dirichlet space, with an arbitrary rule of
# thumb of 1 / n_kernels.
concentrations = np.logspace(np.log10(0.1 / n_kernels), 0, 3)
gammas = generate_dirichlet_samples(n_iter, n_kernels,
                                    concentrations=concentrations,
                                    random_state=0)

# Batch parameters, used to reduce the necessary GPU memory. A larger value
# will be a bit faster, but the solver might crash if it is out of memory.
# Optimal values depend on the size of your dataset.
n_targets_batch = 1000
n_alphas_batch = 20

# If compute_weights == "dual", the solver will use more memory.
# Too mitigate it, you can reduce `n_targets_batch` in the refit
# using `n_targets_batch_refit`.
# If you don't need the dual weights, use compute_weights = None.
compute_weights = 'dual'
n_targets_batch_refit = 200

# Run the solver. For each hyperparameter gamma, it will:
# - fit (n_splits * n_alphas * n_targets) ridge models
# - compute the scores on the validation set of each split
# - average the scores over splits
# - take the maximum over alphas
# - (only if you ask for the ridge weights) refit using the best alphas per
#   target and the entire dataset
# - return for each target the alpha and gamma leading to the best CV score
#   (and the best weights if necessary)
results = solve_multiple_kernel_ridge_random_search(
    Ks=Ks_train,
    Y=Y_train,
    gammas=gammas,
    alphas=alphas,
    score_func=l2_neg_loss,
    cv_splitter=10,
    n_targets_batch=n_targets_batch,
    compute_weights=compute_weights,
    n_alphas_batch=n_alphas_batch,
    n_targets_batch_refit=n_targets_batch_refit,
    jitter_alphas=True,
)

# As we used the torch backend, the results are torch.Tensors.
# As the data was on GPU, the results are also on GPU.
# Here, we cast the results back to CPU, and to numpy arrays.
all_scores_mean = backend.to_numpy(results[0])
best_gammas = backend.to_numpy(results[1])
best_alphas = backend.to_numpy(results[2])
dual_weights = backend.to_numpy(results[3])

###############################################################################
# Plot the convergence curve.
#
# `all_scores_mean` gives the scores for each gamma.
# The convergence curve is thus the current maximum for each target.

current_max = np.maximum.accumulate(all_scores_mean, axis=0)
mean_current_max = np.mean(current_max, axis=1)
x_array = np.arange(1, len(mean_current_max) + 1)
plt.plot(x_array, mean_current_max, '-o')
plt.grid("on")
plt.xlabel("number of gamma samples")
plt.title("Convergence of the L2 negative loss, averaged over targets")
plt.show()

###############################################################################
# Plot the optimal alphas selected by the solver.
#
# This plot is helpful to refine the alpha grid if the range is too small or
# too large.

plot_alphas_diagnostic(best_alphas, alphas)
plt.title("Best alphas selected by cross-validation")
plt.show()

###############################################################################
# Compute the predictions on the test set (requires the dual weights).
#
# The generalization scores are close to zero since the dataset is only noise.

split = False
scores = predict_and_score(Ks_test, dual_weights, best_gammas, Y_test,
                           split=split, n_targets_batch=n_targets_batch,
                           score_func=r2_score_split)
scores = backend.to_numpy(scores)

plt.hist(scores, 50)
plt.title(r"Histogram of $R^2$ generalization score")
plt.show()

###############################################################################
# Compute the split predictions on the test set (requires the dual weights).
#
# Here we apply the dual weights on each kernel separately
# (gamma[i] * kernel[i]), and we compute the R2 scores
# (corrected for correlations) of each prediction.

split = True
scores = predict_and_score(Ks_test, dual_weights, best_gammas, Y_test,
                           split=split, n_targets_batch=n_targets_batch,
                           score_func=r2_score_split)
scores = backend.to_numpy(scores)

bins = np.linspace(scores.min(), scores.max(), 50)
for score in scores:
    plt.hist(score, bins, alpha=0.5)
plt.title(r"Histogram of $R^2$ generalization score split between kernels")
plt.legend(["kernel %d" % kk for kk in range(scores.shape[0])])
plt.show()
