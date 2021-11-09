"""
Refine multiple-kernel ridge results
====================================
This example demonstrates how to solve multiple-kernel ridge regression with
hyperparameter random search, then refine the results with hyperparameter
gradient descent.
"""
import numpy as np

from himalaya.backend import set_backend
from himalaya.kernel_ridge import MultipleKernelRidgeCV
from himalaya.kernel_ridge import Kernelizer
from himalaya.kernel_ridge import ColumnKernelizer
from himalaya.kernel_ridge import generate_dirichlet_samples

from sklearn.pipeline import make_pipeline

###############################################################################
# In this example, we use the ``cupy`` backend (GPU).

backend = set_backend("cupy")

###############################################################################
# We can display the ``scikit-learn`` pipeline with an HTML diagram.
from sklearn import set_config
set_config(display='diagram')  # requires scikit-learn 0.23

###############################################################################
# Generate a random dataset
# -------------------------
# - Xs_train : list of arrays of shape (n_samples_train, n_features)
# - Xs_test : list of arrays of shape (n_samples_test, n_features)
# - Y_train : array of shape (n_samples_train, n_targets)
# - Y_test : array of shape (n_repeat, n_samples_test, n_targets)

n_kernels = 4
n_targets = 500

# We create a few kernel weights
rng = np.random.RandomState(42)
kernel_weights_true = generate_dirichlet_samples(n_targets, n_kernels,
                                                 concentration=[.3],
                                                 random_state=rng)
kernel_weights_true = backend.to_numpy(kernel_weights_true)

# Then, we generate a random dataset, using the arbitrary scalings.
n_samples_train = 1000
n_samples_test = 400
n_features_list = np.full(n_kernels, fill_value=1000)

Xs_train, Xs_test = [], []
Y_train, Y_test = None, None
for ii in range(n_kernels):
    n_features = n_features_list[ii]

    X_train = rng.randn(n_samples_train, n_features)
    X_test = rng.randn(n_samples_test, n_features)
    X_train -= X_train.mean(0)
    Xs_train.append(X_train)
    Xs_test.append(X_test)

    weights = rng.randn(n_features, n_targets) / n_features
    weights *= kernel_weights_true[:, ii] ** 0.5

    if ii == 0:
        Y_train = X_train @ weights
        Y_test = X_test @ weights
    else:
        Y_train += X_train @ weights
        Y_test += X_test @ weights

std = Y_train.std(0)[None]
Y_train /= std
Y_test /= std

noise = 0.1
Y_train += rng.randn(n_samples_train, n_targets) * noise
Y_test += rng.randn(n_samples_test, n_targets) * noise
Y_test -= Y_test.mean(0)
Y_train -= Y_train.mean(0)

# Concatenate the feature spaces.
X_train = np.asarray(np.concatenate(Xs_train, 1), dtype="float32")
X_test = np.asarray(np.concatenate(Xs_test, 1), dtype="float32")

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

solver_params = dict(
    n_iter=5,
    alphas=np.logspace(-10, 10, 41),
)

model_1 = MultipleKernelRidgeCV(kernels="precomputed", solver="random_search",
                                solver_params=solver_params, random_state=rng)
pipe_1 = make_pipeline(column_kernelizer, model_1)

# Fit the model on all targets
pipe_1.fit(X_train, Y_train)

###############################################################################
# Define the gradient-descent model
# ---------------------------------

solver_params = dict(
    max_iter=100,
    hyper_gradient_method="direct",
    max_iter_inner_hyper=10,
    initial_deltas="here_will_go_the_previous_deltas"
)

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
plt.plot(plt.xlim(), plt.xlim(), color='k', lw=1)
plt.xlabel(r"Base model")
plt.ylabel(r"Refined model")
plt.title("$R^2$ generalization score")
plt.tight_layout()
plt.show()
