"""
Fit a multiple-kernel ridge models from fixed hyper-parameters
==============================================================
This example demonstrates how to fit a multiple-kernel ridge model with fixed
hyper-parameters. Here are three different usecases:
- If the kernel weights hyper-parameters are known and identical across
  targets, the kernels can be scaled and summed, and a simple KernelRidgeCV can
  be used to fit the model.
- If the kernel weights hyper-parameters are unknown and different across
  targets, a MultipleKernelRidgeCV can be use to search the best
  hyper-parameters per target.
- If the kernel weights hyper-parameters are known and different across
  targets, a WeightedKernelRidge model can be used to fit the ridge models on
  each target independently.

This method can be used for example in the following workflow:
- fit a MultipleKernelRidgeCV to learn the kernel weights hyper-parameter,
- save the hyper-parameters, but not the ridge weights to save disk space,
- fit a WeightedKernelRidge from the saved hyper-parameters, for further use of
  the model (prediction, interpretation, etc.).
"""
import numpy as np

from himalaya.backend import set_backend
from himalaya.kernel_ridge import WeightedKernelRidge
from himalaya.kernel_ridge import Kernelizer
from himalaya.kernel_ridge import ColumnKernelizer
from himalaya.kernel_ridge import generate_dirichlet_samples

from sklearn.pipeline import make_pipeline

###############################################################################
# In this example, we use the ``torch_cuda`` backend (GPU).

backend = set_backend("torch_cuda")

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
# Define the weighted kernel ridge model
# --------------------------------------
# Here we use the ground truth kernel weights for each target (deltas), but it
# can be typically used with deltas obtained from a MultipleKernelRidgeCV fit.

deltas = backend.log(backend.asarray(kernel_weights_true.T))

model_1 = WeightedKernelRidge(alpha=1, deltas=deltas, kernels="precomputed")
pipe_1 = make_pipeline(column_kernelizer, model_1)

# Fit the model on all targets
pipe_1.fit(X_train, Y_train)

# compute test score
test_scores_1 = pipe_1.score(X_test, Y_test)
test_scores_1 = backend.to_numpy(test_scores_1)

###############################################################################
# We can compare this model to a baseline model where the kernel weights are
# all equal and not learnt.

model_2 = WeightedKernelRidge(alpha=1, deltas="zeros", kernels="precomputed")
pipe_2 = make_pipeline(column_kernelizer, model_2)

# Fit the model on all targets
pipe_2.fit(X_train, Y_train)

# compute test score
test_scores_2 = pipe_2.score(X_test, Y_test)
test_scores_2 = backend.to_numpy(test_scores_2)

###############################################################################
# Compare the predictions on a test set
# -------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 3))
plt.hist(test_scores_2, np.linspace(0, 1, 30), alpha=0.7,
         label="Default deltas")
plt.hist(test_scores_1, np.linspace(0, 1, 30), alpha=0.7,
         label="Ground truth deltas")
plt.xlabel("$R^2$ generalization score")
plt.ylabel("Number of voxels")
plt.legend()
plt.tight_layout()
plt.show()
