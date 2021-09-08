"""
Kernel ridge regression
=======================

This example demonstrates how to solve kernel ridge regression, using
himalaya's estimator ``KernelRidge`` compatible with scikit-learn's API.
"""

###############################################################################
# Create a random dataset
# -----------------------
import numpy as np
n_samples, n_features, n_targets = 10, 5, 4
X = np.random.randn(n_samples, n_features)
Y = np.random.randn(n_samples, n_targets)

###############################################################################
# Scikit-learn API
# ----------------
# Himalaya implements a ``KernelRidge`` estimator, similar to the corresponding
# scikit-learn estimator, with similar parameters and methods.
import sklearn.kernel_ridge
import himalaya.kernel_ridge

# Fit a scikit-learn model
model_skl = sklearn.kernel_ridge.KernelRidge(kernel="linear", alpha=0.1)
model_skl.fit(X, Y)

# Fit a himalaya model
model_him = himalaya.kernel_ridge.KernelRidge(kernel="linear", alpha=0.1)
model_him.fit(X, Y)

Y_pred_skl = model_skl.predict(X)
Y_pred_him = model_him.predict(X)

# The predictions are virtually identical.
print(np.max(np.abs(Y_pred_skl - Y_pred_him)))

###############################################################################
# Small API difference
# --------------------
# Since himalaya focuses on fitting multiple targets, the ``score`` method
# returns the score on each target separately, while scikit-learn returns the
# average score over targets.

print(model_skl.score(X, Y))
print(model_him.score(X, Y))
print(model_him.score(X, Y).mean())
