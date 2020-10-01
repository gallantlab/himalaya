"""
Kernel ridge with cross-validation
==================================

This example demonstrates how to solve kernel ridge regression with a
cross-validation of the regularization parameter, using himalaya's estimator
``KernelRidgeCV``.
"""

###############################################################################
# Create a random dataset
# -----------------------
import numpy as np
np.random.seed(0)
n_samples, n_features, n_targets = 10, 5, 4
X = np.random.randn(n_samples, n_features)
Y = np.random.randn(n_samples, n_targets)

###############################################################################
# Limit of GridSearchCV
# ---------------------
# In scikit-learn, one can use ``GridSearchCV`` to optimize hyperparameters
# over cross-validation.

import sklearn.model_selection
import sklearn.kernel_ridge

estimator = sklearn.kernel_ridge.KernelRidge(kernel="linear")
gscv = sklearn.model_selection.GridSearchCV(
    estimator=estimator,
    param_grid=dict(alpha=np.logspace(-2, 2, 5)),
)
gscv.fit(X, Y)

###############################################################################
# However, since ``GridSearchCV`` optimizes the average score over all targets,
# it returns a single value for alpha.
gscv.best_params_

###############################################################################
# KernelRidgeCV
# -------------
# To optimize each target independently, himalaya implements ``KernelRidgeCV``,
# which supports any cross-validation scheme compatible with scikit-learn.
import himalaya.kernel_ridge

model = himalaya.kernel_ridge.KernelRidgeCV(kernel="linear",
                                            alphas=np.logspace(-2, 2, 5))
model.fit(X, Y)

###############################################################################
# KernelRidgeCV returns a separate best alpha per target.
model.best_alphas_
