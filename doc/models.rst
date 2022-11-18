Model descriptions
==================

This package implements a number of models.

Ridge
-----

Let :math:`X\in \mathbb{R}^{n\times p}` be a feature matrix with :math:`n`
samples and :math:`p` features,  :math:`y\in \mathbb{R}^n` a target vector, and
:math:`\alpha > 0` a fixed regularization hyperparameter. Ridge regression
[1]_ defines the weight vector :math:`b^*\in \mathbb{R}^p` as:

.. math::
    b^* = \arg\min_b \|Xb - y\|_2^2 + \alpha \|b\|_2^2.

The equation has a  closed-form solution :math:`b^* = M y`, where :math:`M =
(X^\top X + \alpha I_p)^{-1}X^\top \in  \mathbb{R}^{p \times n}`.

.. admonition:: This model is implemented in

  - :class:`~himalaya.ridge.Ridge` (scikit-learn-compatible estimator)
  - :func:`~himalaya.ridge.solve_ridge_svd` (function)

KernelRidge
-----------

By the Woodbury matrix identity, :math:`b^*` can be written as :math:`b^* =
X^\top(XX^\top + \alpha I_n)^{-1}y`, or :math:`b^* = X^\top w^*` for some
:math:`w^*\in \mathbb{R}^n`. Noting the linear kernel :math:`K = X X^\top \in
\mathbb{R}^{n\times n}`, this leads to the *equivalent* formulation:

.. math::
    w^* = \arg\min_w \|Kw - y\|_2^2 + \alpha w^\top Kw.

This model can be extended to arbitrary positive semidefinite kernels
:math:`K`, leading to the more general kernel ridge regression [2]_.

.. admonition:: This model is implemented in

  - :class:`~himalaya.kernel_ridge.KernelRidge` (scikit-learn-compatible estimator)
  - :func:`~himalaya.kernel_ridge.solve_kernel_ridge_eigenvalues` (function)
  - :func:`~himalaya.kernel_ridge.solve_kernel_ridge_gradient_descent` (function)
  - :func:`~himalaya.kernel_ridge.solve_kernel_ridge_conjugate_gradient` (function)


RidgeCV and KernelRidgeCV
-------------------------

In practice, because the ridge regression and kernel ridge regression
hyperparameter :math:`\alpha` is unknown, it is typically selected through a
grid-search with cross-validation. In cross-validation, we split the data set
into a training set :math:`(X_{train}, y_{train})` and a validation set
:math:`(X_{val}, y_{val})`. Then, we train the model on the training set, and
evaluate the generalization performance on the validation set. We perform this
process for multiple hyperparameter candidates :math:`\alpha`, typically
defined over a grid of log-spaced values. Finally, we keep the candidate
leading to the best generalization performance, as measured by the validation
loss, averaged over all cross-validation splits.

.. admonition:: These models are implemented in

  - :class:`~himalaya.ridge.RidgeCV` (scikit-learn-compatible estimator)
  - :func:`~himalaya.ridge.solve_ridge_cv_svd` (function)
  - :class:`~himalaya.kernel_ridge.KernelRidgeCV` (scikit-learn-compatible estimator)
  - :func:`~himalaya.kernel_ridge.solve_kernel_ridge_cv_eigenvalues` (function)


GroupRidgeCV / BandedRidgeCV
----------------------------

In some applications, features are naturally grouped into groups (or feature
spaces). To adapt the regularization level to each feature space, ridge
regression can be extended to group-regularized ridge regression (also known
as banded ridge regression [3]_). In this model, a separate hyperparameter is
optimized for each feature space:

.. math::
    b^* = \arg\min_b \|\sum_{i=1}^m X_i b_i - y\|_2^2 + \sum_{i=1}^m \alpha_i \|b_i\|_2^2.

This is equivalent to solving a ridge regression:

.. math::
    b^* = \arg\min_b \|Z b - Y\|_2^2 + \|b\|_2^2

where the feature space :math:`X_i` is scaled by a group scaling :math:`Z_i =
e^{\delta_i} X_i`. The hyperparameters :math:`\delta_i = - \log(\alpha_i)` are
then learned over cross-validation [4]_. 

.. admonition:: This model is implemented in

  - :class:`~himalaya.ridge.GroupRidgeCV` (scikit-learn-compatible estimator)
  - :func:`~himalaya.ridge.solve_group_ridge_random_search` (function)

  See also multiple-kernel ridge regression, which is equivalent to
  group-regularization ridge regression when using one linear kernel per group
  of features:

  - :class:`~himalaya.kernel_ridge.MultipleKernelRidgeCV` (scikit-learn-compatible estimator)
  - :func:`~himalaya.kernel_ridge.solve_multiple_kernel_ridge_random_search` (function)
  - :func:`~himalaya.kernel_ridge.solve_multiple_kernel_ridge_hyper_gradient` (function)

.. note:: "Group ridge regression" is also sometimes called "Banded ridge regression".

WeightedKernelRidge
-------------------

To extend kernel ridge to group-regularization, we can compute the kernel as a
weighted sum of multiple kernels, :math:`K = \sum_{i=1}^m e^{\delta_i} K_i`.
Then, we can use :math:`K_i = X_i X_i^\top` for different groups of features
:math:`X_i`. The model becomes:

.. math::
    w^* = \arg\min_w \left\|\sum_{i=1}^m e^{\delta_i} K_{i} w - y\right\|_2^2
    + \alpha \sum_{i=1}^m e^{\delta_i} w^\top K_{i} w.

This model is called weighted kernel ridge regresion. The log-kernel-weights
:math:`\delta_i` are here fixed. When all the targets use the same
log-kernel-weights, a single weighted kernel can be precomputed and used in a
kernel ridge regression. However, when the log-kernel-weights are different for
each target, the kernel sum cannot be precomputed, and the model requires some
specific algorithms to be fit.

.. admonition:: This model is implemented in

  - :class:`~himalaya.kernel_ridge.WeightedKernelRidge` (scikit-learn-compatible estimator)
  - :func:`~himalaya.kernel_ridge.solve_weighted_kernel_ridge_gradient_descent` (function)
  - :func:`~himalaya.kernel_ridge.solve_weighted_kernel_ridge_conjugate_gradient` (function)
  - :func:`~himalaya.kernel_ridge.solve_weighted_kernel_ridge_neumann_series` (function)


MultipleKernelRidgeCV
---------------------

In weighted kernel ridge regression, when the log-kernel-weights
:math:`\delta_i` are unknown, we can learn them over cross-validation. This
model is called multiple-kernel ridge regression. When the kernels are defined
by :math:`K_i = X_i X_i^\top` for different groups of features :math:`X_i`,
multiple-kernel ridge regression is equivalent with group-ridge regression
(aka banded ridge regression).

.. admonition:: This model is implemented in

  - :class:`~himalaya.kernel_ridge.MultipleKernelRidgeCV` (scikit-learn-compatible estimator)
  - :func:`~himalaya.kernel_ridge.solve_multiple_kernel_ridge_hyper_gradient` (function)
  - :func:`~himalaya.kernel_ridge.solve_multiple_kernel_ridge_random_search` (function)


.. include:: flowchart.rst

References
~~~~~~~~~~

.. [1] Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased
  estimation for nonorthogonal problems. Technometrics, 12(1), 55-67.

.. [2] Saunders, C., Gammerman, A., & Vovk, V. (1998). Ridge regression
  learning algorithm in dual variables.

.. [3] Nunez-Elizalde, A. O., Huth, A. G., & Gallant, J. L. (2019). Voxelwise
  encoding models with non-spherical multivariate normal priors. Neuroimage,
  197, 482-492.

.. [4] Dupré La Tour, T., Eickenberg, M., Nunez-Elizalde, A.O., & Gallant, J. L. (2022).
   Feature-space selection with banded ridge regression. NeuroImage.
