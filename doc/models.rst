Model descriptions
==================

This package implements a number of models.

KernelRidge
-----------

Let :math:`X\in \mathbb{R}^{n\times p}` be a feature matrix with :math:`n`
samples and :math:`p` features,  :math:`y\in \mathbb{R}^n` a target vector, and
:math:`\alpha > 0` a fixed regularization hyperparameter. Ridge regression
[1]_ considers the weight vector :math:`b^*\in \mathbb{R}^p` defined as

.. math::
    b^* = \arg\min_b \|Xb - y\|_2^2 + \alpha \|b\|_2^2.

The equation has a  closed-form solution :math:`b^* = M y`, where :math:`M =
(X^\top X + \alpha I_p)^{-1}X^\top \in  \mathbb{R}^{p \times n}`. By the
Woodbury matrix identity, :math:`b^*` can be written :math:`b^* =
X^\top(XX^\top + \alpha I_n)^{-1}y`, or :math:`b^* = X^\top w^*` for some
:math:`w^*\in \mathbb{R}^n`. Noting the linear kernel :math:`K = X X^\top \in
\mathbb{R}^{n\times n}`, this leads to the \emph{equivalent} formulation

.. math::
    w^* = \arg\min_w \|Kw - y\|_2^2 + \alpha w^\top Kw.

This model can be extended to arbitrary positive semidefinite kernels
:math:`K`, leading to the more general kernel ridge regression [2]_.

This model is implemented in a scikit-learn-compatible estimator
:class:`~himalaya.kernel_ridge.KernelRidge`, or through the functions
:func:`~himalaya.kernel_ridge.solve_kernel_ridge_eigenvalues`,
:func:`~himalaya.kernel_ridge.solve_kernel_ridge_gradient_descent`, and
:func:`~himalaya.kernel_ridge.solve_kernel_ridge_conjugate_gradient`.


KernelRidgeCV
-------------

In practice in kernel ridge regression, because the hyperparameter
:math:`\alpha` is unknown, it is typically selected through a grid-search with
cross-validation. In cross-validation, we split the data set into a training
set :math:`(X_{train}, y_{train})` and a validation set :math:`(X_{val},
y_{val})`. Then, we train the model on the training set, and evaluate the
generalization performances on the validation set. We perform this process for
multiple hyperparameter candidates :math:`\alpha`, typically defined over a
grid of log-spaced values. Finally, we keep the candidate leading to the best
generalization performance, as measured by the validation loss, averaged over
all cross-validation splits.

This model is implemented in a scikit-learn-compatible estimator
:class:`~himalaya.kernel_ridge.KernelRidgeCV`, or through the function
:func:`~himalaya.kernel_ridge.solve_kernel_ridge_cv_eigenvalues`.

WeightedKernelRidge
-------------------

Kernel ridge regression can be naturally extend to a weighted sum of multiple
kernels, :math:`K = \sum_{i=1}^m e^{\delta_i} K_i`. A typical example is to use
:math:`K_i = X_i X_i^\top` for different subsets of features :math:`X_i`.
The model becomes:

.. math::
    w^* = \arg\min_w \left\|\sum_{i=1}^m e^{\delta_i} K_{i} w - y\right\|_2^2
    + \alpha \sum_{i=1}^m e^{\delta_i} w^\top K_{i} w.

Contrarily to :class:`~himalaya.kernel_ridge.MultipleKernelRidgeCV`, this model
does not optimize the log kernel weights :math:`\delta_i`. However, it is not
equivalent to :class:`~himalaya.kernel_ridge.KernelRidge`, since the log kernel
weights :math:`\delta_i` can be different for each target, therefore the
kernel sum is not precomputed.

This model is a scikit-learn-compatible estimator
:class:`~himalaya.kernel_ridge.WeightedKernelRidgeCV`, or through the functions
:func:`~himalaya.kernel_ridge.solve_weighted_kernel_ridge_gradient_descent`,
:func:`~himalaya.kernel_ridge.solve_weighted_kernel_ridge_conjugate_gradient`,
and :func:`~himalaya.kernel_ridge.solve_weighted_kernel_ridge_neumann_series`.

MultipleKernelRidgeCV
---------------------

In weighted kernel ridge regression, when the log kernel weights
:math:`\delta_i` are unknown, we can learn them over cross-validation.

This model is a scikit-learn-compatible estimator
:class:`~himalaya.kernel_ridge.MultipleKernelRidgeCV`, or through the functions
:func:`~himalaya.kernel_ridge.solve_multiple_kernel_ridge_hyper_gradient`, and
:func:`~himalaya.kernel_ridge.solve_multiple_kernel_ridge_random_search`.

SparseGroupLassoCV
------------------

...

References
~~~~~~~~~~

.. [1] Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased
  estimation for nonorthogonal problems. Technometrics, 12(1), 55-67.

.. [2] Saunders, C., Gammerman, A., & Vovk, V. (1998). Ridge regression
  learning algorithm in dual variables.
