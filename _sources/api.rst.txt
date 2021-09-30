.. _api_documentation:

=================
API Documentation
=================

Backend
=======

Public functions in ``himalaya.backend``.

.. currentmodule:: himalaya.backend

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: function.rst

   set_backend
   get_backend
   ALL_BACKENDS

|

_____

Kernel ridge
============

Public functions and classes in ``himalaya.kernel_ridge``.

.. currentmodule:: himalaya.kernel_ridge

Estimators
----------
Estimators compatible with the ``scikit-learn`` API.

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   :template: class.rst
   KernelRidge
   KernelRidgeCV
   WeightedKernelRidge
   MultipleKernelRidgeCV

   Kernelizer
   ColumnKernelizer
   :template: function.rst
   make_column_kernelizer



Solver functions
----------------
.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: function.rst

   KERNEL_RIDGE_SOLVERS
   solve_kernel_ridge_cv_eigenvalues
   solve_kernel_ridge_eigenvalues
   solve_kernel_ridge_gradient_descent
   solve_kernel_ridge_conjugate_gradient

   WEIGHTED_KERNEL_RIDGE_SOLVERS
   solve_weighted_kernel_ridge_gradient_descent
   solve_weighted_kernel_ridge_conjugate_gradient
   solve_weighted_kernel_ridge_neumann_series

   MULTIPLE_KERNEL_RIDGE_SOLVERS
   solve_multiple_kernel_ridge_hyper_gradient
   solve_multiple_kernel_ridge_random_search



Helpers
-------
.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: function.rst

   generate_dirichlet_samples
   predict_weighted_kernel_ridge
   predict_and_score_weighted_kernel_ridge
   primal_weights_kernel_ridge
   primal_weights_weighted_kernel_ridge



Kernels
-------
.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: function.rst

   PAIRWISE_KERNEL_FUNCTIONS
   linear_kernel
   polynomial_kernel
   rbf_kernel
   sigmoid_kernel
   cosine_similarity_kernel

|

_____


Lasso
=====

Public functions and classes in ``himalaya.lasso``.

.. currentmodule:: himalaya.lasso

Estimators
----------
Estimators compatible with the ``scikit-learn`` API.

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   :template: class.rst
   SparseGroupLassoCV

Solver functions
----------------
.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: function.rst

   solve_sparse_group_lasso
   solve_sparse_group_lasso_cv


|

_____

Ridge
=====

Public functions and classes in ``himalaya.ridge``.

.. currentmodule:: himalaya.ridge

Estimators
----------
Estimators compatible with the ``scikit-learn`` API.

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   :template: class.rst
   Ridge
   RidgeCV
   GroupRidgeCV
   BandedRidgeCV

   ColumnTransformerNoStack
   :template: function.rst
   make_column_transformer_no_stack

Solver functions
----------------
.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: function.rst

   RIDGE_SOLVERS
   solve_ridge_svd
   solve_ridge_cv_svd
   GROUP_RIDGE_SOLVERS
   BANDED_RIDGE_SOLVERS
   solve_group_ridge_random_search
   solve_banded_ridge_random_search


|

_____


Other modules
=============

Public functions and classes in other minor modules.

.. currentmodule:: himalaya

Progress bar
------------
.. autosummary::
   :toctree: _generated/
   :nosignatures:

   :template: class.rst
   progress_bar.ProgressBar
   :template: function.rst
   progress_bar.bar


Scoring functions
-----------------
.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: function.rst

   scoring.l2_neg_loss
   scoring.r2_score
   scoring.correlation_score
   scoring.r2_score_split


Utils
-----
.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: function.rst

   utils.compute_lipschitz_constants


Visualization
-------------
.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: function.rst

   viz.plot_alphas_diagnostic
