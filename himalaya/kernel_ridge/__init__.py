from ._solvers import solve_weighted_kernel_ridge_gradient_descent
from ._solvers import solve_weighted_kernel_ridge_conjugate_gradient
from ._solvers import solve_weighted_kernel_ridge_neumann_series
from ._solvers import solve_kernel_ridge_eigenvalues
from ._solvers import solve_kernel_ridge_gradient_descent
from ._solvers import solve_kernel_ridge_conjugate_gradient
from ._solvers import KERNEL_RIDGE_SOLVERS
from ._solvers import WEIGHTED_KERNEL_RIDGE_SOLVERS
from ._hyper_gradient import solve_multiple_kernel_ridge_hyper_gradient
from ._hyper_gradient import MULTIPLE_KERNEL_RIDGE_SOLVERS
from ._random_search import solve_multiple_kernel_ridge_random_search
from ._random_search import generate_dirichlet_samples
from ._random_search import solve_kernel_ridge_cv_eigenvalues
from ._predictions import predict_weighted_kernel_ridge
from ._predictions import predict_and_score_weighted_kernel_ridge
from ._sklearn_api import KernelRidge
from ._sklearn_api import KernelRidgeCV
from ._sklearn_api import MultipleKernelRidgeCV
from ._sklearn_api import WeightedKernelRidge
from ._kernels import PAIRWISE_KERNEL_FUNCTIONS
from ._kernels import linear_kernel
from ._kernels import polynomial_kernel
from ._kernels import rbf_kernel
from ._kernels import sigmoid_kernel
from ._kernels import cosine_similarity_kernel

__all__ = [
    # kernel ridge solvers
    solve_weighted_kernel_ridge_gradient_descent,
    solve_weighted_kernel_ridge_conjugate_gradient,
    solve_weighted_kernel_ridge_neumann_series,
    solve_kernel_ridge_cv_eigenvalues,
    solve_kernel_ridge_eigenvalues,
    solve_kernel_ridge_gradient_descent,
    solve_kernel_ridge_conjugate_gradient,
    KERNEL_RIDGE_SOLVERS,
    WEIGHTED_KERNEL_RIDGE_SOLVERS,
    # multiple kernel ridge solvers
    MULTIPLE_KERNEL_RIDGE_SOLVERS,
    solve_multiple_kernel_ridge_hyper_gradient,
    solve_multiple_kernel_ridge_random_search,
    # helpers
    generate_dirichlet_samples,
    predict_weighted_kernel_ridge,
    predict_and_score_weighted_kernel_ridge,
    # scikit-learn API
    KernelRidge,
    KernelRidgeCV,
    MultipleKernelRidgeCV,
    WeightedKernelRidge,
    # kernels
    PAIRWISE_KERNEL_FUNCTIONS,
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
    cosine_similarity_kernel,
]
