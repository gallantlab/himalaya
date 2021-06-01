from ._column import ColumnTransformerNoStack
from ._column import make_column_transformer_no_stack
from ._random_search import solve_group_ridge_random_search
from ._random_search import solve_ridge_cv_svd
from ._random_search import GROUP_RIDGE_SOLVERS
from ._solvers import solve_ridge_svd
from ._solvers import RIDGE_SOLVERS
from ._sklearn_api import Ridge
from ._sklearn_api import RidgeCV
from ._sklearn_api import GroupRidgeCV

# alternative names - kept for backward compatibility
BandedRidgeCV = GroupRidgeCV
solve_banded_ridge_random_search = solve_group_ridge_random_search
BANDED_RIDGE_SOLVERS = GROUP_RIDGE_SOLVERS

__all__ = [
    # column transformers
    "ColumnTransformerNoStack",
    "make_column_transformer_no_stack",
    # group ridge solvers
    "solve_group_ridge_random_search",
    "GROUP_RIDGE_SOLVERS",
    # ridge solvers
    "solve_ridge_svd",
    "solve_ridge_cv_svd",
    "RIDGE_SOLVERS",
    # sklearn API
    "Ridge",
    "RidgeCV",
    "GroupRidgeCV",
]
