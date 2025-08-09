import pytest
import sklearn.utils.estimator_checks
from sklearn.utils.validation import check_is_fitted

from himalaya.backend import set_backend
from himalaya.backend import get_backend
from himalaya.backend import ALL_BACKENDS
from himalaya._sklearn_compat import validate_data
from himalaya.validation import check_array
from himalaya.scoring import r2_score

from himalaya.lasso import SparseGroupLassoCV

###############################################################################
# scikit-learn.utils.estimator_checks


class SparseGroupLassoCV_(SparseGroupLassoCV):
    """Cast predictions to numpy arrays, to be used in scikit-learn tests.

    Used for testing only.
    """

    def __init__(self, groups=None, l1_regs=(0, 0.1), l21_regs=(0, 0.1),
                 solver="proximal_gradient", solver_params=None, cv=2):
        super().__init__(groups=groups, l1_regs=l1_regs, l21_regs=l21_regs,
                         solver=solver, solver_params=solver_params, cv=cv)

    def predict(self, X):
        backend = get_backend()
        # Use check_array directly like the main SparseGroupLassoCV predict method
        check_is_fitted(self, ['coef_'])
        # Get dtype from fitted estimator, fallback to default if not available
        dtype = getattr(self, 'dtype_', ["float32", "float64"])
        X = check_array(X, dtype=dtype, accept_sparse=False, ndim=2)
        return backend.to_numpy(super().predict(X))

    def score(self, X, y):
        backend = get_backend()

        # Validate data once to avoid double validation
        check_is_fitted(self, ['coef_'])
        # Get dtype from fitted estimator, fallback to default if not available
        dtype = getattr(self, 'dtype_', ["float32", "float64"])
        X = check_array(X, dtype=dtype, accept_sparse=False, ndim=2)
        y_true = check_array(y, dtype=self.dtype_, ndim=self.coef_.ndim)
        
        # Use internal prediction logic to avoid re-validation
        Y_hat = backend.to_numpy(X) @ backend.to_numpy(self.coef_)
        y_pred = backend.asarray_like(Y_hat, ref=X)

        if y_true.ndim == 1:
            return backend.to_numpy(
                r2_score(y_true[:, None], y_pred[:, None])[0])
        else:
            return backend.to_numpy(r2_score(y_true, y_pred))


@sklearn.utils.estimator_checks.parametrize_with_checks([
    SparseGroupLassoCV_(),
])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_check_estimator(estimator, check, backend):
    backend = set_backend(backend)
    check(estimator)
