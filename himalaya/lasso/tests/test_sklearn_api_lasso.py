import numpy as np
import pytest
import sklearn.utils.estimator_checks

from himalaya.backend import set_backend
from himalaya.backend import get_backend
from himalaya.backend import ALL_BACKENDS

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
        result = backend.to_numpy(super().predict(X))
        # Convert to float64 for sklearn compatibility if backend is torch_mps
        if backend.name == "torch_mps" and result.dtype == np.float32:
            result = result.astype(np.float64)
        return result

    def score(self, X, y):
        from himalaya.validation import check_array
        from himalaya.scoring import r2_score
        backend = get_backend()

        y_pred = super().predict(X)
        y_true = check_array(y, dtype=self.dtype_, ndim=self.coef_.ndim)

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
