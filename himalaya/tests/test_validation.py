import numpy as np
import sklearn
import pytest

from himalaya.backend import set_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.validation import _assert_all_finite
from himalaya.validation import check_cv


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_suppress_validation(backend):
    backend = set_backend(backend)
    X = backend.asarray([0, np.inf])
    with pytest.raises(ValueError):
        _assert_all_finite(X, True)
    sklearn.set_config(assume_finite=True)
    _assert_all_finite(X, True)
    sklearn.set_config(assume_finite=False)
    with pytest.raises(ValueError):
        _assert_all_finite(X, True)


def test_check_cv():
    cv = [([0, 1], [2]), ([0, 2], [1]), ([1, 2], [0])]

    # works because cv does not exceed y.shape[0]
    y = np.zeros(4)
    check_cv(cv, y)
    # fails because cv does exceed y.shape[0]
    with pytest.raises(ValueError, match="exceed number of samples"):
        y = np.zeros(2)
        check_cv(cv, y)
