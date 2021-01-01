import numpy as np
import sklearn
import pytest

from himalaya.backend import set_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.validation import _assert_all_finite


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
