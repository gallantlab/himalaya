import numpy as np
import sklearn
import pytest

from himalaya.validation import _assert_all_finite


def test_suppress_validation():
    X = np.array([0, np.inf])
    with pytest.raises(ValueError):
        _assert_all_finite(X, True)
    sklearn.set_config(assume_finite=True)
    _assert_all_finite(X, True)
    sklearn.set_config(assume_finite=False)
    with pytest.raises(ValueError):
        _assert_all_finite(X, True)
