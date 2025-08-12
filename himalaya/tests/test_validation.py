import numpy as np
import sklearn
import pytest

from himalaya.backend import set_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.validation import _assert_all_finite
from himalaya.validation import check_cv
from himalaya.validation import validate_data


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


class DummyEstimator:
    """Dummy estimator for testing validate_data"""
    def __init__(self):
        pass


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_validate_data_X_only(backend):
    backend = set_backend(backend)
    X = backend.asarray([[1, 2], [3, 4]])
    estimator = DummyEstimator()
    
    # Test reset=True (fit behavior)
    X_val = validate_data(estimator, X, reset=True, ndim=2)
    assert hasattr(estimator, 'n_features_in_')
    assert estimator.n_features_in_ == 2
    assert X_val.shape == (2, 2)
    
    # Test reset=False (predict behavior) - should work
    X_val2 = validate_data(estimator, X, reset=False, ndim=2)
    assert X_val2.shape == (2, 2)
    
    # Test reset=False with wrong number of features - should fail
    X_wrong = backend.asarray([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="X has 3 features.*expecting 2 features"):
        validate_data(estimator, X_wrong, reset=False, ndim=2)


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_validate_data_X_and_y(backend):
    backend = set_backend(backend)
    X = backend.asarray([[1, 2], [3, 4]])
    y = backend.asarray([1, 0])
    estimator = DummyEstimator()
    
    # Test with both X and y - X gets ndim=2, y gets default [1,2]
    X_val, y_val = validate_data(estimator, X, y, reset=True, ndim=2)
    assert hasattr(estimator, 'n_features_in_')
    assert estimator.n_features_in_ == 2
    assert X_val.shape == (2, 2)
    assert y_val.shape == (2,)


@pytest.mark.parametrize('backend', ALL_BACKENDS) 
def test_validate_data_no_validation(backend):
    backend = set_backend(backend)
    estimator = DummyEstimator()
    
    # Test X='no_validation' only
    result = validate_data(estimator, X='no_validation', reset=True)
    assert result == 'no_validation'
    
    # Test y='no_validation'
    X = backend.asarray([[1, 2], [3, 4]])
    X_val = validate_data(estimator, X, y='no_validation', reset=True, ndim=2)
    assert X_val.shape == (2, 2)
    assert hasattr(estimator, 'n_features_in_')
    
    # Test both 'no_validation'
    result = validate_data(estimator, X='no_validation', y='no_validation', reset=True)
    assert result == 'no_validation'


def test_validate_data_error_without_n_features_in():
    # Test that predict without prior fit doesn't crash
    estimator = DummyEstimator()
    X = np.array([[1, 2], [3, 4]])
    
    # Should work fine if estimator doesn't have n_features_in_ yet
    X_val = validate_data(estimator, X, reset=False, ndim=2)
    assert X_val.shape == (2, 2)
