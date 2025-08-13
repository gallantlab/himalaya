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


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_validate_data_3d_feature_axis(backend):
    """Test feature axis handling for 3D arrays (precomputed kernels).

    This test ensures that validate_data correctly handles the feature dimension
    for 3D precomputed kernel arrays where:
    - During fit: shape (n_kernels, n_samples_train, n_samples_train)
    - During predict: shape (n_kernels, n_samples_test, n_samples_train)

    The "feature" dimension (last axis) should be consistent between fit and predict.
    """
    backend = set_backend(backend)
    estimator = DummyEstimator()

    # Simulate fit with 3D precomputed kernels: (n_kernels=2, n_train=600, n_train=600)
    X_fit = backend.asarray(np.random.randn(2, 600, 600))
    X_val_fit = validate_data(estimator, X_fit, reset=True, ndim=3)

    # Should store n_features_in_ as last axis (600)
    assert hasattr(estimator, 'n_features_in_')
    assert estimator.n_features_in_ == 600
    assert X_val_fit.shape == (2, 600, 600)

    # Simulate predict with 3D kernels: (n_kernels=2, n_test=300, n_train=600)
    # The middle axis changes (test samples) but last axis stays same (training samples)
    X_predict = backend.asarray(np.random.randn(2, 300, 600))
    X_val_predict = validate_data(estimator, X_predict, reset=False, ndim=3)

    # Should validate successfully - last axis (600) matches stored n_features_in_
    assert X_val_predict.shape == (2, 300, 600)

    # Test failure case: wrong last dimension
    X_wrong = backend.asarray(np.random.randn(2, 300, 500))  # wrong last dim
    with pytest.raises(ValueError, match="X has 500 features.*expecting 600 features"):
        validate_data(estimator, X_wrong, reset=False, ndim=3)

    # Test explicit feature_axis parameter
    estimator2 = DummyEstimator()

    # Using feature_axis=1 (middle axis) for 3D - this was the old buggy behavior
    validate_data(estimator2, X_fit, reset=True, ndim=3, feature_axis=1)
    assert estimator2.n_features_in_ == 600  # middle axis

    # This should fail with the test data because middle axis is different (300 vs 600)
    with pytest.raises(ValueError, match="X has 300 features.*expecting 600 features"):
        validate_data(estimator2, X_predict, reset=False, ndim=3, feature_axis=1)
