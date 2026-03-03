import numpy as np
import pytest

from himalaya.backend import ALL_BACKENDS, set_backend
from himalaya.utils import (
    assert_array_almost_equal,
    compute_lipschitz_constants,
    generate_multikernel_dataset,
    skip_torch_mps_precision_checks,
    to_numpy_float64,
)


@pytest.mark.parametrize('kernelize', ["XXT", "XTX", "X"])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_compute_lipschitz_constants(backend, kernelize):
    backend = set_backend(backend)

    Xs = backend.randn(3, 5, 6)
    if kernelize == "X":
        XTs = backend.transpose(Xs, (0, 2, 1))
        Xs = backend.matmul(XTs, Xs)

    L = compute_lipschitz_constants(Xs)
    assert L.ndim == 1
    assert L.shape[0] == Xs.shape[0]


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_compute_lipschitz_constants_error(backend):
    backend = set_backend(backend)

    Xs = backend.randn(3, 5, 6)
    with pytest.raises(ValueError):
        compute_lipschitz_constants(Xs, "wrong")


# A small number of sets of parameters
_parameters = {
    "params_1":
    dict(n_kernels=4, n_targets=50, n_samples_train=100, n_samples_test=40,
         kernel_weights=None, n_features_list=[10, 10, 20, 5]),
    "params_2":
    dict(n_kernels=3, n_targets=40, n_samples_train=90, n_samples_test=40,
         kernel_weights=np.random.rand(40, 3), n_features_list=None),
}


@pytest.mark.parametrize("name", ["params_1", "params_2"])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_generate_multikernel_dataset(backend, name):
    backend = set_backend(backend)

    kwargs = _parameters[name]

    (X_train, X_test, Y_train, Y_test, kernel_weights,
     n_features_list) = generate_multikernel_dataset(**kwargs)

    assert X_train.shape[0] == kwargs["n_samples_train"]
    assert X_test.shape[0] == kwargs["n_samples_test"]
    assert Y_train.shape[0] == kwargs["n_samples_train"]
    assert Y_test.shape[0] == kwargs["n_samples_test"]
    assert Y_train.shape[1] == kwargs["n_targets"]
    assert Y_test.shape[1] == kwargs["n_targets"]
    assert len(n_features_list) == kwargs["n_kernels"]
    assert kernel_weights.shape[1] == kwargs["n_kernels"]
    assert kernel_weights.shape[0] == kwargs["n_targets"]

    if kwargs["kernel_weights"] is not None:
        assert_array_almost_equal(kwargs["kernel_weights"],
                                  kernel_weights)
    if kwargs["n_features_list"] is not None:
        assert np.sum(kwargs["n_features_list"]) == X_train.shape[1]
        assert np.sum(kwargs["n_features_list"]) == X_test.shape[1]


def test_assert_array_almost_equal_torch_mps_precision_warning():
    """Test that torch_mps backend automatically reduces precision and warns."""
    from unittest.mock import patch

    # Mock a torch_mps-like backend so this test runs on CI (no MPS needed)
    mock_backend = type('MockBackend', (), {
        'name': 'torch_mps',
        'to_numpy': staticmethod(lambda x: np.asarray(x)),
    })()

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0001, 2.0001, 3.0001])  # diff ~1e-4

    with patch('himalaya.utils.get_backend', return_value=mock_backend):
        # decimal > 3 triggers warning and auto-reduction
        with pytest.warns(UserWarning, match="Reducing precision from decimal=6 to decimal=3"):
            assert_array_almost_equal(x, y, decimal=6)

        # decimal <= 3 should not warn
        import warnings
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            assert_array_almost_equal(x, y, decimal=3)
        precision_warnings = [w for w in warning_list
                              if "Reducing precision" in str(w.message)]
        assert len(precision_warnings) == 0

    # Non-torch_mps backends should not warn
    set_backend('numpy')
    x_eq = np.array([1.0, 2.0, 3.0])
    y_eq = np.array([1.0, 2.0, 3.0])
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        assert_array_almost_equal(x_eq, y_eq, decimal=6)
    precision_warnings = [w for w in warning_list
                          if "Reducing precision" in str(w.message)]
    assert len(precision_warnings) == 0


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_to_numpy_float64(backend):
    """Test that to_numpy_float64 returns float64 numpy arrays."""
    backend = set_backend(backend)
    x = backend.asarray([1.0, 2.0, 3.0])
    result = to_numpy_float64(x)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


class _MockBackend:
    """Mock backend for testing skip_torch_mps_precision_checks without MPS."""
    def __init__(self, name):
        self.name = name


class _MockEstimator:
    """Mock estimator for testing purposes."""
    def __init__(self, name):
        self.__class__.__name__ = name


class _MockCheck:
    """Mock sklearn check function for testing purposes."""
    def __init__(self, func_name):
        self.func = type('MockFunc', (), {'__name__': func_name})()


def test_skip_torch_mps_precision_checks():
    """Test skip_torch_mps_precision_checks with all default cases."""
    mps_backend = _MockBackend("torch_mps")

    # All default estimator/check combinations should be skipped
    default_cases = [
        ('KernelRidge_', 'check_methods_subset_invariance'),
        ('KernelRidge_', 'check_sample_weight_equivalence_on_dense_data'),
        ('KernelRidge_', 'check_sample_weight_equivalence_on_sparse_data'),
        ('KernelRidgeCV_', 'check_methods_subset_invariance'),
        ('Kernelizer_', 'check_methods_subset_invariance'),
        ('WeightedKernelRidge_', 'check_sample_weight_equivalence_on_dense_data'),
        ('WeightedKernelRidge_', 'check_sample_weight_equivalence_on_sparse_data'),
        ('WeightedKernelRidge_', 'check_methods_subset_invariance'),
        ('MultipleKernelRidgeCV_', 'check_methods_subset_invariance'),
    ]
    for estimator_name, check_name in default_cases:
        estimator = _MockEstimator(estimator_name)
        check = _MockCheck(check_name)
        assert skip_torch_mps_precision_checks(
            mps_backend, estimator, check) is True, \
            f"Should skip {estimator_name}.{check_name}"

    # Non-sensitive checks should NOT be skipped
    estimator = _MockEstimator('KernelRidge_')
    for check_name in ['check_estimator_repr', 'check_fit_score_takes_y']:
        check = _MockCheck(check_name)
        assert skip_torch_mps_precision_checks(
            mps_backend, estimator, check) is False

    # Unknown estimator should NOT be skipped
    estimator = _MockEstimator('UnknownEstimator')
    check = _MockCheck('check_methods_subset_invariance')
    assert skip_torch_mps_precision_checks(
        mps_backend, estimator, check) is False

    # Non-torch_mps backends should NOT be skipped
    for name in ["numpy", "torch", "torch_cuda"]:
        backend = _MockBackend(name)
        estimator = _MockEstimator('KernelRidge_')
        check = _MockCheck('check_methods_subset_invariance')
        assert skip_torch_mps_precision_checks(
            backend, estimator, check) is False

    # Check without func attribute should NOT be skipped
    check_no_func = type('MockCheckNoFunc', (), {})()
    assert skip_torch_mps_precision_checks(
        mps_backend, _MockEstimator('KernelRidge_'), check_no_func) is False

    # Backend without name attribute should NOT be skipped
    backend_no_name = type('MockBackendNoName', (), {})()
    assert skip_torch_mps_precision_checks(
        backend_no_name, _MockEstimator('KernelRidge_'),
        _MockCheck('check_methods_subset_invariance')) is False

    # Custom config should work
    custom_config = {'CustomEstimator': ['custom_check']}
    assert skip_torch_mps_precision_checks(
        mps_backend, _MockEstimator('CustomEstimator'),
        _MockCheck('custom_check'), custom_config) is True
    assert skip_torch_mps_precision_checks(
        mps_backend, _MockEstimator('CustomEstimator'),
        _MockCheck('other_check'), custom_config) is False
