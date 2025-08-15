import pytest
import numpy as np

from himalaya.backend import set_backend
from himalaya.backend import ALL_BACKENDS

from himalaya.utils import compute_lipschitz_constants
from himalaya.utils import generate_multikernel_dataset
from himalaya.utils import assert_array_almost_equal


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
    # The torch_mps backend will automatically skip this test if MPS is not available
    backend = set_backend('torch_mps')

    # Create test arrays that are close but require precision > 4
    x = backend.asarray([1.0, 2.0, 3.0])
    y = backend.asarray([1.00001, 2.00001, 3.00001])  # diff ~1e-5

    # Test that decimal > 4 triggers warning and auto-reduction
    with pytest.warns(UserWarning, match="Reducing precision from decimal=6 to decimal=4"):
        assert_array_almost_equal(x, y, decimal=6)

    # Test that decimal <= 4 doesn't trigger warning
    import warnings
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        assert_array_almost_equal(x, y, decimal=4)
    # Filter out unrelated warnings (like float64->float32 conversion)
    precision_warnings = [w for w in warning_list if "Reducing precision" in str(w.message)]
    assert len(precision_warnings) == 0

    # Test that other backends are unaffected
    for backend_name in ['numpy', 'torch']:
        try:
            backend = set_backend(backend_name)
            x = backend.asarray([1.0, 2.0, 3.0])
            y = backend.asarray([1.0, 2.0, 3.0])

            with warnings.catch_warnings(record=True) as warning_list:
                warnings.simplefilter("always")
                assert_array_almost_equal(x, y, decimal=6)
            # Should not have precision reduction warnings
            precision_warnings = [w for w in warning_list if "Reducing precision" in str(w.message)]
            assert len(precision_warnings) == 0
        except Exception:
            # Skip if backend not available
            pass
