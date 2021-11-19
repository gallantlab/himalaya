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
