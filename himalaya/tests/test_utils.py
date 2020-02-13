import pytest

from himalaya.backend import change_backend
from himalaya.backend import ALL_BACKENDS

from himalaya.utils import compute_lipschitz_constants


@pytest.mark.parametrize('kernelize', ["XXT", "XTX", "X"])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_compute_lipschitz_constants(backend, kernelize):
    backend = change_backend(backend)

    Xs = backend.randn(3, 5, 6)
    if kernelize == "X":
        XTs = backend.transpose(Xs, (0, 2, 1))
        Xs = backend.matmul(XTs, Xs)

    L = compute_lipschitz_constants(Xs)
    assert L.ndim == 1
    assert L.shape[0] == Xs.shape[0]


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_compute_lipschitz_constants_error(backend):
    backend = change_backend(backend)

    Xs = backend.randn(3, 5, 6)
    with pytest.raises(ValueError):
        compute_lipschitz_constants(Xs, "wrong")
