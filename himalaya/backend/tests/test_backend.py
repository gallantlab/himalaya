import pytest

from himalaya.backend import change_backend
from himalaya.backend import ALL_BACKENDS


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_apply_argmax(backend):
    backend = change_backend(backend)
    for array in [
            backend.randn(1),
            backend.randn(10),
            backend.randn(10, 1),
            backend.randn(10, 4),
            backend.randn(10, 1, 8),
            backend.randn(10, 4, 8),
    ]:
        for axis in range(array.ndim):
            argmax = backend.argmax(array, axis=axis)
            backend.assert_array_equal(
                backend.max(array, axis=axis),
                backend.apply_argmax(array, argmax, axis=axis),
            )
