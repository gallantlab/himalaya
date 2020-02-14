import numpy as np
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
            backend.assert_allclose(
                backend.max(array, axis=axis),
                backend.apply_argmax(array, argmax, axis=axis),
            )


@pytest.mark.parametrize('dtype_str', ["float32", "float64"])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_std_float64(backend, dtype_str):
    backend = change_backend(backend)
    for array in [
            backend.randn(1),
            backend.randn(10),
            backend.randn(10, 1),
            backend.randn(10, 4),
            backend.randn(10, 1, 8),
            backend.randn(10, 4, 8),
    ]:
        array = backend.asarray(array, dtype=dtype_str)
        array_64 = backend.asarray(array, dtype="float64")
        for axis in range(array.ndim):
            result = backend.std_float64(array, axis=axis)
            reference = np.asarray(array_64).std(axis=axis, dtype="float64")
            reference = backend.asarray(reference, dtype=dtype_str)
            backend.assert_allclose(result, reference)
