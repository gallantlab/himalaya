import pytest

from himalaya.backend import change_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.utils import assert_array_almost_equal


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
            reference = backend.to_numpy(array_64).std(axis=axis,
                                                       dtype="float64")
            reference = backend.asarray(reference, dtype=dtype_str)
            assert_array_almost_equal(result, reference)


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_diagonal_view(backend):
    backend = change_backend(backend)
    import torch
    import numpy as np

    for array in [
            backend.randn(10, 4),
            backend.randn(10, 4).T,
            backend.randn(10, 4, 8),
            backend.randn(10, 4, 8).T,
            backend.randn(3, 4, 8, 5),
    ]:
        for axis1 in range(array.ndim):
            for axis2 in range(array.ndim):
                if axis1 != axis2:
                    result = backend.diagonal_view(array, axis1=axis1,
                                                   axis2=axis2)
                    # compare with torch diagonal
                    reference = torch.diagonal(
                        torch.from_numpy(backend.to_numpy(array)), dim1=axis1,
                        dim2=axis2)
                    assert_array_almost_equal(result, reference)
                    # compare with numpy diagonal
                    reference = np.diagonal(backend.to_numpy(array),
                                            axis1=axis1, axis2=axis2)
                    assert_array_almost_equal(result, reference)
                    # test that this is a modifiable view
                    result += 1
                    reference = np.diagonal(backend.to_numpy(array),
                                            axis1=axis1, axis2=axis2)
                    assert_array_almost_equal(result, reference)
