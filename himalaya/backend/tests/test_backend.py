import pytest

from himalaya.backend import set_backend
from himalaya.backend import get_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.backend._utils import _dtype_to_str
from himalaya.utils import assert_array_almost_equal


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_set_backend_correct(backend):
    # test the change of backend
    module = set_backend(backend)
    assert module.__name__.split('.')[-1] == backend

    # test idempotence
    module = set_backend(set_backend(backend))
    assert module.__name__.split('.')[-1] == backend

    # test set and get
    module = set_backend(get_backend())
    assert module.__name__.split('.')[-1] == backend

    assert set_backend(backend)


def test_set_backend_incorrect():
    for backend in ["wrong", ["numpy"], True, None, 10]:
        with pytest.raises(ValueError):
            set_backend(backend)
        with pytest.raises(ValueError):
            set_backend(backend, on_error="raise")
        with pytest.warns(Warning):
            set_backend(backend, on_error="warn")
        with pytest.raises(ValueError):
            set_backend(backend, on_error="foo")


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_apply_argmax(backend):
    backend = set_backend(backend)
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
            assert_array_almost_equal(
                backend.max(array, axis=axis),
                backend.apply_argmax(array, argmax, axis=axis),
            )


@pytest.mark.parametrize('dtype_str', ["float32", "float64"])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_std_float64(backend, dtype_str):
    backend = set_backend(backend)
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
    backend = set_backend(backend)
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed.")
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


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_eigh(backend):
    import scipy.linalg
    backend = set_backend(backend)

    array = backend.randn(10, 20)
    array = backend.asarray(array, dtype='float64')
    kernel = array @ array.T

    values, vectors = backend.eigh(kernel)
    values_ref, vectors_ref = scipy.linalg.eigh(backend.to_numpy(kernel))

    assert_array_almost_equal(values, values_ref)

    # vectors can be flipped in sign
    assert vectors.shape == vectors_ref.shape
    for ii in range(vectors.shape[1]):
        try:
            assert_array_almost_equal(vectors[:, ii], vectors_ref[:, ii])
        except AssertionError:
            assert_array_almost_equal(vectors[:, ii], -vectors_ref[:, ii])


@pytest.mark.parametrize('backend', ALL_BACKENDS)
@pytest.mark.parametrize('full_matrices', [True, False])
@pytest.mark.parametrize('three_dim', [True, False])
def test_svd(backend, full_matrices, three_dim):
    import numpy.linalg
    backend = set_backend(backend)

    if three_dim:
        array = backend.randn(3, 5, 7)
    else:
        array = backend.randn(5, 7)

    array = backend.asarray(array, dtype='float64')

    U, s, V = backend.svd(array, full_matrices=full_matrices)
    U_ref, s_ref, V_ref = numpy.linalg.svd(backend.to_numpy(array),
                                           full_matrices=full_matrices)

    assert_array_almost_equal(s, s_ref)

    if not three_dim:
        U_ref = U_ref[None]
        U = U[None]
        V_ref = V_ref[None]
        V = V[None]

    # vectors can be flipped in sign
    assert U.shape == U_ref.shape
    assert V.shape == V_ref.shape
    for kk in range(U.shape[0]):
        for ii in range(U.shape[2]):
            try:
                assert_array_almost_equal(U[kk, :, ii], U_ref[kk, :, ii])
                assert_array_almost_equal(V[kk, ii, :], V_ref[kk, ii, :])
            except AssertionError:
                assert_array_almost_equal(U[kk, :, ii], -U_ref[kk, :, ii])
                assert_array_almost_equal(V[kk, ii, :], -V_ref[kk, ii, :])


@pytest.mark.parametrize('backend_out', ALL_BACKENDS)
@pytest.mark.parametrize('backend_in', ALL_BACKENDS)
def test_changed_backend_asarray(backend_in, backend_out):
    backend = set_backend(backend_in)
    array_in = backend.asarray([1.2, 2.4, 4.8])
    assert array_in is not None

    # change the backend, and cast to the correct class
    backend = set_backend(backend_out)
    array_out = backend.asarray(array_in)
    assert array_out is not None

    if backend_in == backend_out or backend_in[:5] == backend_out[:5]:
        # assert the class did not change
        assert array_in.__class__ == array_out.__class__
    else:
        # assert the class did change
        assert array_in.__class__ != array_out.__class__

    # assert the new class is correct
    array_out2 = backend.randn(3)
    assert array_out.__class__ == array_out2.__class__

    # test check_arrays
    array_out3, array_out4, array_out5 = backend.check_arrays(
        array_in, array_in, [array_in])
    assert array_out.__class__ == array_out3.__class__
    assert array_out.__class__ == array_out4.__class__
    assert array_out.__class__ == array_out5[0].__class__


@pytest.mark.parametrize('dtype_out', ["float32", "float64"])
@pytest.mark.parametrize('dtype_in', ["float32", "float64"])
@pytest.mark.parametrize('backend_out', ALL_BACKENDS)
@pytest.mark.parametrize('backend_in', ALL_BACKENDS)
def test_asarray_dtype(backend_in, backend_out, dtype_in, dtype_out):
    backend = set_backend(backend_in)
    array_in = backend.asarray([1.2, 2.4, 4.8], dtype=dtype_in)
    assert _dtype_to_str(array_in.dtype) == dtype_in

    backend = set_backend(backend_out)
    array_out = backend.asarray(array_in, dtype=dtype_out)
    assert _dtype_to_str(array_out.dtype) == dtype_out


def test_dtype_to_str_wrong_input():
    assert _dtype_to_str(None) is None

    with pytest.raises(NotImplementedError):
        _dtype_to_str(42)
