import pytest
import sklearn.utils.estimator_checks
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from sklearn.base import clone

from himalaya.backend import set_backend
from himalaya.backend import get_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.utils import assert_array_almost_equal

from himalaya.kernel_ridge import Kernelizer
from himalaya.kernel_ridge import ColumnKernelizer
from himalaya.kernel_ridge import make_column_kernelizer
from himalaya.kernel_ridge._kernelizer import _end_with_a_kernel
from himalaya.kernel_ridge import KernelRidge
from himalaya.kernel_ridge import MultipleKernelRidgeCV


@pytest.mark.parametrize('kernel', Kernelizer.ALL_KERNELS)
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernelizer(backend, kernel):
    backend = set_backend(backend)

    X1 = backend.randn(10, 5)
    X2 = backend.randn(8, 5)

    function = Kernelizer.ALL_KERNELS[kernel]
    ker = Kernelizer(kernel=kernel)

    assert_array_almost_equal(function(X1), ker.fit_transform(X1))
    assert_array_almost_equal(function(X2, X1), ker.transform(X2))


@pytest.mark.parametrize('kernel', Kernelizer.ALL_KERNELS)
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_column_kernelizer_all_columns(backend, kernel):
    backend = set_backend(backend)

    X1 = backend.randn(10, 5)
    X2 = backend.randn(8, 5)

    function = Kernelizer.ALL_KERNELS[kernel]
    ker = ColumnKernelizer([
        ("name", Kernelizer(kernel=kernel), slice(0, 5)),
    ])

    assert ker.fit_transform(X1).shape == (1, 10, 10)
    assert ker.transform(X2).shape == (1, 8, 10)

    assert_array_almost_equal(function(X1), ker.fit_transform(X1)[0])
    assert_array_almost_equal(function(X2, X1), ker.transform(X2)[0])


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_column_kernelizer_passthrough(backend):
    backend = set_backend(backend)

    X1 = backend.randn(10, 5)
    X2 = backend.randn(8, 5)

    function = Kernelizer.ALL_KERNELS["linear"]
    ker = ColumnKernelizer([
        ("name", "passthrough", slice(0, 5)),
    ])

    assert ker.fit_transform(X1).shape == (1, 10, 10)
    assert ker.transform(X2).shape == (1, 8, 10)

    assert_array_almost_equal(function(X1), ker.fit_transform(X1)[0])
    assert_array_almost_equal(function(X2, X1), ker.transform(X2)[0])


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_column_kernelizer_remainder(backend):
    backend = set_backend(backend)

    X1 = backend.randn(10, 5)
    X2 = backend.randn(8, 5)

    function = Kernelizer.ALL_KERNELS["linear"]
    ker = ColumnKernelizer([
        ("name", Kernelizer(kernel="wrong"), []),
    ], remainder="passthrough")

    assert ker.fit_transform(X1).shape == (1, 10, 10)
    assert ker.transform(X2).shape == (1, 8, 10)

    assert_array_almost_equal(function(X1), ker.fit_transform(X1)[0])
    assert_array_almost_equal(function(X2, X1), ker.transform(X2)[0])


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_column_kernelizer_multiple(backend):
    backend = set_backend(backend)

    X1 = backend.randn(10, 5)
    X2 = backend.randn(8, 5)

    linear = Kernelizer.ALL_KERNELS["linear"]
    poly = Kernelizer.ALL_KERNELS["poly"]
    ker = ColumnKernelizer([
        ("name0", Kernelizer(kernel="linear"), [0, 1]),
        ("name1", Kernelizer(kernel="poly"), [2, 3]),
    ], remainder="passthrough")

    assert ker.fit_transform(X1).shape == (3, 10, 10)
    assert ker.transform(X2).shape == (3, 8, 10)

    assert_array_almost_equal(linear(X1[:, :2]), ker.fit_transform(X1)[0])
    assert_array_almost_equal(poly(X1[:, 2:4]), ker.fit_transform(X1)[1], 5)
    assert_array_almost_equal(linear(X1[:, 4:]), ker.fit_transform(X1)[2])
    assert_array_almost_equal(linear(X2[:, :2], X1[:, :2]),
                              ker.transform(X2)[0])
    assert_array_almost_equal(poly(X2[:, 2:4], X1[:, 2:4]),
                              ker.transform(X2)[1], 5)
    assert_array_almost_equal(linear(X2[:, 4:], X1[:, 4:]),
                              ker.transform(X2)[2])


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_column_kernelizer_pipeline(backend):
    backend = set_backend(backend)

    X1 = backend.randn(10, 5)
    X2 = backend.randn(8, 5)

    pipe = make_pipeline("passthrough", Kernelizer(kernel="poly"))

    function = Kernelizer.ALL_KERNELS["poly"]
    ker = ColumnKernelizer([
        ("name", pipe, slice(0, 4)),
    ])

    assert ker.fit_transform(X1).shape == (1, 10, 10)
    assert ker.transform(X2).shape == (1, 8, 10)

    assert_array_almost_equal(function(X1[:, :4]), ker.fit_transform(X1)[0])
    assert_array_almost_equal(function(X2[:, :4], X1[:, :4]),
                              ker.transform(X2)[0])


def test_column_kernelizer_end_with_a_kernel():
    with pytest.raises(ValueError):
        _end_with_a_kernel("passthrough")
    with pytest.raises(ValueError):
        _end_with_a_kernel(Kernelizer)

    assert _end_with_a_kernel(StandardScaler()) is False
    assert _end_with_a_kernel(KernelRidge()) is False
    assert _end_with_a_kernel(ColumnKernelizer([])) is False

    assert _end_with_a_kernel(Kernelizer()) is True

    pipe = make_pipeline(Kernelizer(), StandardScaler())
    assert _end_with_a_kernel(pipe) is False
    pipe = make_pipeline(make_pipeline(Kernelizer()), StandardScaler())
    assert _end_with_a_kernel(pipe) is False

    pipe = make_pipeline(StandardScaler(), Kernelizer())
    assert _end_with_a_kernel(pipe) is True
    pipe = make_pipeline(Kernelizer())
    assert _end_with_a_kernel(pipe) is True
    pipe = make_pipeline(Kernelizer())
    pipe = make_pipeline(StandardScaler(), make_pipeline(Kernelizer()))
    assert _end_with_a_kernel(pipe) is True


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_make_column_kernelizer(backend):
    backend = set_backend(backend)

    trans = Kernelizer(kernel="cosine")
    ck = make_column_kernelizer((trans, slice(0, 3)))

    assert isinstance(ck, ColumnKernelizer)
    assert len(ck.transformers) == 1
    assert len(ck.transformers[0]) == 3
    assert ck.transformers[0][0] == "kernelizer"
    assert ck.transformers[0][1] == trans
    assert ck.transformers[0][2] == slice(0, 3)

    trans = Kernelizer(kernel="cosine")
    ck = make_column_kernelizer((trans, slice(0, 3)), ("passthrough", [3, 4]))

    assert isinstance(ck, ColumnKernelizer)
    assert len(ck.transformers) == 2
    assert len(ck.transformers[0]) == 3
    assert len(ck.transformers[1]) == 3
    assert ck.transformers[0][0] == "kernelizer"
    assert ck.transformers[0][1] == trans
    assert ck.transformers[0][2] == slice(0, 3)
    assert ck.transformers[1][0] == "passthrough"
    assert ck.transformers[1][1] == "passthrough"
    assert ck.transformers[1][2] == [3, 4]


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_column_kernelizer_in_pipeline(backend):
    backend = set_backend(backend)

    X = backend.randn(10, 5)
    Y = backend.randn(10, 3)

    ck = make_column_kernelizer(
        (Kernelizer("linear"), slice(0, 4)),
        (Kernelizer("linear"), slice(4, 6)),
    )
    pipe = make_pipeline(
        ck,
        MultipleKernelRidgeCV(
            kernels="precomputed",
            solver_params=dict(n_iter=backend.ones_like(X, shape=(1, 2)),
                               progress_bar=False)),
    )
    pipe.fit(X, Y)


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_kernelizer_get_X_fit(backend):
    backend = set_backend(backend)
    X = backend.randn(10, 5)

    estimator = Kernelizer("linear")
    with pytest.raises(NotFittedError):
        estimator.get_X_fit()

    K = estimator.fit_transform(X)
    X = estimator.get_X_fit()
    assert len(K) == len(X)  # same number of samples


@pytest.mark.parametrize('estimator', [
    make_column_kernelizer((Kernelizer("linear"), slice(0, 4))),
    make_column_kernelizer(
        (Kernelizer("linear"), slice(0, 4)),
        ("drop", slice(4, 6)),
        remainder='passthrough',
    ),
    make_column_kernelizer(
        ("passthrough", slice(0, 2)),
        ("passthrough", slice(0, 4)),
        ("passthrough", slice(4, 5)),
    ),
])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_column_kernelizer_get_X_fit(estimator, backend):
    backend = set_backend(backend)
    X = backend.randn(10, 5)

    estimator = clone(estimator)  # pytest reuses the same object over backends
    with pytest.raises(NotFittedError):
        estimator.get_X_fit()

    Ks = estimator.fit_transform(X)
    Xs = estimator.get_X_fit()
    assert len(Ks) == len(Xs)  # same number of feature spaces
    for X, K in zip(Xs, Ks):
        assert X.shape[0] == K.shape[0]  # same number of samples


###############################################################################
# scikit-learn.utils.estimator_checks


class Kernelizer_(Kernelizer):
    """Cast transforms to numpy arrays, to be used in scikit-learn tests.

    Used for testing only.
    """

    def fit_transform(self, X, y=None):
        backend = get_backend()
        return backend.to_numpy(super().fit_transform(X, y))

    def transform(self, X):
        backend = get_backend()
        return backend.to_numpy(super().transform(X))


@sklearn.utils.estimator_checks.parametrize_with_checks([
    Kernelizer_(),
])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_check_estimator(estimator, check, backend):
    backend = set_backend(backend)
    check(estimator)
