import pytest

from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelCenterer
from himalaya.kernel_ridge import Kernelizer
from himalaya.kernel_ridge import ColumnKernelizer
from himalaya.kernel_ridge import make_column_kernelizer
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.kernel_ridge import MultipleKernelRidgeCV
from himalaya.ridge import RidgeCV
from himalaya.ridge import GroupRidgeCV
from himalaya.lasso import SparseGroupLassoCV

GPU_BACKENDS = [
    "cupy",
    "torch_cuda",
]


@pytest.mark.parametrize('backend', GPU_BACKENDS)
@pytest.mark.parametrize('force_cpu', [True, False])
def test_kernel_centerer(backend, force_cpu):
    backend = set_backend(backend)
    X = backend.randn(5, 5)
    K = X @ X.T

    Kc = KernelCenterer(force_cpu=force_cpu).fit_transform(K)
    assert backend.is_in_gpu(Kc) != force_cpu


@pytest.mark.parametrize('backend', GPU_BACKENDS)
@pytest.mark.parametrize('force_cpu', [True, False])
def test_kernelizer(backend, force_cpu):
    backend = set_backend(backend)
    X = backend.randn(10, 5)

    K = Kernelizer(force_cpu=force_cpu).fit_transform(X)
    assert backend.is_in_gpu(K) != force_cpu


@pytest.mark.parametrize('backend', GPU_BACKENDS)
@pytest.mark.parametrize('force_cpu', [True, False])
def test_column_kernelizer(backend, force_cpu):
    backend = set_backend(backend)
    X = backend.randn(10, 5)

    Ks = ColumnKernelizer([
        ("name", Kernelizer(), slice(0, 5)),
    ], force_cpu=force_cpu).fit_transform(X)

    assert backend.is_in_gpu(Ks) != force_cpu


@pytest.mark.parametrize('backend', GPU_BACKENDS)
@pytest.mark.parametrize('force_cpu', [True, False])
def test_make_column_kernelizer(backend, force_cpu):
    backend = set_backend(backend)
    X = backend.randn(10, 5)

    Ks = make_column_kernelizer((Kernelizer(), slice(0, 5)),
                                force_cpu=force_cpu).fit_transform(X)
    assert backend.is_in_gpu(Ks) != force_cpu


@pytest.mark.parametrize('backend', GPU_BACKENDS)
@pytest.mark.parametrize('force_cpu', [True, False])
def test_kernel_ridge_cv(backend, force_cpu):
    backend = set_backend(backend)
    X = backend.randn(10, 5)
    Y = backend.randn(10, 2)

    best_alphas_ = KernelRidgeCV(force_cpu=force_cpu).fit(X, Y).best_alphas_
    assert backend.is_in_gpu(best_alphas_) != force_cpu


@pytest.mark.parametrize('backend', GPU_BACKENDS)
@pytest.mark.parametrize('force_cpu', [True, False])
def test_multiple_kernel_ridge_cv(backend, force_cpu):
    backend = set_backend(backend)
    X = backend.randn(10, 5)
    Y = backend.randn(10, 2)

    deltas_ = MultipleKernelRidgeCV(
        kernels=["linear"], force_cpu=force_cpu,
        solver_params=dict(n_iter=2, progress_bar=False)).fit(X, Y).deltas_
    assert backend.is_in_gpu(deltas_) != force_cpu


@pytest.mark.parametrize('backend', GPU_BACKENDS)
@pytest.mark.parametrize('force_cpu', [True, False])
def test_ridge_cv(backend, force_cpu):
    backend = set_backend(backend)
    X = backend.randn(10, 5)
    Y = backend.randn(10, 2)

    best_alphas_ = RidgeCV(force_cpu=force_cpu).fit(X, Y).best_alphas_
    assert backend.is_in_gpu(best_alphas_) != force_cpu


@pytest.mark.parametrize('backend', GPU_BACKENDS)
@pytest.mark.parametrize('force_cpu', [True, False])
def test_group_ridge_cv(backend, force_cpu):
    backend = set_backend(backend)
    X = backend.randn(10, 5)
    Y = backend.randn(10, 2)

    deltas_ = GroupRidgeCV(groups=[0, 1, 0, 1, 1],
                           force_cpu=force_cpu, solver_params=dict(
                               n_iter=2, progress_bar=False)).fit(X, Y).deltas_
    assert backend.is_in_gpu(deltas_) != force_cpu


@pytest.mark.parametrize('backend', GPU_BACKENDS)
@pytest.mark.parametrize('force_cpu', [True, False])
def test_group_ridge_cv(backend, force_cpu):
    backend = set_backend(backend)
    X = backend.randn(10, 5)
    Y = backend.randn(10, 2)

    best_l21_reg_ = SparseGroupLassoCV(
        groups=[0, 1, 0, 1, 1], force_cpu=force_cpu,
        solver_params=dict(progress_bar=False)).fit(X, Y).best_l21_reg_
    assert backend.is_in_gpu(best_l21_reg_) != force_cpu
