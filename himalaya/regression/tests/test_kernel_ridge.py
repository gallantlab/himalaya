import pytest

from himalaya.backend import change_backend
from himalaya.backend import ALL_BACKENDS

from himalaya.regression.kernel_ridge import multi_kernel_ridge_gradient


@pytest.mark.parametrize("double_K", [False, True])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_multi_kernel_ridge_gradient(backend, double_K):
    backend = change_backend(backend)

    n_samples, n_targets = 20, 3
    alpha = 1.

    Xs = [backend.randn(n_samples, 10), backend.randn(n_samples, 5)]
    Ks = backend.stack([backend.matmul(X, X.T) for X in Xs])
    Ys = backend.randn(n_samples, n_targets)
    dual_weights = backend.randn(n_samples, n_targets)
    gammas = backend.rand(2, n_targets)

    grad = backend.zeros((n_samples, n_targets))
    func = backend.zeros(n_targets)
    for tt in range(n_targets):
        K = backend.sum(
            backend.stack([K * g for K, g in zip(Ks, gammas[:, tt])]), 0)
        grad[:, tt] = (backend.matmul(K, dual_weights[:, tt]) - Ys[:, tt] +
                       alpha * dual_weights[:, tt])

        pred = backend.matmul(K, dual_weights[:, tt])
        func[tt] = backend.sum((pred - Ys[:, tt]) ** 2, 0)
        func[tt] += alpha * (dual_weights[:, tt] @ K @ dual_weights[:, tt])

        if double_K:
            grad[:, tt] = backend.matmul(K.T, grad[:, tt])

    ########################
    grad2, func2 = multi_kernel_ridge_gradient(Ks, Ys, dual_weights, gammas,
                                               alpha, double_K=double_K,
                                               return_objective=True)
    backend.assert_allclose(grad, grad2)
    backend.assert_allclose(func, func2)
