import pytest

from himalaya.backend import set_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.utils import assert_array_almost_equal

from himalaya.kernel_ridge import primal_weights_weighted_kernel_ridge
from himalaya.kernel_ridge import predict_weighted_kernel_ridge


def _create_dataset(backend):
    n_samples, n_targets = 30, 3

    Xs = [
        backend.asarray(backend.randn(n_samples, n_features), backend.float64)
        for n_features in [100, 200]
    ]
    Ks = backend.stack([backend.matmul(X, X.T) for X in Xs])
    Y = backend.asarray(backend.randn(n_samples, n_targets), backend.float64)
    dual_weights = backend.asarray(backend.randn(n_samples, n_targets),
                                   backend.float64)
    exp_deltas = backend.asarray(backend.rand(Ks.shape[0], n_targets),
                                 backend.float64)
    deltas = backend.log(exp_deltas)

    return Xs, Ks, Y, deltas, dual_weights


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_predict_weighted_kernel_ridge(backend):
    backend = set_backend(backend)
    Xs, Ks, _, deltas, dual_weights = _create_dataset(backend)

    primal_weights = primal_weights_weighted_kernel_ridge(
        dual_weights, deltas, Xs)
    predictions_primal = backend.stack(
        [X @ backend.asarray(w) for X, w in zip(Xs, primal_weights)]).sum(0)

    predictions_dual = predict_weighted_kernel_ridge(Ks, dual_weights, deltas)

    assert_array_almost_equal(predictions_primal, predictions_dual)
