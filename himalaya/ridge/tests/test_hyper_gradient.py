import pytest

from himalaya.backend import change_backend
from himalaya.backend import ALL_BACKENDS

from himalaya.ridge._hyper_gradient import _compute_delta_gradient
from himalaya.ridge._hyper_gradient import _compute_delta_loss
from himalaya.ridge import solve_multiple_kernel_ridge_hyper_gradient
from himalaya.ridge import solve_multiple_kernel_ridge_random_search
from himalaya.ridge import solve_kernel_ridge_conjugate_gradient

from himalaya.ridge import generate_dirichlet_samples
from himalaya.ridge._utils import predict_and_score
from himalaya.scoring import r2_score
from himalaya.utils import assert_array_almost_equal


def _create_dataset(backend):
    n_featuress = [100, 200, 150]
    n_samples_train = 100
    n_samples_val = 50
    n_targets = 4

    Xs = [
        backend.asarray(backend.randn(n_samples_train, n_features),
                        backend.float64) for n_features in n_featuress
    ]
    Ks = backend.stack([X @ X.T for X in Xs])
    Xs_val = [
        backend.asarray(backend.randn(n_samples_val, n_features),
                        backend.float64) for n_features in n_featuress
    ]
    Ks_val = backend.stack([X_val @ X.T for X, X_val in zip(Xs, Xs_val)])

    true_gammas = backend.asarray(backend.rand(len(Xs), n_targets),
                                  backend.float64)

    ws = [
        backend.asarray(backend.randn(n_features, n_targets), backend.float64)
        for n_features in n_featuress
    ]
    Ys = backend.stack(
        [X @ w * backend.sqrt(g) for X, w, g in zip(Xs, ws, true_gammas)])
    Y = Ys.sum(0)
    Ys_val = backend.stack(
        [X @ w * backend.sqrt(g) for X, w, g in zip(Xs_val, ws, true_gammas)])
    Y_val = Ys_val.sum(0)

    gammas = backend.asarray(backend.rand(len(Xs), n_targets), backend.float64)

    dual_weights = backend.asarray(backend.randn(*Y.shape), backend.float64)

    return Ks, Y, dual_weights, gammas, Ks_val, Y_val


@pytest.mark.parametrize('n_targets_batch', [None, 3])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_delta_gradient_direct(backend, n_targets_batch):
    backend = change_backend(backend)

    Ks, Y, dual_weights, gammas, Ks_val, Y_val = _create_dataset(backend)
    alphas = backend.asarray_like(backend.logspace(-1, 1, Y.shape[1]), Ks)
    deltas = backend.log(gammas / alphas)
    epsilons = backend.asarray_like(backend.randn(*deltas.shape), Ks)
    epsilons /= backend.norm(epsilons, axis=1)[:, None]
    step = 0.0000001
    deltas2 = deltas + epsilons * step

    # check direct gradient with a finite difference
    gradients = _compute_delta_gradient(Ks, Y, deltas, dual_weights,
                                        hyper_gradient_method='direct')[0]
    # gradients2 = _compute_delta_gradient(Ks, Y, deltas2, dual_weights,
    #                                      hyper_gradient_method='direct')[0]
    scores = _compute_delta_loss(Ks, Y, deltas, dual_weights)
    scores2 = _compute_delta_loss(Ks, Y, deltas2, dual_weights)

    directional_derivatives = (scores2 - scores) / step
    gradient_direction_product = (gradients * epsilons[:, :]).sum(0)
    norm = backend.norm(gradient_direction_product)

    assert_array_almost_equal(gradient_direction_product / norm,
                              directional_derivatives / norm, decimal=5)


@pytest.mark.parametrize('n_targets_batch', [None, 3])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_delta_gradient_indirect(backend, n_targets_batch):
    backend = change_backend(backend)

    Ks, Y, dual_weights, gammas, Ks_val, Y_val = _create_dataset(backend)
    alphas = backend.asarray_like(backend.logspace(-1, 1, Y.shape[1]), Ks)
    deltas = backend.log(gammas / alphas)
    epsilons = backend.asarray_like(backend.randn(*deltas.shape), Ks)
    epsilons /= backend.norm(epsilons, axis=1)[:, None]
    step = 0.0000001
    deltas2 = deltas + epsilons * step

    # check direct and indirect gradient with a finite difference
    # to get the indirect gradient, we need to refit the kernel ridge during
    # the validation loss computation.

    def score_func(Y_val, Ypred):
        return 0.5 * backend.norm(Ypred - Y_val, axis=0) ** 2

    def compute_loss(deltas):
        gammas = backend.exp(deltas)
        dual_weights = solve_kernel_ridge_conjugate_gradient(
            Ks, Y, gammas, initial_dual_weights=None, alpha=1, max_iter=1000,
            tol=1e-5)
        loss = predict_and_score(Ks_val, dual_weights, gammas, Y_val,
                                 score_func=score_func)
        return loss, dual_weights

    loss, dual_weights = compute_loss(deltas)
    loss2, dual_weights2 = compute_loss(deltas2)

    gradients = _compute_delta_gradient(Ks_val, Y_val, deltas, dual_weights,
                                        Ks_train=Ks,
                                        hyper_gradient_method='conjugate',
                                        tol=1e-5)[0]

    directional_derivatives = (loss2 - loss) / step
    gradient_direction_product = (gradients * epsilons[:, :]).sum(0)
    norm = backend.norm(gradient_direction_product)

    assert_array_almost_equal(gradient_direction_product / norm,
                              directional_derivatives / norm, decimal=4)


@pytest.mark.parametrize('n_targets_batch', [None, 3])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_multiple_kernel_ridge_hyper_gradient_n_targets_batch(
        backend, n_targets_batch):
    _test_solve_multiple_kernel_ridge_hyper_gradient(
        backend=backend, n_targets_batch=n_targets_batch)


@pytest.mark.parametrize('method', ["direct", "conjugate", "neumann"])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_multiple_kernel_ridge_hyper_gradient_method(backend, method):
    _test_solve_multiple_kernel_ridge_hyper_gradient(backend=backend,
                                                     method=method)


@pytest.mark.parametrize('initial_deltas', [0, 5, 'dirichlet', 'ridgecv'])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_multiple_kernel_ridge_hyper_gradient_initial_deltas(
        backend, initial_deltas):
    _test_solve_multiple_kernel_ridge_hyper_gradient(
        backend=backend, initial_deltas=initial_deltas)


@pytest.mark.parametrize('kernel_ridge', ["conjugate", "gradient"])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_solve_multiple_kernel_ridge_hyper_gradient_kernel_ridge(
        backend, kernel_ridge):
    _test_solve_multiple_kernel_ridge_hyper_gradient(backend=backend,
                                                     kernel_ridge=kernel_ridge)


def _test_solve_multiple_kernel_ridge_hyper_gradient(backend,
                                                     n_targets_batch=None,
                                                     method="direct",
                                                     initial_deltas=0,
                                                     kernel_ridge="conjugate"):
    backend = change_backend(backend)
    Ks, Y, dual_weights, gammas, Ks_val, Y_val = _create_dataset(backend)
    cv = 3
    progress_bar = False

    # compare bilinear gradient descent and dirichlet sampling
    all_scores_mean, dual_weights, deltas = \
        solve_multiple_kernel_ridge_hyper_gradient(
            Ks, Y, max_iter=100, n_targets_batch=n_targets_batch,
            max_iter_inner_dual=1, max_iter_inner_hyper=1, tol=None,
            score_func=r2_score, cv_splitter=cv,
            hyper_gradient_method=method, initial_deltas=initial_deltas,
            kernel_ridge_method=kernel_ridge, progress_bar=progress_bar)
    scores_1 = all_scores_mean[all_scores_mean.sum(axis=1) != 0][-1]

    alphas = backend.logspace(-5, 5, 11)
    gammas = generate_dirichlet_samples(50, len(Ks), concentrations=[.1, 1.],
                                        random_state=0)
    all_scores_mean, best_gammas, best_alphas, refit_weights = \
        solve_multiple_kernel_ridge_random_search(
            Ks, Y, gammas, alphas, n_targets_batch=n_targets_batch,
            score_func=r2_score, cv_splitter=cv, progress_bar=progress_bar)
    scores_2 = backend.max(all_scores_mean, axis=0)

    assert_array_almost_equal(scores_1, scores_2, decimal=1)
