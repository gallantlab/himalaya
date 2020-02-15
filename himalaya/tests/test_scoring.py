import pytest
from sklearn.metrics import r2_score as r2_score_sklearn
from sklearn.metrics import mean_squared_error

from himalaya.backend import change_backend
from himalaya.backend import ALL_BACKENDS

from himalaya.scoring import r2_score
from himalaya.scoring import l2_neg_loss
from himalaya.scoring import correlation_score
from himalaya.scoring import r2_score_split


def _create_data(backend, dtype_str):
    n_alphas, n_samples, n_targets = 10, 20, 30
    y_pred = backend.asarray(backend.randn(n_alphas, n_samples, n_targets),
                             dtype=getattr(backend, dtype_str))
    y_true = backend.asarray(backend.randn(n_samples, n_targets),
                             dtype=getattr(backend, dtype_str))
    y_true -= y_true.mean(0)
    return y_pred, y_true


@pytest.mark.parametrize('dtype_str', ["float32", "float64"])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_r2_score(backend, dtype_str):
    backend = change_backend(backend)
    y_pred, y_true = _create_data(backend, dtype_str)
    rtol = 1e-5 if dtype_str == "float32" else 1e-6

    # multi predictions
    s_1 = backend.stack([
        backend.asarray(r2_score_sklearn(y_true, pp, multioutput='raw_values'),
                        dtype=dtype_str) for pp in y_pred
    ])
    s_2 = r2_score(y_true, y_pred)
    backend.assert_allclose(s_1, s_2, rtol=rtol, atol=1e-6)

    # single prediction
    s_1 = backend.asarray(
        r2_score_sklearn(y_true, y_pred[0], multioutput='raw_values'),
        dtype=dtype_str)
    s_2 = r2_score(y_true, y_pred[0])
    backend.assert_allclose(s_1, s_2, rtol=rtol, atol=1e-6)


@pytest.mark.parametrize('dtype_str', ["float32", "float64"])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_r2_score_split(backend, dtype_str):
    # r2_score_split(include_correlation=False) is equivalent to r2_score
    backend = change_backend(backend)
    y_pred, y_true = _create_data(backend, dtype_str)
    rtol = 1e-5 if dtype_str == "float32" else 1e-6

    # multi predictions
    s_1 = backend.stack([
        backend.asarray(r2_score_sklearn(y_true, pp, multioutput='raw_values'),
                        dtype=dtype_str) for pp in y_pred
    ])
    s_2 = r2_score_split(y_true, y_pred, False)
    backend.assert_allclose(s_1, s_2, rtol=rtol, atol=1e-6)

    # single prediction
    s_1 = backend.asarray(
        r2_score_sklearn(y_true, y_pred[0], multioutput='raw_values'),
        dtype=dtype_str)
    s_2 = r2_score_split(y_true, y_pred[0], False)
    backend.assert_allclose(s_1, s_2, rtol=rtol, atol=1e-6)


@pytest.mark.parametrize('dtype_str', ["float32", "float64"])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_r2_score_split_include_correlation(backend, dtype_str):
    # r2_score_split(include_correlation=True) gives results that sum to the
    # r2 score of the sum.
    backend = change_backend(backend)
    y_pred, y_true = _create_data(backend, dtype_str)
    rtol = 1e-5 if dtype_str == "float32" else 1e-6

    s_1 = r2_score_split(y_true, y_pred.sum(0), True)
    s_2 = r2_score_split(y_true, y_pred, True)
    backend.assert_allclose(s_1, s_2.sum(0), rtol=rtol, atol=1e-6)


@pytest.mark.parametrize('dtype_str', ["float32", "float64"])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_l2_neg_loss(backend, dtype_str):
    backend = change_backend(backend)
    y_pred, y_true = _create_data(backend, dtype_str)
    rtol = 1e-5 if dtype_str == "float32" else 1e-6

    # multi predictions
    s_1 = -backend.stack([
        backend.asarray(
            mean_squared_error(y_true, pp, multioutput='raw_values'),
            dtype=dtype_str) for pp in y_pred
    ])
    s_2 = l2_neg_loss(y_true, y_pred)
    backend.assert_allclose(s_1, s_2, rtol=rtol, atol=1e-6)

    # single prediction
    s_1 = -backend.asarray(
        mean_squared_error(y_true, y_pred[0], multioutput='raw_values'),
        dtype=dtype_str)
    s_2 = l2_neg_loss(y_true, y_pred[0])
    backend.assert_allclose(s_1, s_2, rtol=rtol, atol=1e-6)


def _correlation(backend, y_true, y_pred, dtype):
    y_true = backend.asarray(y_true, dtype="float64")
    y_pred = backend.asarray(y_pred, dtype="float64")

    y_pred -= y_pred.mean(0)
    y_pred -= y_pred.mean(0)

    y_true /= backend.std_float64(y_true, 0)
    y_pred /= backend.std_float64(y_pred, 0)

    return backend.asarray((y_true * y_pred).mean(0), dtype=dtype)


@pytest.mark.parametrize('dtype_str', ["float32", "float64"])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_correlation_score(backend, dtype_str):
    backend = change_backend(backend)
    y_pred, y_true = _create_data(backend, dtype_str)
    rtol = 1e-5 if dtype_str == "float32" else 1e-6

    # multi predictions
    s_1 = backend.stack(
        [_correlation(backend, y_true, pp, dtype=dtype_str) for pp in y_pred])
    s_2 = correlation_score(y_true, y_pred)
    backend.assert_allclose(s_1, s_2, rtol=rtol, atol=1e-6)

    # single prediction
    s_1 = _correlation(backend, y_true, y_pred[0], dtype=dtype_str)
    s_2 = correlation_score(y_true, y_pred[0])
    backend.assert_allclose(s_1, s_2, rtol=rtol, atol=1e-6)


@pytest.mark.parametrize('dtype_str', ["float32", "float64"])
@pytest.mark.parametrize('scoring', [
    r2_score,
    l2_neg_loss,
    correlation_score,
    r2_score_split,
])
@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_infinite_and_nans(backend, dtype_str, scoring):
    backend = change_backend(backend)
    y_pred, y_true = _create_data(backend, dtype_str)

    # nan as zero
    y_pred[0][10, 10] = backend.nan
    with pytest.warns(UserWarning) as record:
        s_2 = scoring(y_true, y_pred[0])
    assert not backend.any(backend.isnan(s_2))
    assert len(record) == 1
    assert record[0].message.args[0] == "nan in y_pred."

    # inf as zero
    y_pred[0][10, 10] = backend.inf
    with pytest.warns(UserWarning) as record:
        s_2 = scoring(y_true, y_pred[0])
    assert not backend.any(backend.isnan(s_2))
    assert len(record) == 1
    assert record[0].message.args[0] == "inf in y_pred."
