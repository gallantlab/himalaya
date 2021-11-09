"""Adapt functions from scikit-learn to use different backends."""

import itertools
from functools import partial
import math

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..backend import get_backend
from ..backend import force_cpu_backend
from ..validation import _get_string_dtype
from ..validation import check_array
from ..validation import issparse


def check_pairwise_arrays(X, Y, precomputed=False, dtype=None,
                          accept_sparse='csr', force_all_finite=True):
    """ Set X and Y appropriately and checks inputs

    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats (or dtype if provided). Finally, the function
    checks that the size of the second dimension of the two arrays is equal, or
    the equivalent check for a precomputed distance matrix.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_X, n_features)

    Y : {array-like, sparse matrix}, shape (n_samples_Y, n_features)

    precomputed : bool
        True if X is to be treated as precomputed distances to the samples in
        Y.

    dtype : string, type, list of types or None (default=None)
        Data type required for X and Y. If None, the dtype will be an
        appropriate float type selected by _return_float_dtype.

    accept_sparse : string, boolean or list/tuple of strings
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accept both np.inf and np.nan in array.
        - 'allow-nan': accept only np.nan values in array. Values cannot
          be infinite.

    Returns
    -------
    X : {array-like, sparse matrix}, shape (n_samples_X, n_features)
        An array equal to X.

    Y : {array-like, sparse matrix}, shape (n_samples_Y, n_features)
        An array equal to Y if Y was not None.
        If Y was None, Y will be a pointer to X.

    """
    if dtype is None:
        dtype = _return_float_dtype(X, Y)

    if Y is X or Y is None:
        X = Y = check_array(X, accept_sparse=accept_sparse, dtype=dtype,
                            force_all_finite=force_all_finite)
    else:
        X = check_array(X, accept_sparse=accept_sparse, dtype=dtype,
                        force_all_finite=force_all_finite)
        Y = check_array(Y, accept_sparse=accept_sparse, dtype=dtype,
                        force_all_finite=force_all_finite)

    if precomputed:
        pass
    elif X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" %
                         (X.shape[1], Y.shape[1]))

    return X, Y


def _return_float_dtype(X, Y):
    """
    1. If dtype of X and Y is float32, then dtype float32 is returned.
    2. Else dtype float64 is returned.
    """
    X_dtype = _get_string_dtype(X)

    if Y is None:
        Y_dtype = X_dtype
    else:
        Y_dtype = _get_string_dtype(Y)

    if X_dtype == Y_dtype == "float32":
        dtype = "float32"
    else:
        dtype = "float64"

    return dtype


def _row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.

    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.

    Performs no input validation.

    Parameters
    ----------
    X : array_like
        The input array
    squared : bool, optional (default = False)
        If True, return squared norms.

    Returns
    -------
    array_like
        The row-wise (squared) Euclidean norm of X.
    """
    backend = get_backend()

    if issparse(X):
        import scipy.sparse
        if not isinstance(X, scipy.sparse.csr_matrix):
            X = scipy.sparse.csr_matrix(X)
        from sklearn.utils.sparsefuncs_fast import csr_row_norms
        norms = csr_row_norms(X)
    else:
        norms = backend.einsum('ij,ij->i', X, X)

    if not squared:
        backend.sqrt(norms, out=norms)
    return norms


def _normalize(X):
    """L2 normalizer with copy."""
    if issparse(X):
        X = X.copy()
        from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l2
        inplace_csr_row_normalize_l2(X)
    else:
        norms = _row_norms(X)
        norms[norms == 0.0] = 1.0
        X = X / norms[:, None]
    return X


###############################################################################


def linear_kernel(X, Y=None):
    """
    Compute the linear kernel between X and Y.

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
        Train or test features.
    Y : array of shape (n_samples_Y, n_features)
        Train features.

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)
        Computed kernel.
    """
    backend = get_backend()
    X, Y = check_pairwise_arrays(X, Y)

    K = X @ Y.T
    if issparse(K):
        K = K.toarray()
    K = backend.asarray(K)
    return K


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    """
    Compute the polynomial kernel between X and Y::

        K(X, Y) = (gamma <X, Y> + coef0)^degree

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
        Train or test features.
    Y : array of shape (n_samples_Y, n_features)
        Train features.
    degree : int, default 3
        Degree of the polynomial.
    gamma : float, default None
        if None, defaults to 1.0 / n_features
    coef0 : float, default 1
        Intercept.

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)
        Computed kernel.
    """
    backend = get_backend()
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = X @ Y.T
    if issparse(K):
        K = K.toarray()
    K = backend.asarray(K)
    K *= gamma
    K += coef0
    K = backend.power(K, degree, out=K)
    return K


def sigmoid_kernel(X, Y=None, gamma=None, coef0=1):
    """
    Compute the sigmoid kernel between X and Y::

        K(X, Y) = tanh(gamma <X, Y> + coef0)

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
        Train or test features.
    Y : array of shape (n_samples_Y, n_features)
        Train features.
    gamma : float, default None
        If None, defaults to 1.0 / n_features
    coef0 : float, default 1
        Intercept.

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)
        Computed kernel.
    """
    backend = get_backend()
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = X @ Y.T
    if issparse(K):
        K = K.toarray()
    K = backend.asarray(K)
    K *= gamma
    K += coef0
    K = backend.tanh(K, out=K)
    return K


def rbf_kernel(X, Y=None, gamma=None):
    """
    Compute the rbf (gaussian) kernel between X and Y::

        K(x, y) = exp(-gamma ||x-y||^2)

    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
        Train or test features.
    Y : array of shape (n_samples_Y, n_features)
        Train features.
    gamma : float, default None
        If None, defaults to 1.0 / n_features

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)
        Computed kernel.
    """
    backend = get_backend()
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = euclidean_distances(X, Y, squared=True)
    K = backend.asarray(K)
    K *= -gamma
    K = backend.exp(K, out=K)
    return K


def cosine_similarity_kernel(X, Y=None):
    """Compute cosine similarity between samples in X and Y.

    Cosine similarity, or the cosine kernel, computes similarity as the
    normalized dot product of X and Y:

        K(X, Y) = <X, Y> / (||X||*||Y||)

    On L2-normalized data, this function is equivalent to linear_kernel.

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
        Train or test features.
    Y : array of shape (n_samples_Y, n_features)
        Train features.

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)
        Computed kernel.
    """
    backend = get_backend()
    X, Y = check_pairwise_arrays(X, Y)

    X_normalized = _normalize(X)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = _normalize(Y)

    K = X_normalized @ Y_normalized.T

    if issparse(K):
        K = K.toarray()
    K = backend.asarray(K)

    return K


PAIRWISE_KERNEL_FUNCTIONS = {
    'linear': linear_kernel,
    'polynomial': polynomial_kernel,
    'poly': polynomial_kernel,
    'rbf': rbf_kernel,
    'sigmoid': sigmoid_kernel,
    'cosine': cosine_similarity_kernel,
}

###############################################################################


def euclidean_distances(X, Y=None, squared=False):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
        Train or test features.
    Y : array of shape (n_samples_Y, n_features)
        Train features.
    squared : boolean, optional
        Whether to return squared distances.

    Returns
    -------
    distances : array, shape (n_samples_X, n_samples_Y)
        Computed distances.
    """
    backend = get_backend()
    X, Y = check_pairwise_arrays(X, Y)

    distances = _euclidean_distances_upcast(X, Y)
    if not squared:
        distances = backend.sqrt(distances, out=distances)

    return distances


def _euclidean_distances_upcast(X, Y, batch_size=None):
    """Euclidean distances between X and Y.

    X and Y are upcast to float64 by chunks, which size is chosen to limit
    memory increase by approximately 10% (at least 10MiB).
    """
    backend = get_backend()

    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    n_features = X.shape[1]

    if issparse(X):
        distances = backend.zeros(shape=(n_samples_X, n_samples_Y),
                                  dtype=_get_string_dtype(X))
    else:
        distances = backend.zeros_like(X, shape=(n_samples_X, n_samples_Y))

    if batch_size is None:
        X_size = backend.prod(backend.asarray(X.shape))
        Y_size = backend.prod(backend.asarray(Y.shape))
        x_density = X.nnz / X_size if issparse(X) else 1
        y_density = Y.nnz / Y_size if issparse(Y) else 1

        # Allow 10% more memory than X, Y and the distance matrix take (at
        # least 10MiB)
        maxmem = max(
            ((x_density * n_samples_X + y_density * n_samples_Y) * n_features +
             (x_density * n_samples_X * y_density * n_samples_Y)) / 10.,
            10 * 2 ** 17)

        # The increase amount of memory in 8-byte blocks is:
        # - x_density * batch_size * n_features (copy of chunk of X)
        # - y_density * batch_size * n_features (copy of chunk of Y)
        # - batch_size * batch_size (chunk of distance matrix)
        # Hence x² + (xd+yd)kx = M, where x=batch_size, k=n_features, M=maxmem
        #                                 xd=x_density and yd=y_density
        tmp = (x_density + y_density) * n_features
        batch_size = (-tmp + math.sqrt(tmp ** 2 + 4 * maxmem)) / 2
        batch_size = max(int(batch_size), 1)

    for x_start in range(0, n_samples_X, batch_size):
        x_batch = slice(x_start, x_start + batch_size)

        if issparse(X):
            X_chunk = X[x_batch].astype("float64", copy=False)
        else:
            X_chunk = backend.asarray(X[x_batch], dtype=backend.float64)
        XX_chunk = _row_norms(X_chunk, squared=True)[:, None]

        for y_start in range(0, n_samples_Y, batch_size):
            y_batch = slice(y_start, y_start + batch_size)

            if X is Y and y_batch < x_batch:
                # when X is Y the distance matrix is symmetric so we only need
                # to compute half of it.
                d = distances[y_batch, x_batch].T
            else:
                if issparse(Y):
                    Y_chunk = Y[y_batch].astype("float64", copy=False)
                else:
                    Y_chunk = backend.asarray(Y[y_batch],
                                              dtype=backend.float64)
                YY_chunk = _row_norms(Y_chunk, squared=True)[None, :]

                d = -2 * X_chunk @ Y_chunk.T
                if issparse(d):
                    d = d.toarray()
                d += XX_chunk
                d += YY_chunk

            if issparse(X):
                distances[x_batch, y_batch] = backend.asarray(d)
            else:
                distances[x_batch, y_batch] = backend.asarray_like(d, X)

    return distances


def _pairwise_callable(X, Y, metric, force_all_finite=True, **params):
    """Handle the callable case for pairwise_{distances,kernels}.
    """
    backend = get_backend()
    X, Y = check_pairwise_arrays(X, Y, force_all_finite=force_all_finite)
    out = backend.zeros_like(X, shape=(X.shape[0], Y.shape[0]), dtype='float')

    if X is Y:
        # Only calculate metric for upper triangle
        iterator = itertools.combinations(range(X.shape[0]), 2)
        for i, j in iterator:
            out[i, j] = metric(X[i], Y[j], **params)

        # Make symmetric
        out = out + out.T

        # Calculate diagonal
        # NB: nonzero diagonals are allowed for both metrics and kernels
        for i in range(X.shape[0]):
            x = X[i]
            out[i, i] = metric(x, x, **params)

    else:
        # Calculate all cells
        iterator = itertools.product(range(X.shape[0]), range(Y.shape[0]))
        for i, j in iterator:
            out[i, j] = metric(X[i], Y[j], **params)

    return out


def pairwise_kernels(X, Y=None, metric="linear", n_jobs=None, **params):
    """Compute the kernel between arrays X and optional array Y.

    This method takes either a vector array or a kernel matrix, and returns
    a kernel matrix. If the input is a vector array, the kernels are
    computed. If the input is a kernel matrix, it is returned instead.

    This method provides a safe way to take a kernel matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    If Y is given (default is None), then the returned matrix is the pairwise
    kernel between the arrays from both X and Y.

    Valid values for metric are:
        ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf',
        'laplacian', 'sigmoid', 'cosine']

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array of shape (n_samples_X, n_samples_X) if metric == "precomputed", \
             (n_samples_X, n_features) otherwise
        Array of pairwise kernels between samples, or a feature array.

    Y : array of shape (n_samples_Y, n_features)
        A second feature array only if X metric != "precomputed".

    metric : string, or callable
        The metric to use when calculating kernel between instances in a
        feature array. If metric is a string, it must be one of the metrics
        in PAIRWISE_KERNEL_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two rows from X as input and return the corresponding
        kernel value as a single number. This means that callables from
        :mod:`sklearn.metrics.pairwise` are not allowed, as they operate on
        matrices, not single samples. Use the string identifying the kernel
        instead.

    **params : optional keyword parameters
        Any further parameters are passed directly to the kernel function.

    Returns
    -------
    K : array [n_samples_X, n_samples_X] or [n_samples_X, n_samples_Y]
        A kernel matrix K such that K_{i, j} is the kernel between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then K_{i, j} is the kernel between the ith array
        from X and the jth array from Y.

    Notes
    -----
    If metric is 'precomputed', Y is ignored and X is returned.
    """
    if metric == "precomputed":
        X, _ = check_pairwise_arrays(X, Y, precomputed=True)
        return X

    elif metric in PAIRWISE_KERNEL_FUNCTIONS:
        func = PAIRWISE_KERNEL_FUNCTIONS[metric]
    elif callable(metric):
        func = partial(_pairwise_callable, metric=metric, **params)
    else:
        raise ValueError("Unknown metric=%r." % metric)

    return func(X, Y, **params)


class KernelCenterer(TransformerMixin, BaseEstimator):
    """Center a kernel matrix.

    Adapt sklearn.preprocessing.KernelCenterer to use other backends.

    Let K(x, z) be a kernel defined by phi(x)^T phi(z), where phi is a
    function mapping x to a Hilbert space. KernelCenterer centers (i.e.,
    normalize to have zero mean) the data without explicitly computing phi(x).
    It is equivalent to centering phi(x) with
    sklearn.preprocessing.StandardScaler(with_std=False).

    Parameters
    ----------
    force_cpu : bool
        If True, computations will be performed on CPU, ignoring the
        current backend. If False, use the current backend.

    Attributes
    ----------
    K_fit_rows_ : array of shape (n_samples,)
        Average of each column of kernel matrix.

    K_fit_all_ : float
        Average of kernel matrix.

    Examples
    --------
    >>> from himalaya.kernel_ridge import KernelCenterer
    >>> from himalaya.kernel_ridge import pairwise_kernels
    >>> X = [[ 1., -2.,  2.],
    ...      [ -2.,  1.,  3.],
    ...      [ 4.,  1., -2.]]
    >>> K = pairwise_kernels(X, metric='linear')
    >>> K
    array([[  9.,   2.,  -2.],
           [  2.,  14., -13.],
           [ -2., -13.,  21.]])
    >>> transformer = KernelCenterer().fit(K)
    >>> transformer
    KernelCenterer()
    >>> transformer.transform(K)
    array([[  5.,   0.,  -5.],
           [  0.,  14., -14.],
           [ -5., -14.,  19.]])
    """

    def __init__(self, force_cpu=False):
        self.force_cpu = force_cpu

    @force_cpu_backend
    def fit(self, K, y=None):
        """Fit KernelCenterer

        Parameters
        ----------
        K : ndarray of shape (n_samples, n_samples)
            Kernel matrix.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        backend = get_backend()
        K = check_array(K, ndim=2)

        if K.shape[0] != K.shape[1]:
            raise ValueError("Kernel matrix must be a square matrix."
                             " Input is a {}x{} matrix.".format(
                                 K.shape[0], K.shape[1]))

        self.K_fit_rows_ = backend.mean_float64(K, axis=0)
        self.K_fit_all_ = backend.mean_float64(self.K_fit_rows_, axis=0)
        return self

    @force_cpu_backend
    def transform(self, K, copy=True):
        """Center kernel matrix.

        Parameters
        ----------
        K : ndarray of shape (n_samples1, n_samples2)
            Kernel matrix.

        copy : bool, default=True
            Set to False to perform inplace computation.

        Returns
        -------
        K_new : ndarray of shape (n_samples1, n_samples2)
        """
        check_is_fitted(self)
        backend = get_backend()
        K = check_array(K, ndim=2, copy=copy)

        K_pred_cols = backend.mean_float64(K, axis=1, keepdims=True)

        K -= self.K_fit_rows_
        K -= K_pred_cols
        K += self.K_fit_all_

        return K

    def _more_tags(self):
        return {'pairwise': True}
