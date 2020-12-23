from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector  # noqa
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import make_pipeline, _name_estimators

from ..backend import get_backend
from ..validation import check_array
from ..validation import _get_string_dtype

from ._kernels import pairwise_kernels
from ._kernels import PAIRWISE_KERNEL_FUNCTIONS


class Kernelizer(TransformerMixin, BaseEstimator):
    """Transform tabular data into a kernel.

    Parameters
    ----------
    kernel : str or callable, default="linear"
        Kernel mapping. Available kernels are: 'linear',
        'polynomial, 'poly', 'rbf', 'sigmoid', 'cosine', or 'precomputed'.
        Set to 'precomputed' in order to pass a precomputed kernel matrix to
        the estimator methods instead of samples.
        A callable should accept two arguments and the keyword arguments passed
        to this object as kernel_params, and should return a floating point
        number.

    kernel_params : dict or None
        Additional parameters for the kernel function.
        See more details in the docstring of the function:
        Kernelizer.ALL_KERNELS[kernel]

    Attributes
    ----------
    X_fit_ : array of shape (n_samples, n_features)
        Training data. If kernel == "precomputed" this is None.

    n_features_in_ : int
        Number of features (or number of samples if kernel == "precomputed")
        used during the fit.

    dtype_ : str
        Dtype of input data.

    Examples
    --------
    >>> from himalaya.kernel_ridge import Kernelizer
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 10, 5, 3
    >>> X = np.random.randn(n_samples, n_features)
    >>> model = Kernelizer()
    >>> model.fit_transform(X).shape
    (10, 10)
    """

    ALL_KERNELS = PAIRWISE_KERNEL_FUNCTIONS
    kernelizer = True

    def __init__(self, kernel="linear", kernel_params=None):
        self.kernel = kernel
        self.kernel_params = kernel_params

    def fit_transform(self, X, y=None):
        """Compute the kernel on the training set.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data. If kernel == "precomputed" this is instead
            a precomputed kernel array of shape (n_samples, n_samples).

        y : array of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        K : array of shape (n_samples, n_samples)
            Kernel of the input data.
        """
        accept_sparse = False if self.kernel == "precomputed" else ("csr",
                                                                    "csc")
        X = check_array(X, accept_sparse=accept_sparse, ndim=2)

        self.X_fit_ = _to_cpu(X) if self.kernel != "precomputed" else None
        self.dtype_ = _get_string_dtype(X)
        self.n_features_in_ = X.shape[1]

        K = self._get_kernel(X)
        return K

    def fit(self, X, y=None):
        """Compute the kernel on the training set.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data. If kernel == "precomputed" this is instead
            a precomputed kernel array of shape (n_samples, n_samples).

        y : array of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        """Compute the kernel on any data set.

        Parameters
        ----------
        X : array of shape (n_samples_transform, n_features)
            Training data. If kernel == "precomputed" this is instead
            a precomputed kernel array of shape
            (n_samples_transform, n_samples_fit).

        Returns
        -------
        K : array of shape (n_samples_transform, n_samples_fit)
            Kernel of the input data.
        """
        check_is_fitted(self)
        accept_sparse = False if self.kernel == "precomputed" else ("csr",
                                                                    "csc")
        X = check_array(X, dtype=self.dtype_, accept_sparse=accept_sparse,
                        ndim=2)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                'Different number of features in X than during fit.')
        K = self._get_kernel(X, self.X_fit_)
        return K

    def _get_kernel(self, X, Y=None):
        """Helper function to get the kernel."""
        backend = get_backend()
        kernel_params = self.kernel_params or {}
        kernel = pairwise_kernels(X, Y, metric=self.kernel, **kernel_params)
        return backend.asarray(kernel)

    def get_X_fit(self):
        """Helper to get the input data X seen during the fit.

        Returns
        -------
        X : array of shape (n_samples, n_features)
            Input array for the kernelizer.
        """
        check_is_fitted(self)
        return self.X_fit_

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"


class ColumnKernelizer(ColumnTransformer):
    """Applies transformers to columns of an array, ending with kernelizers.

    This estimator allows different columns or column subsets of the input
    to be transformed separately. Each transformer pipeline either ends with a
    kernelizer, or a linear kernelizer is added at the end. The different
    kernels generated are then stacked together to be used e.g. in a
    MultipleKernelRidgeCV(kernels="precomputed"). This is useful to perform
    separate transformations and kernels on different feature spaces.

    Warning : This class does not perfectly abide by scikit-learn's API.
    Indeed, it returns stacked kernels of shape (n_kernels, n_samples,
    n_samples), while scikit-learn's API only allows arrays of shape
    (n_samples, n_samples) or (n_samples, n_features). This class is intended
    to be used in a scikit-learn pipeline *just* before a
    MultipleKernelRidgeCV(kernels="precomputed").

    Parameters
    ----------
    transformers : list of tuples
        List of (name, transformer, columns) tuples specifying the
        transformer objects to be applied to subsets of the data.

        name : str
            Like in Pipeline and FeatureUnion, this allows the transformer and
            its parameters to be set using ``set_params`` and searched in grid
            search.
        transformer : {'drop', 'passthrough'} or estimator
            Estimator must support ``fit`` and ``transform``.
            Special-cased strings 'drop' and 'passthrough' are accepted as
            well, to indicate to drop the columns or to pass them through
            untransformed, respectively. If the transformer does not return a
            kernel (as informed by the attribute kernelizer=True), a linear
            kernelizer is applied after the transformer.
        columns :  str, array-like of str, int, array-like of int, \
                array-like of bool, slice or callable
            Indexes the data on its second axis. Integers are interpreted as
            positional columns, while strings can reference DataFrame columns
            by name.  A scalar string or int should be used where
            ``transformer`` expects X to be a 1d array-like (vector),
            otherwise a 2d array will be passed to the transformer.
            A callable is passed the input data `X` and can return any of the
            above. To select multiple columns by name or dtype, you can use
            :obj:`make_column_selector`.

    remainder : {'drop', 'passthrough'} or estimator, default='drop'
        By default, only the specified columns in `transformers` are
        transformed and combined in the output, and the non-specified
        columns are dropped. (default of ``'drop'``).
        By specifying ``remainder='passthrough'``, all remaining columns that
        were not specified in `transformers` will be automatically passed
        through. This subset of columns is concatenated with the output of
        the transformers.
        By setting ``remainder`` to be an estimator, the remaining
        non-specified columns will use the ``remainder`` estimator. The
        estimator must support ``fit`` and ``transform``.
        Note that using this feature requires that the DataFrame columns
        input at ``fit`` and ``transform`` have identical order.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        n_jobs does not work with GPU backends.

    transformer_weights : dict, default=None
        Multiplicative weights for features per transformer. The output of the
        transformer is multiplied by these weights. Keys are transformer names,
        values the weights.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    Attributes
    ----------
    transformers_ : list
        The collection of fitted transformers as tuples of
        (name, fitted_transformer, column). `fitted_transformer` can be an
        estimator, 'drop', or 'passthrough'. In case there were no columns
        selected, this will be the unfitted transformer.
        If there are remaining columns, the final element is a tuple of the
        form:
        ('remainder', transformer, remaining_columns) corresponding to the
        ``remainder`` parameter. If there are remaining columns, then
        ``len(transformers_)==len(transformers)+1``, otherwise
        ``len(transformers_)==len(transformers)``.

    named_transformers_ : :class:`~sklearn.utils.Bunch`
        Read-only attribute to access any transformer by given name.
        Keys are transformer names and values are the fitted transformer
        objects.

    n_features_in_ : int
        Number of features (or number of samples if kernel == "precomputed")
        used during the fit.

    sparse_output_ : False

    Notes
    -----
    The order of the columns in the transformed feature matrix follows the
    order of how the columns are specified in the `transformers` list.
    Columns of the original feature matrix that are not specified are
    dropped from the resulting transformed feature matrix, unless specified
    in the `passthrough` keyword. Those columns specified with `passthrough`
    are added at the right to the output of the transformers.

    See also
    --------
    himalaya.kernel_ridge.make_column_kernelizer : convenience function for
        combining the outputs of multiple kernelizer objects applied to
        column subsets of the original feature space.
    sklearn.compose.make_column_selector : convenience function for selecting
        columns based on datatype or the columns name with a regex pattern.

    Examples
    --------
    >>> import numpy as np
    >>> from himalaya.kernel_ridge import ColumnKernelizer
    >>> from himalaya.kernel_ridge import Kernelizer
    >>> ck = ColumnKernelizer(
    ...     [("kernel_1", Kernelizer(kernel="linear"), [0, 1, 2]),
    ...      ("kernel_2", Kernelizer(kernel="polynomial"), slice(3, 5))])
    >>> X = np.array([[0., 1., 2., 2., 3.],
                      [0., 2., 0., 0., 3.],
                      [0., 0., 1., 0., 3.],
    ...               [1., 1., 0., 1., 2.]])
    >>> # Kernelize separately the first three columns and the last two
    >>> # columns, creating two kernels of shape (n_samples, n_samples).
    >>> ck.fit_transform(X).shape
    (2, 4, 4)
    """
    # This is not a kernelizer, since it returns multiple kernels
    kernelizer = False

    def __init__(self, transformers, remainder='drop', n_jobs=None,
                 transformer_weights=None, verbose=False):
        self.transformers = transformers
        self.remainder = remainder
        self.sparse_threshold = 0
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose

    def _iter(self, fitted=False, replace_strings=False):
        """
        Generate (name, trans, column, weight) tuples.

        Add a default (linear) Kernelizer to any transformer that does not end
        with a Kernelizer.
        """
        for name, trans, column, weight in super()._iter(
                fitted=fitted, replace_strings=replace_strings):

            if not fitted:
                if trans == 'drop':
                    pass
                elif trans == 'passthrough':
                    trans = Kernelizer()
                elif not _end_with_a_kernel(trans):
                    trans = make_pipeline(trans, Kernelizer())

            yield (name, trans, column, weight)

    def _hstack(self, Xs):
        """Stack the kernels.

        In ColumnTransformer, this methods stacks Xs horizontally.
        Here instead, we stack all kernels in a new dimension.

        Parameters
        ----------
        Ks : array of shape (n_kernels, n_samples, n_samples)
        """
        backend = get_backend()
        return backend.stack(Xs)

    def get_X_fit(self):
        """Helper to get the input data X seen during the fit.

        Returns
        -------
        Xs : list of arrays of shape (n_samples, n_features_i)
            Input arrays for each kernelizer.
        """
        check_is_fitted(self)

        Xs = []
        for (_, trans, _, _) in self._iter(fitted=True, replace_strings=True):
            if hasattr(trans, "get_X_fit"):
                X = trans.get_X_fit()
            else:
                X = trans[-1].get_X_fit()

            Xs.append(X)
        return Xs


def make_column_kernelizer(*transformers, **kwargs):
    """Construct a ColumnKernelizer from the given transformers.

    This is a shorthand for the ColumnKernelizer constructor; it does not
    require, and does not permit, naming the transformers. Instead, they will
    be given names automatically based on their types. It also does not allow
    weighting with ``transformer_weights``.

    Parameters
    ----------
    *transformers : tuples
        Tuples of the form (transformer, columns) specifying the
        transformer objects to be applied to subsets of the data.

        transformer : {'drop', 'passthrough'} or estimator
            Estimator must support ``fit`` and ``transform``.
            Special-cased strings 'drop' and 'passthrough' are accepted as
            well, to indicate to drop the columns or to pass them through
            untransformed, respectively. If the transformer does not return a
            kernel (as informed by the attribute kernelizer=True), a linear
            kernelizer is applied after the transformer.
        columns : str,  array-like of str, int, array-like of int, slice, \
                array-like of bool or callable
            Indexes the data on its second axis. Integers are interpreted as
            positional columns, while strings can reference DataFrame columns
            by name. A scalar string or int should be used where
            ``transformer`` expects X to be a 1d array-like (vector),
            otherwise a 2d array will be passed to the transformer.
            A callable is passed the input data `X` and can return any of the
            above. To select multiple columns by name or dtype, you can use
            :obj:`make_column_selector`.

    remainder : {'drop', 'passthrough'} or estimator, default='drop'
        By default, only the specified columns in `transformers` are
        transformed and combined in the output, and the non-specified
        columns are dropped. (default of ``'drop'``).
        By specifying ``remainder='passthrough'``, all remaining columns that
        were not specified in `transformers` will be automatically passed
        through. This subset of columns is concatenated with the output of
        the transformers.
        By setting ``remainder`` to be an estimator, the remaining
        non-specified columns will use the ``remainder`` estimator. The
        estimator must support ``fit`` and ``transform``.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        n_jobs does not work with with GPU backends.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    Returns
    -------
    column_kernelizer : ColumnKernelizer

    See also
    --------
    himalaya.kernel_ridge.ColumnKernelizer : Class that allows combining the
        outputs of multiple transformer objects used on column subsets
        of the data into a single feature space.

    Examples
    --------
    >>> import numpy as np
    >>> from himalaya.kernel_ridge import make_column_kernelizer
    >>> from himalaya.kernel_ridge import Kernelizer
    >>> ck = make_column_kernelizer(
    ...     (Kernelizer(kernel="linear"), [0, 1, 2]),
    ...     (Kernelizer(kernel="polynomial"), slice(3, 5)))
    >>> X = np.array([[0., 1., 2., 2., 3.],
                      [0., 2., 0., 0., 3.],
                      [0., 0., 1., 0., 3.],
    ...               [1., 1., 0., 1., 2.]])
    >>> # Kernelize separately the first three columns and the last two
    >>> # columns, creating two kernels of shape (n_samples, n_samples).
    >>> ck.fit_transform(X).shape
    (2, 4, 4)
    """
    # transformer_weights keyword is not passed through because the user
    # would need to know the automatically generated names of the transformers
    n_jobs = kwargs.pop('n_jobs', None)
    remainder = kwargs.pop('remainder', 'drop')
    verbose = kwargs.pop('verbose', False)
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'.format(
            list(kwargs.keys())[0]))
    transformer_list = _get_transformer_list(transformers)
    return ColumnKernelizer(transformer_list, n_jobs=n_jobs,
                            remainder=remainder, verbose=verbose)


def _get_transformer_list(estimators):
    """
    Construct (name, trans, column) tuples from list
    """
    transformers, columns = zip(*estimators)
    names, _ = zip(*_name_estimators(transformers))

    transformer_list = list(zip(names, transformers, columns))
    return transformer_list


def _end_with_a_kernel(estimator):
    """Return True if the estimator returns a kernel."""

    if not isinstance(estimator, BaseEstimator):
        raise ValueError("This function requires a scikit-learn estimator.")

    if getattr(estimator, "kernelizer", False):
        return True
    try:
        return _end_with_a_kernel(estimator[-1])
    except TypeError:
        pass

    return False


def _to_cpu(X):
    from ..validation import issparse
    backend = get_backend()
    if issparse(X):
        return X
    else:
        return backend.to_cpu(X)
