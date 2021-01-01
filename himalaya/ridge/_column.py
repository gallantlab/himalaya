from sklearn.compose import ColumnTransformer
from sklearn.pipeline import _name_estimators


class ColumnTransformerNoStack(ColumnTransformer):
    """Applies transformers to columns of an array, and does not stack them.

    This estimator allows different columns or column subsets of the input to
    be transformed separately. The different groups of features generated are
    *not* stacked together, to be used e.g. in a BandedRidgeCV(groups="auto").
    This is useful to perform separate transformations on different feature
    spaces.

    Warning : This class does not perfectly abide by scikit-learn's API.
    Indeed, it returns a list of ``n_groups`` matrices of shape (n_samples,
    n_features_i), while scikit-learn's API only allows arrays of shape
    (n_samples, n_features). This class is intended to be used in a
    scikit-learn pipeline *just* before a BandedRidgeCV(groups="auto").

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
            untransformed, respectively.
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
        Number of features used during the fit.

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
    himalaya.ridge.make_column_transform_no_stack : convenience function for
        combining the outputs of multiple pipelines applied to
        column subsets of the original feature space.
    sklearn.compose.make_column_selector : convenience function for selecting
        columns based on datatype or the columns name with a regex pattern.

    Examples
    --------
    >>> import numpy as np
    >>> from himalaya.ridge import ColumnTransformerNoStack
    >>> from sklearn.preprocessing import StandardScaler
    >>> ct = ColumnTransformerNoStack(
    ...     [("group_1", StandardScaler(), [0, 1, 2]),
    ...      ("group_2", StandardScaler(), slice(3, 5))])
    >>> X = np.random.randn(10, 5)
    >>> # Group separately the first three columns and the last two
    >>> # columns, creating two feature spaces.
    >>> Xs = ct.fit_transform(X)
    >>> print(Xi.shape for Xi in Xs)
    (2, 4, 4)
    """

    def _hstack(self, Xs):
        """Do *not* stack the feature spaces.

        In ColumnTransformer, this methods stacks Xs horizontally.
        Here instead, we return the list of Xs.

        Parameters
        ----------
        Xs : list of arrays of shape (n_samples, n_features_i)
        """
        return Xs


def make_column_transformer_no_stack(*transformers, **kwargs):
    """Construct a ColumnTransformerNoStack from the given transformers.

    This is a shorthand for the ColumnTransformerNoStack constructor; it does
    not require, and does not permit, naming the transformers. Instead, they
    will be given names automatically based on their types. It also does not
    allow weighting with ``transformer_weights``.

    Parameters
    ----------
    *transformers : tuples
        Tuples of the form (transformer, columns) specifying the
        transformer objects to be applied to subsets of the data.

        transformer : {'drop', 'passthrough'} or estimator
            Estimator must support ``fit`` and ``transform``.
            Special-cased strings 'drop' and 'passthrough' are accepted as
            well, to indicate to drop the columns or to pass them through
            untransformed, respectively.
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
    column_transformer_no_stack : ColumnTransformerNoStack

    See also
    --------
    himalaya.ridge.ColumnTransformerNoStack : Class that allows combining the
        outputs of multiple transformer objects used on column subsets of the
        data into a single feature space.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from himalaya.ridge import make_column_transformer_no_stack
    >>> ct = make_column_transformer_no_stack(
    ...     (StandardScaler(), [0, 1, 2]),
    ...     (StandardScaler(), slice(3, 5)))
    >>> # Group separately the first three columns and the last two
    >>> # columns, creating two feature spaces.
    >>> Xs = ct.fit_transform(X)
    >>> print([Xi.shape for Xi in Xs])
    [(4, 3), (4, 2)]
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
    return ColumnTransformerNoStack(transformer_list, n_jobs=n_jobs,
                                    remainder=remainder, verbose=verbose)


def _get_transformer_list(estimators):
    """
    Construct (name, trans, column) tuples from list
    """
    transformers, columns = zip(*estimators)
    names, _ = zip(*_name_estimators(transformers))

    transformer_list = list(zip(names, transformers, columns))
    return transformer_list
