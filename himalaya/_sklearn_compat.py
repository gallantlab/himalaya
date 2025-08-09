"""
Custom sklearn compatibility layer for Himalaya.

This module provides compatibility across different sklearn versions,
specifically handling the transition from sklearn < 1.6 to >= 1.6.
"""

import warnings
from dataclasses import dataclass, field, fields
from packaging.version import parse as parse_version

import sklearn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

# Version detection
sklearn_version = parse_version(sklearn.__version__)
SKLEARN_1_6_PLUS = sklearn_version >= parse_version("1.6.0")

# Import version-specific components
if SKLEARN_1_6_PLUS:
    try:
        from sklearn.utils._tags import Tags
        from sklearn.base import validate_data as sklearn_validate_data
        _HAS_NEW_TAGS = True
        _HAS_VALIDATE_DATA = True
    except ImportError:
        _HAS_NEW_TAGS = False
        _HAS_VALIDATE_DATA = False
else:
    _HAS_NEW_TAGS = False
    _HAS_VALIDATE_DATA = False


# Custom Tags class with _xfail_checks support
if _HAS_NEW_TAGS:
    @dataclass
    class HimalayaTags(Tags):
        """Custom Tags class with _xfail_checks support."""
        _xfail_checks: dict = field(default_factory=dict)




def get_estimator_tags(estimator_type="ridge"):
    """
    Get appropriate tags for an estimator, handling sklearn version differences.
    
    Parameters
    ----------
    estimator_type : str
        Type of estimator: "ridge", "kernel_ridge", or "lasso".
        
    Returns
    -------
    tags : Tags or dict
        Appropriate tags object/dict for the sklearn version.
    """
    if _HAS_NEW_TAGS:
        # sklearn >= 1.6: use dataclass-based tags
        from sklearn.utils._tags import default_tags, TargetTags, InputTags
        
        # Start with default regressor tags
        tags = default_tags('regressor')
        
        # Customize target tags - set required=True
        target_tags = TargetTags(
            required=True,
            one_d_labels=False,
            two_d_labels=False,
            positive_only=False,
            multi_output=True,  # All Himalaya estimators support multi-output
            single_output=True
        )
        tags.target_tags = target_tags
        
        # Customize for specific estimator types
        if estimator_type == "kernel_ridge":
            # Convert to HimalayaTags to support _xfail_checks
            himalaya_tags = HimalayaTags(
                estimator_type=tags.estimator_type,
                target_tags=tags.target_tags,
                transformer_tags=tags.transformer_tags,
                classifier_tags=tags.classifier_tags,
                regressor_tags=tags.regressor_tags,
                array_api_support=tags.array_api_support,
                no_validation=tags.no_validation,
                non_deterministic=tags.non_deterministic,
                requires_fit=tags.requires_fit,
                _skip_test=tags._skip_test,
                input_tags=tags.input_tags,
                _xfail_checks={
                    'check_sample_weights_invariance':
                    'zero sample_weight is not equivalent to removing samples, '
                    'because of the cross-validation splits.',
                }
            )
            
            # Allow sparse input for kernel ridge
            input_tags = InputTags(
                one_d_array=False,
                two_d_array=True,
                three_d_array=False,
                sparse=True,  # Enable sparse support
                categorical=False,
                string=False,
                dict=False,
                positive_only=False,
                allow_nan=False,
                pairwise=False
            )
            himalaya_tags.input_tags = input_tags
            tags = himalaya_tags
            
        return tags
    else:
        # sklearn < 1.6: use dictionary-based tags
        return {'requires_y': True}


def create_sklearn_tags_method(estimator_type="ridge"):
    """
    Create appropriate __sklearn_tags__ method for an estimator.
    
    Parameters
    ----------
    estimator_type : str
        Type of estimator: "ridge", "kernel_ridge", or "lasso".
        
    Returns
    -------
    method : callable
        Method to use as __sklearn_tags__ or None for older sklearn.
    """
    if _HAS_NEW_TAGS:
        def __sklearn_tags__(self):
            return get_estimator_tags(estimator_type)
        return __sklearn_tags__
    else:
        return None


def create_more_tags_method():
    """
    Create _more_tags method for sklearn < 1.6.
    
    Returns
    -------
    method : callable
        Method to use as _more_tags for older sklearn versions.
    """
    def _more_tags(self):
        return {'requires_y': True}
    return _more_tags


def validate_data(estimator, X, y=None, reset=True, validate_separately=False,
                  cast_to_ndarray=True, **check_params):
    """
    Input validation for estimators compatible with Himalaya's backend system.
    
    This function provides sklearn 1.6+ compatible validate_data functionality
    while working with Himalaya's multi-backend architecture.
    
    Parameters
    ----------
    estimator : estimator instance
        The estimator instance.
    X : array-like
        The input samples.
    y : array-like, default=None
        The target values.
    reset : bool, default=True
        Whether to reset the `n_features_in_` attribute.
    validate_separately : bool, default=False
        Whether to validate X and y separately.
    cast_to_ndarray : bool, default=True
        Whether to cast to ndarray (for backward compatibility).
    **check_params
        Additional parameters passed to check_array.
        
    Returns
    -------
    X_validated : array-like
        The validated input samples.
    y_validated : array-like, optional
        The validated target values (if y is not None).
    """
    # Import here to avoid circular imports
    from .validation import check_array
    
    if y is None:
        # Only validate X
        X_validated = check_array(X, **check_params)
        
        # Set n_features_in_ for sklearn compatibility
        if reset or not hasattr(estimator, 'n_features_in_'):
            estimator.n_features_in_ = X_validated.shape[1]
        elif hasattr(estimator, 'n_features_in_'):
            # Check feature count consistency
            if X_validated.shape[1] != estimator.n_features_in_:
                raise ValueError(
                    f"X has {X_validated.shape[1]} features, but {estimator.__class__.__name__} "
                    f"is expecting {estimator.n_features_in_} features as input."
                )
        
        return X_validated
    
    else:
        # Validate both X and y
        if validate_separately:
            X_validated = check_array(X, **check_params)
            y_check_params = check_params.copy()
            y_check_params.pop('accept_sparse', None)  # y typically shouldn't be sparse
            # Allow y to have flexible dimensions (1D or 2D)
            if 'ndim' in y_check_params:
                y_check_params['ndim'] = [1, 2]
            y_validated = check_array(y, **y_check_params)
        else:
            # Use default check_array approach - allow y to be flexible
            X_validated = check_array(X, **check_params)
            y_check_params = check_params.copy()
            y_check_params.pop('accept_sparse', None)  
            y_check_params.pop('ndim', None)  # Allow y flexible dimensions
            y_validated = check_array(y, **y_check_params)
        
        # Check consistent number of samples
        if X_validated.shape[0] != y_validated.shape[0]:
            raise ValueError(
                f"X and y have inconsistent numbers of samples: "
                f"{X_validated.shape[0]} != {y_validated.shape[0]}"
            )
        
        # Set n_features_in_ for sklearn compatibility
        if reset or not hasattr(estimator, 'n_features_in_'):
            estimator.n_features_in_ = X_validated.shape[1]
        elif hasattr(estimator, 'n_features_in_'):
            # Check feature count consistency  
            if X_validated.shape[1] != estimator.n_features_in_:
                raise ValueError(
                    f"X has {X_validated.shape[1]} features, but {estimator.__class__.__name__} "
                    f"is expecting {estimator.n_features_in_} features as input."
                )
        
        return X_validated, y_validated


def setup_estimator_tags(cls, estimator_type="ridge"):
    """
    Setup appropriate tags methods for an estimator class.
    
    Parameters
    ----------
    cls : class
        The estimator class to modify.
    estimator_type : str
        Type of estimator: "ridge", "kernel_ridge", or "lasso".
        
    Returns
    -------
    cls : class
        The modified estimator class.
    """
    if _HAS_NEW_TAGS:
        # Add __sklearn_tags__ method for sklearn >= 1.6
        tags_method = create_sklearn_tags_method(estimator_type)
        if tags_method:
            setattr(cls, '__sklearn_tags__', tags_method)
    
    # Always add _more_tags for backward compatibility
    more_tags_method = create_more_tags_method()
    setattr(cls, '_more_tags', more_tags_method)
    
    return cls


# Convenience functions for specific estimator types
def setup_ridge_tags(cls):
    """Setup tags for Ridge estimators."""
    return setup_estimator_tags(cls, "ridge")


def setup_kernel_ridge_tags(cls):
    """Setup tags for KernelRidge estimators."""  
    return setup_estimator_tags(cls, "kernel_ridge")


def setup_lasso_tags(cls):
    """Setup tags for Lasso estimators."""
    return setup_estimator_tags(cls, "lasso")