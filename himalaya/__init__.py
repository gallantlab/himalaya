from . import backend, kernel_ridge, lasso, progress_bar, ridge, scoring, utils, viz

try:
    from ._version import __version__
except ImportError:  # running from a source tree that was never built/installed
    __version__ = "unknown"

__all__ = [
    backend,
    kernel_ridge,
    lasso,
    ridge,
    progress_bar,
    scoring,
    utils,
    viz,
]
