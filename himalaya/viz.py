import numpy as np


def plot_alphas_diagnostic(best_alphas, alphas, ax=None):
    """
    Plot a diagnostic plot for the selected alphas during cross-validation,
    to figure out whether to increase the range of alphas.

    Parameters
    ----------
    best_alphas : array of shape (n_targets, )
        Alphas selected during cross-validation for each target.
    alphas : array of shape (n_alphas)
        Alphas used while fitting the model.
    ax : None or figure axis

    Returns
    -------
    ax : figure axis
    """
    import matplotlib.pyplot as plt
    alphas = np.sort(alphas)
    n_alphas = len(alphas)
    indices = np.argmin(np.abs(best_alphas[None] - alphas[:, None]), 0)
    hist = np.bincount(indices, minlength=n_alphas)
    hist = hist / hist.sum() * 100

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    log10alphas = np.log(alphas) / np.log(10)
    ax.plot(log10alphas, hist, '.-', markersize=12)
    ax.set_ylabel('Density [%]')
    ax.set_xlabel('log10(alpha)')
    ax.grid("on")
    return ax
