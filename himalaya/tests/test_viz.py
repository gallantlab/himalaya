import numpy as np
import matplotlib.pyplot as plt
from himalaya.viz import plot_alphas_diagnostic


def test_smoke_viz():
    alphas = np.logspace(0, 5, 6)
    best_alphas = np.random.choice(np.logspace(0, 5, 6), 10)
    plot_alphas_diagnostic(best_alphas, alphas, ax=None)
    plot_alphas_diagnostic(best_alphas, alphas, ax=plt.gca())
