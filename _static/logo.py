import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from himalaya.kernel_ridge import generate_dirichlet_samples
kernel_weights = generate_dirichlet_samples(10000, n_kernels=3,
                                            concentration=[1.], random_state=0)
pca = PCA(2).fit(kernel_weights)

darkgreen = "#446455"
white = "white"


def plot_simplex(bias=(0, 0), ax=None):
    corners = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    corners = pca.transform(corners).T

    if ax is None:
        plt.figure(figsize=(2, 2))
        ax = plt.gca()

    # Faces
    ax.add_patch(
        Polygon(corners[:2].T + bias, closed=True, edgecolor=None, fill=True,
                facecolor=white, alpha=0.6))
    # Edges
    ax.add_patch(
        Polygon(corners[:2].T + bias, closed=True, edgecolor=darkgreen,
                fill=False, alpha=1, linewidth=2))

    ax.axis('equal')
    ax.axis('off')
    return ax


fig, ax = plt.subplots(figsize=(2, 2))

bias = [0.4, 0.04]
for factor in np.linspace(2, 0, 3):
    bias_ = np.array(bias) * factor
    plot_simplex(bias_, ax=ax)

ax.text(-0.4, -1, "himalaya", fontsize=20, color=darkgreen)
fig.savefig("logo.svg", bbox_inches='tight', pad_inches=0)
