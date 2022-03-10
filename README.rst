Himalaya: Multiple-target linear models
=======================================

|Github| |Python| |License| |Build| |Codecov| |Downloads|

``Himalaya`` implements machine learning linear models in Python, focusing on
computational efficiency for large numbers of targets.

Use ``himalaya`` if you need a library that:

- estimates linear models on large numbers of targets,
- runs on CPU and GPU hardware,
- provides estimators compatible with ``scikit-learn``'s API.

``Himalaya`` is stable (with particular care for backward compatibility) and
open for public use (give it a star!).

Example
=======

.. code-block:: python

    import numpy as np
    n_samples, n_features, n_targets = 10, 5, 4
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_targets)

    from himalaya.ridge import RidgeCV
    model = RidgeCV(alphas=[1, 10, 100])
    model.fit(X, Y)
    print(model.best_alphas_)  # [ 10. 100.  10. 100.]


- The model ``RidgeCV`` uses the same API as ``scikit-learn``
  estimators, with methods such as ``fit``, ``predict``, ``score``, etc.
- The model is able to efficiently fit a large number of targets (routinely
  used with 100k targets).
- The model selects the best hyperparameter ``alpha`` for each target
  independently.

Gallery of examples
-------------------

Check more examples of use of ``himalaya`` in the `gallery of examples
<https://gallantlab.github.io/himalaya/_auto_examples/index.html>`_.

Tutorials using ``himalaya`` for fMRI
-------------------------------------

``Himalaya`` was designed primarily for functional magnetic resonance imaging
(fMRI) encoding models. In depth tutorials about using ``himalaya`` for fMRI
encoding models can be found at `gallantlab/voxelwise_tutorials
<https://github.com/gallantlab/voxelwise_tutorials>`_.

Models
======

``Himalaya`` implements the following models:

- Ridge, RidgeCV
- KernelRidge, KernelRidgeCV
- GroupRidgeCV, MultipleKernelRidgeCV, WeightedKernelRidge
- SparseGroupLassoCV


See the `model descriptions
<https://gallantlab.github.io/himalaya/models.html>`_ in the documentation
website.

Himalaya backends
=================

``Himalaya`` can be used seamlessly with different backends.
The available backends are ``numpy`` (default), ``cupy``, ``torch``, and
``torch_cuda``.
To change the backend, call:

.. code-block:: python

    from himalaya.backend import set_backend
    backend = set_backend("torch")


and give ``torch`` arrays inputs to the ``himalaya`` solvers. For convenience,
estimators implementing ``scikit-learn``'s API can cast arrays to the correct
input type.

GPU acceleration
----------------

To run ``himalaya`` on a graphics processing unit (GPU), you can use either
the ``cupy`` or the ``torch_cuda`` backend:

.. code-block:: python

    from himalaya.backend import set_backend
    backend = set_backend("cupy")  # or "torch_cuda"

    data = backend.asarray(data)


Installation
============

Dependencies
------------

- Python 3
- Numpy
- Scikit-learn

Optional (GPU backends):

- PyTorch (1.9+ preferred)
- Cupy


Standard installation
---------------------
You may install the latest version of ``himalaya`` using the package manager
``pip``, which will automatically download ``himalaya`` from the Python Package
Index (PyPI):

.. code-block:: bash

    pip install himalaya


Installation from source
------------------------

To install ``himalaya`` from the latest source (``main`` branch), you may
call:

.. code-block:: bash

    pip install git+https://github.com/gallantlab/himalaya.git


Developers can also install ``himalaya`` in editable mode via:

.. code-block:: bash

    git clone https://github.com/gallantlab/himalaya
    cd himalaya
    pip install --editable .


.. |Github| image:: https://img.shields.io/badge/github-himalaya-blue
   :target: https://github.com/gallantlab/himalaya

.. |Python| image:: https://img.shields.io/badge/python-3.7%2B-blue
   :target: https://www.python.org/downloads/release/python-370

.. |License| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause

.. |Build| image:: https://github.com/gallantlab/himalaya/actions/workflows/run_tests.yml/badge.svg
   :target: https://github.com/gallantlab/himalaya/actions/workflows/run_tests.yml

.. |Codecov| image:: https://codecov.io/gh/gallantlab/himalaya/branch/main/graph/badge.svg?token=ECzjd9gvrw
   :target: https://codecov.io/gh/gallantlab/himalaya

.. |Downloads| image:: https://pepy.tech/badge/himalaya
   :target: https://pepy.tech/project/himalaya


Cite this package
=================

If you use ``himalaya`` in your work, please give it a star and cite our
(future) publication:

.. [1] Dupré La Tour, T., Eickenberg, M., & Gallant, J. L. (2021).
	Feature-space selection with banded ridge regression. *In preparation*.
