.. raw:: html

   <h1>Himalaya: Multiple-target linear models</h1>

``Himalaya`` implements machine learning linear models in Python, focusing on
computational efficiency for large numbers of targets.

|Github| |Python| |License|

Use ``himalaya`` if you need a library that:

- estimates linear models on large numbers of targets,
- runs on CPU and GPU hardware,
- provides estimators compatible with ``scikit-learn``'s API.

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


Check more examples of use of ``himalaya`` in the gallery of examples.

Models
======

``Himalaya`` implements the following models:

- Ridge
- RidgeCV
- BandedRidgeCV
- KernelRidge
- KernelRidgeCV
- WeightedKernelRidge
- MultipleKernelRidgeCV
- SparseGroupLassoCV

Himalaya backends
=================

``Himalaya`` can be used seamlessly with different backends.
The available backends are ``numpy`` (default), ``cupy``, and ``pytorch``.
To change the backend (e.g. to ``cupy``), call:

.. code-block:: python

    from himalaya.backend import set_backend
    backend = set_backend("cupy")


and give ``cupy`` arrays inputs to the ``himalaya`` solvers. For convenience,
estimators implementing ``scikit-learn``'s API can cast arrays to the correct
input type.

GPU acceleration
----------------

To run ``himalaya`` on a graphics processing unit (GPU), you can use both
``cupy`` or ``pytorch`` backends.

To use the ``cupy`` backend, call:

.. code-block:: python

    from himalaya.backend import set_backend
    backend = set_backend("cupy")

    data = backend.asarray(data)  # cupy arrays are always on GPU


To use the ``pytorch`` backend, call:

.. code-block:: python

    from himalaya.backend import set_backend
    set_backend("torch")

    data = backend.asarray(data)  # torch tensors are on CPU by default...
    data = data.cuda()  # ...and you can move them to GPU with the `cuda` method.

    # or directly use
    set_backend("torch_cuda")
    data = backend.asarray(data)


Installation
============

Dependencies
------------

``Himalaya`` requires:

- Python 3
- Numpy
- Scikit-learn
- PyTorch (optional GPU backend)
- Cupy (optional GPU backend)
- Matplotlib (optional, for visualization only)
- Pytest (optional, for testing only)


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
