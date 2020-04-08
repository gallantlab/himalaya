# Himalaya: Multiple-target machine learning

Himalaya implements machine learning models in the Python programming language,
focusing on computational efficiency for large numbers of targets.

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Use Himalaya if you need a library that:
* estimates models on large numbers of targets
* runs on CPU and GPU hardware

## Himalaya backends

Himalaya can be used seamlessly with different backends.
The available backends are Numpy (default), Cupy, and PyTorch.
To change the backend (e.g. to Cupy), call:

```python
from himalaya.backend import set_backend
backend = set_backend("cupy")
```

and give `cupy` arrays inputs to the Himalaya solvers. 

### GPU acceleration

To run Himalaya on a graphics processing unit (GPU), you can use both Cupy
or PyTorch backends.

To use the Cupy backend, call:

```python
from himalaya.backend import set_backend
backend = set_backend("cupy")

data = backend.asarray(data)  # cupy arrays are always on GPU
```

To use the PyTorch backend, call:

```python
from himalaya.backend import set_backend
set_backend("torch")

data = backend.asarray(data)  # torch tensors are on CPU by default...
data = data.cuda()  # ...and you can move them to GPU with the `cuda` method.

# or directly use
set_backend("torch_cuda")
data = backend.asarray(data)
```

## Installation

### Dependencies

Himalaya requires:

* Python 3
* Numpy
* Scikit-learn
* PyTorch (optional backend)
* Cupy (optional backend)
* Matplotlib (optional, for visualization only)
* Pytest (optional, for testing only)

<!--
### Standard installation
You may install the latest version of Himalaya using the package manager `pip`,
which will automatically download Himalaya from the Python Package Index
(PyPI):

```
pip install himalaya
```
-->

### Installation from source

To install Himalaya from the latest source (`master` branch), you may call:

```bash
pip install git+https://github.com/gallantlab/himalaya.git
```

Assuming the source has been downloaded manually, you may install it by
running:

```bash
pip install .
```

Developers can also install Himalaya in editable mode via:

```bash
pip install --editable .
```

### Examples

To get examples of use of Himalaya, see the ["examples"](examples) directory.
