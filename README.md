# Himalaya: Multiple-target machine learning

Himalaya implements machine learning models in the Python programming language,
focusing on computational efficiency for large numbers of targets.

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Use Himalaya if you need a library that:
* estimates models on large numbers of targets
* runs on CPU and GPU hardware

## Himalaya backends

Himalaya can be used seamlessly with different backends.
The available backends are Numpy, and PyTorch.
To change the backend (e.g. to PyTorch), simply call:

```python
from himalaya.backend import change_backend
change_backend("torch")
```

and give `torch.Tensor` inputs to Himalaya solvers.

### GPU acceleration

To run Himalaya on a graphics processing unit (GPU), you can use the PyTorch
backend, and move your data to GPU memory with the `.cuda()` method.

```python
from himalaya.backend import change_backend
change_backend("torch")

...  # (load data as a torch.Tensor)
data = data.cuda()  # then move the data to GPU
```

## Installation

### Dependencies

Himalaya requires:

* Python
* Numpy
* Scikit-learn
* PyTorch (optional backend)

<!--
### Standard installation
You may install the latest version of Himalaya using the package manager `pip`,
which will automatically download Himalaya from the Python Package Index (PyPI):

```
pip install himalaya
```
-->

### Installation from source

To install Himalaya from the latest source (`master` branch), you may call:

```bash
pip install git+https://github.com/TomDLT/himalaya.git#egg=himalaya
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
