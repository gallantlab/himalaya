Troubleshooting
===============
We detail here common issues encountered with ``himalaya``, and how to fix
them.

.. toctree::
   :maxdepth: 2


GPU out of memory
-----------------

The GPU memory is often smaller than the CPU memory, so it requires more
attention to avoid running out of memory. Himalaya implements a series of
options to limit the GPU memory, often at the cost of computational speed:

- Some solvers implement computations over batches, to limit the size of
  intermediate arrays. See for instance ``n_targets_batch``, or
  ``n_alphas_batch`` in :class:`~himalaya.kernel_ridge.KernelRidgeCV`.
- Some solvers implement an option to keep the input kernels or the targets in
  CPU memory. See for instance ``Y_in_cpu`` in
  :class:`~himalaya.kernel_ridge.MultipleKernelRidgeCV`.
- Some estimators can also be forced to use CPU, ignoring the current backend,
  using the parameter ``force_cpu=True``. To limit GPU memory, some estimators
  in the same pipeline can use ``force_cpu=True`` and others
  ``force_cpu=False``. In particular, it is possible to precompute kernels on
  CPU, using :class:`~himalaya.kernel_ridge.Kernelizer` or
  :class:`~himalaya.kernel_ridge.ColumnKernelizer` with the parameter
  ``force_cpu=True`` before fitting a
  :class:`~himalaya.kernel_ridge.KernelRidgeCV` or a
  :class:`~himalaya.kernel_ridge.MultipleKernelRidgeCV` on GPU.

A CUDA out of memory issue can also arise with ``pytorch < 1.9``, for example
with :class:`~himalaya.kernel_ridge.KernelRidge`, where a solver requires
ridiculously high peak memory during a broadcasting matmul operation. This
`issue <https://github.com/pytorch/pytorch/pull/54616>`_ can be fixed by
updating to ``pytorch = 1.9`` or newer versions.


Slow check_array
----------------

In himalaya, the scikit-learn compatible estimators validate the input data,
checking the absence of NaN or infinite values. For large datasets, this check
can take significant computational time. To skip this check, simply call
``sklearn.set_config(assume_finite=True)`` before fitting your models.


Eigenvalue decomposition error in kernel ridge solvers
------------------------------------------------------

When using GPU backends (e.g. ``torch_cuda``, ``torch_mps``) with float32 precision, the
eigenvalue decomposition (``eigh``) used internally by
:class:`~himalaya.kernel_ridge.KernelRidgeCV` and
:class:`~himalaya.kernel_ridge.MultipleKernelRidgeCV` solvers can fail on
ill-conditioned kernel matrices. This typically raises errors from the
underlying ``eigh`` routine.

To work around this, pass ``solver_params=dict(diagonalize_method="svd")`` to
use SVD instead of ``eigh`` for the decomposition. SVD is slower but more
numerically robust::

    model = KernelRidgeCV(solver_params=dict(diagonalize_method="svd"))


Apple Silicon MPS backend (torch_mps)
--------------------------------------

The ``torch_mps`` backend uses Apple's Metal Performance Shaders for GPU
acceleration on Apple Silicon Macs. There are some limitations to be aware of:

- **float32 only**: MPS does not support float64, so all computations use
  float32 precision. Results may differ slightly from CPU backends.
- **CPU fallback for eigh/svd**: The MPS framework does not support eigenvalue
  decomposition or SVD, so these operations automatically fall back to CPU.
  This affects solvers in :class:`~himalaya.kernel_ridge.KernelRidgeCV` and
  :class:`~himalaya.kernel_ridge.MultipleKernelRidgeCV`.
- **Memory pressure**: MPS devices share memory with the system. Use
  ``n_targets_batch`` in ``solver_params`` to limit memory usage::

      model = KernelRidgeCV(solver_params=dict(n_targets_batch=200))
