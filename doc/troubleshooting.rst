Troubleshooting
===============
We detail here common issues encountered with ``himalaya``, and how to fix
them.

.. toctree::
   :maxdepth: 2


CUDA out of memory
------------------

The GPU memory is often smaller than the CPU memory, so it requires more
attention to avoid running out of memory. Himalaya implements a series of
options to limit the GPU memory, often at the cost of computational speed:

- Some solvers implement computations over batches, to limit the size of
  intermediate arrays. See for instance ``n_targets_batch``, or
  ``n_alphas_batch`` in :class:`himalaya.kernel_ridge.KernelRidgeCV`.
- Some solvers implement an option to keep the input kernels or the targets in
  CPU memory. See for instance ``Y_in_cpu`` in
  :class:`kernel_ridge.MultipleKernelRidgeCV`.
- GPU memory can also be limited by limiting GPU use to some estimators only.
  To force one estimator to use the CPU ignoring the current backend, use the
  parameter ``force_cpu=True``. In the same pipeline, some estimators can use
  ``force_cpu=True`` and others ``force_cpu=False``. In particular, it is
  possible to precompute kernels on CPU, before fitting a
  :class:`himalaya.kernel_ridge.KernelRidgeCV` or a
  :class:`kernel_ridge.MultipleKernelRidgeCV` on GPU. To do so, use a
  :class:`kernel_ridge.Kernelizer` or :class:`kernel_ridge.ColumnKernelizer`
  with the parameter ``force_cpu=True``.

Slow ``check_array``
--------------------

In himalaya, the scikit-learn compatible estimators validate the input data,
checking the absence of NaN or infinite values. For large datasets, this check
can take significant computational time. To skip this check, simply call
``sklearn.set_config(assume_finite=True)`` before fitting your models.
