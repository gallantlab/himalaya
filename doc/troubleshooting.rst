Troobleshooting
===============
We detail here common issues encountered with ``himalaya``, and how to fix
them.

.. toctree::
   :maxdepth: 2


CUDA out of memory
------------------

The GPU memory is often smaller than the CPU memory, so it requires more
attention to avoid running out of memory. Himalaya implements a series of
option to limit the GPU memory, at the cost of computational speed:

- Some solvers implement computations over batches, to limit the size of
  intermediate arrays. See for instance ``n_targets_batch``, or
  ``n_alphas_batch`` in :class:`himalaya.kernel_ridge.KernelRidgeCV`.
- Some solvers implement an option to keep the input kernels or the targets in
  CPU memory. See for instance ``Y_in_cpu`` in
  :class:`kernel_ridge.MultipleKernelRidgeCV`.

Slow ``check_array``
--------------------

In himalaya, the scikit-learn compatible estimators validate the input data,
checking the absence of NaN or infinite values. For large datasets, this check
can take significant computational time. To skip this check, simply call
``sklearn.set_config(assume_finite=True)`` before fitting your models.
