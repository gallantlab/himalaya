#!/usr/bin/env python
"""Benchmark himalaya backends: speed and correctness of RidgeCV, KernelRidgeCV,
and MultipleKernelRidgeCV across numpy, torch (CPU), torch_cuda, and torch_mps.

Usage
-----
    python benchmarks/benchmark_backends.py [OPTIONS]

Options
-------
    --n_samples INT       Number of samples (default: 3000)
    --n_features INT      Number of features (default: 1000)
    --n_targets INT       Number of targets (default: 30000)
    --n_alphas INT        Number of alpha values (default: 20)
    --alpha_max FLOAT     Upper bound for logspace alphas (default: 20)
    --cv INT              Number of CV folds (default: 3)
    --n_iter INT          Random-search iterations for MKR (default: 5)
    --n_repetitions INT   Repetitions per benchmark (default: 10)
    --seed INT            Random seed (default: 42)
    --backends STR,...     Comma-separated list of backends to test
                          (default: auto-detect available)
    --output_dir DIR      Directory for result files
                          (default: benchmarks/benchmark_results)
    --float64             Use float64 input instead of the default float32
    --fast                Use small dimensions for a quick smoke test
    --warmup INT          Untimed warmup iterations for GPU backends (default: 1)
    --n_targets_batch INT Target batch size for CV (default: 5000)

The script generates (datetime-stamped so successive runs don't overwrite):
    <output_dir>/benchmark_<YYYYMMDD_HHMMSS>.json   Full results with metadata
    <output_dir>/benchmark_<YYYYMMDD_HHMMSS>.csv    Flat table for quick comparison
"""
import argparse
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np


def _get_system_info():
    """Collect system and package metadata."""
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
    }
    try:
        import himalaya
        info["himalaya_version"] = himalaya.__version__
    except Exception:
        info["himalaya_version"] = "unknown"
    try:
        info["numpy_version"] = np.__version__
    except Exception:
        info["numpy_version"] = "unknown"
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
        info["mps_available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        info["torch_version"] = "not installed"
        info["cuda_available"] = False
        info["mps_available"] = False
    return info


def _detect_backends():
    """Return list of backends that are available on this machine."""
    available = ["numpy"]
    try:
        import torch  # noqa: F401
        available.append("torch")
    except ImportError:
        return available
    try:
        if torch.cuda.is_available():
            available.append("torch_cuda")
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            available.append("torch_mps")
    except Exception:
        pass
    return available


def _to_numpy(arr):
    """Convert any backend array to a numpy float64 array for comparison."""
    try:
        return np.asarray(arr.detach().cpu(), dtype=np.float64)
    except Exception:
        pass
    try:
        return np.asarray(arr, dtype=np.float64)
    except Exception:
        return np.array(arr, dtype=np.float64)


def _sync_backend(backend_name):
    """Synchronise GPU to get accurate timings."""
    if backend_name == "torch_cuda":
        import torch
        torch.cuda.synchronize()
    elif backend_name == "torch_mps":
        import torch
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def _compare_arrays(ref, test, label):
    """Compare two arrays and return a dict of numerical-difference metrics."""
    ref = _to_numpy(ref).ravel()
    test = _to_numpy(test).ravel()
    if ref.shape != test.shape:
        return {
            "label": label,
            "error": f"shape mismatch: {ref.shape} vs {test.shape}",
        }
    abs_diff = np.abs(ref - test)
    rel_diff = abs_diff / (np.abs(ref) + 1e-30)
    return {
        "label": label,
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
        "correlation": float(np.corrcoef(ref, test)[0, 1])
        if ref.size > 1 else None,
    }


# ---------------------------------------------------------------------------
# Benchmark helpers – each returns (fitted_model, predictions, time_fit,
# time_predict, extra_outputs) where extra_outputs is a dict of arrays that
# should be compared across backends.
# ---------------------------------------------------------------------------

def _run_ridgecv(X_train, Y_train, X_test, alphas, cv, backend_name,
                 n_targets_batch=None):
    from himalaya.backend import set_backend
    from himalaya.ridge import RidgeCV

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        set_backend(backend_name)

    backend = __import__("himalaya.backend", fromlist=["get_backend"]).get_backend()
    X_tr = backend.asarray(X_train)
    Y_tr = backend.asarray(Y_train)
    X_te = backend.asarray(X_test)

    solver_params = {}
    if n_targets_batch is not None:
        solver_params["n_targets_batch"] = n_targets_batch
    model = RidgeCV(alphas=alphas, cv=cv,
                    solver_params=solver_params or None)

    _sync_backend(backend_name)
    t0 = time.perf_counter()
    model.fit(X_tr, Y_tr)
    _sync_backend(backend_name)
    t_fit = time.perf_counter() - t0

    _sync_backend(backend_name)
    t0 = time.perf_counter()
    preds = model.predict(X_te)
    _sync_backend(backend_name)
    t_pred = time.perf_counter() - t0

    extra = {
        "coef": model.coef_,
        "best_alphas": model.best_alphas_,
        "cv_scores": model.cv_scores_,
    }
    return model, preds, t_fit, t_pred, extra


def _run_kernelridgecv(X_train, Y_train, X_test, alphas, cv, backend_name,
                       n_targets_batch=None):
    from himalaya.backend import set_backend
    from himalaya.kernel_ridge import KernelRidgeCV

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        set_backend(backend_name)

    backend = __import__("himalaya.backend", fromlist=["get_backend"]).get_backend()
    X_tr = backend.asarray(X_train)
    Y_tr = backend.asarray(Y_train)
    X_te = backend.asarray(X_test)

    solver_params = {}
    if n_targets_batch is not None:
        solver_params["n_targets_batch"] = n_targets_batch
    model = KernelRidgeCV(alphas=alphas, kernel="linear", cv=cv, warn=False,
                          solver_params=solver_params or None)

    _sync_backend(backend_name)
    t0 = time.perf_counter()
    model.fit(X_tr, Y_tr)
    _sync_backend(backend_name)
    t_fit = time.perf_counter() - t0

    _sync_backend(backend_name)
    t0 = time.perf_counter()
    preds = model.predict(X_te)
    _sync_backend(backend_name)
    t_pred = time.perf_counter() - t0

    extra = {
        "dual_coef": model.dual_coef_,
        "best_alphas": model.best_alphas_,
        "cv_scores": model.cv_scores_,
    }
    return model, preds, t_fit, t_pred, extra


def _run_multiplekernelridgecv(
    X_train, Y_train, X_test, cv, n_iter, random_state, backend_name,
    n_targets_batch=None
):
    from himalaya.backend import set_backend
    from himalaya.kernel_ridge import MultipleKernelRidgeCV

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        set_backend(backend_name)

    backend = __import__("himalaya.backend", fromlist=["get_backend"]).get_backend()
    X_tr = backend.asarray(X_train)
    Y_tr = backend.asarray(Y_train)
    X_te = backend.asarray(X_test)

    solver_params = dict(n_iter=n_iter, progress_bar=False)
    if n_targets_batch is not None:
        solver_params["n_targets_batch"] = n_targets_batch
    model = MultipleKernelRidgeCV(
        kernels=["linear", "polynomial"],
        solver="random_search",
        solver_params=solver_params,
        cv=cv,
        random_state=random_state,
    )

    _sync_backend(backend_name)
    t0 = time.perf_counter()
    model.fit(X_tr, Y_tr)
    _sync_backend(backend_name)
    t_fit = time.perf_counter() - t0

    _sync_backend(backend_name)
    t0 = time.perf_counter()
    preds = model.predict(X_te)
    _sync_backend(backend_name)
    t_pred = time.perf_counter() - t0

    extra = {
        "dual_coef": model.dual_coef_,
        "deltas": model.deltas_,
        "best_alphas": model.best_alphas_,
    }
    return model, preds, t_fit, t_pred, extra


# ---------------------------------------------------------------------------
# Main benchmark logic
# ---------------------------------------------------------------------------

MODEL_RUNNERS = {
    "RidgeCV": _run_ridgecv,
    "KernelRidgeCV": _run_kernelridgecv,
    "MultipleKernelRidgeCV": _run_multiplekernelridgecv,
}


def run_benchmarks(args):
    rng = np.random.RandomState(args.seed)

    # Generate synthetic data once (float32 by default, float64 if requested)
    dtype = np.float64 if args.float64 else np.float32
    X = rng.randn(args.n_samples, args.n_features).astype(dtype)
    # Create targets with a real linear relationship + noise so models
    # produce meaningful results
    W_true = rng.randn(args.n_features, args.n_targets).astype(dtype)
    Y = X @ W_true + 0.5 * rng.randn(args.n_samples, args.n_targets).astype(dtype)

    split = int(0.8 * args.n_samples)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    alphas = np.logspace(-3, args.alpha_max, args.n_alphas, dtype=dtype).tolist()

    backends = args.backends
    print(f"Backends to benchmark: {backends}")
    print(
        f"Data: n_samples={args.n_samples}, n_features={args.n_features}, "
        f"n_targets={args.n_targets}, dtype={dtype.__name__}"
    )
    print(f"Repetitions per benchmark: {args.n_repetitions}")
    if args.warmup > 0:
        print(f"Warmup iterations for GPU backends: {args.warmup}")
    if args.n_targets_batch is not None:
        print(f"Target batch size: {args.n_targets_batch}")
    print()

    all_results = []

    for model_name in MODEL_RUNNERS:
        print(f"{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        # Reference results come from the first backend (usually numpy)
        ref_preds = None
        ref_extras = None
        ref_backend = None

        for backend_name in backends:
            timings_fit = []
            timings_pred = []
            last_preds = None
            last_extras = None
            error_msg = None

            try:
                # Warmup iterations for GPU backends (untimed)
                n_warmup = args.warmup if backend_name in (
                    "torch_cuda", "torch_mps") else 0
                n_tb = getattr(args, "n_targets_batch", None)
                for _ in range(n_warmup):
                    if model_name == "MultipleKernelRidgeCV":
                        _run_multiplekernelridgecv(
                            X_train, Y_train, X_test,
                            cv=args.cv, n_iter=args.n_iter,
                            random_state=args.seed,
                            backend_name=backend_name,
                            n_targets_batch=n_tb,
                        )
                    else:
                        runner = MODEL_RUNNERS[model_name]
                        runner(
                            X_train, Y_train, X_test,
                            alphas=alphas, cv=args.cv,
                            backend_name=backend_name,
                            n_targets_batch=n_tb,
                        )

                for rep in range(args.n_repetitions):
                    if model_name == "MultipleKernelRidgeCV":
                        _, preds, t_fit, t_pred, extras = (
                            _run_multiplekernelridgecv(
                                X_train, Y_train, X_test,
                                cv=args.cv, n_iter=args.n_iter,
                                random_state=args.seed,
                                backend_name=backend_name,
                                n_targets_batch=n_tb,
                            )
                        )
                    else:
                        runner = MODEL_RUNNERS[model_name]
                        _, preds, t_fit, t_pred, extras = runner(
                            X_train, Y_train, X_test,
                            alphas=alphas, cv=args.cv,
                            backend_name=backend_name,
                            n_targets_batch=n_tb,
                        )
                    timings_fit.append(t_fit)
                    timings_pred.append(t_pred)
                    last_preds = preds
                    last_extras = extras
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                print(f"  [{backend_name}] ERROR: {error_msg}")

            # Record timing stats
            record = {
                "model": model_name,
                "backend": backend_name,
                "n_samples": args.n_samples,
                "n_features": args.n_features,
                "n_targets": args.n_targets,
                "input_dtype": dtype.__name__,
                "n_repetitions": args.n_repetitions,
            }

            if error_msg:
                record["status"] = "error"
                record["error"] = error_msg
                record["fit_time_mean"] = None
                record["fit_time_std"] = None
                record["fit_time_min"] = None
                record["predict_time_mean"] = None
                record["predict_time_std"] = None
                record["predict_time_min"] = None
                record["total_time_mean"] = None
                record["total_time_std"] = None
                all_results.append(record)
                continue

            record["status"] = "ok"
            record["fit_time_mean"] = float(np.mean(timings_fit))
            record["fit_time_std"] = float(np.std(timings_fit))
            record["fit_time_min"] = float(np.min(timings_fit))
            record["predict_time_mean"] = float(np.mean(timings_pred))
            record["predict_time_std"] = float(np.std(timings_pred))
            record["predict_time_min"] = float(np.min(timings_pred))
            timings_total = [f + p for f, p in zip(timings_fit, timings_pred)]
            record["total_time_mean"] = float(np.mean(timings_total))
            record["total_time_std"] = float(np.std(timings_total))

            # Print timing summary
            print(
                f"  [{backend_name:12s}] fit: {record['fit_time_mean']:.4f}s "
                f"(+/- {record['fit_time_std']:.4f}s), "
                f"predict: {record['predict_time_mean']:.4f}s "
                f"(+/- {record['predict_time_std']:.4f}s)"
            )

            # Correctness comparison against reference backend
            if ref_preds is None:
                ref_preds = last_preds
                ref_extras = last_extras
                ref_backend = backend_name
                record["reference_backend"] = None
                record["predictions_max_abs_diff"] = 0.0
                record["predictions_mean_abs_diff"] = 0.0
                record["predictions_max_rel_diff"] = 0.0
                record["predictions_correlation"] = 1.0
                # Extra per-attribute diffs stored only in JSON
                record["attribute_diffs"] = {}
            else:
                record["reference_backend"] = ref_backend
                pred_cmp = _compare_arrays(ref_preds, last_preds, "predictions")
                record["predictions_max_abs_diff"] = pred_cmp.get(
                    "max_abs_diff"
                )
                record["predictions_mean_abs_diff"] = pred_cmp.get(
                    "mean_abs_diff"
                )
                record["predictions_max_rel_diff"] = pred_cmp.get(
                    "max_rel_diff"
                )
                record["predictions_correlation"] = pred_cmp.get("correlation")

                attr_diffs = {}
                if ref_extras and last_extras:
                    for key in ref_extras:
                        if key in last_extras:
                            cmp = _compare_arrays(
                                ref_extras[key], last_extras[key], key
                            )
                            attr_diffs[key] = cmp
                record["attribute_diffs"] = attr_diffs

                # Print correctness summary
                corr = record["predictions_correlation"]
                corr_str = f"{corr:.8f}" if corr is not None else "N/A"
                print(
                    f"               vs {ref_backend}: "
                    f"max_abs_diff={record['predictions_max_abs_diff']:.2e}, "
                    f"mean_abs_diff={record['predictions_mean_abs_diff']:.2e}, "
                    f"correlation={corr_str}"
                )

            all_results.append(record)

        # Reset reference for next model
        ref_preds = None
        ref_extras = None
        ref_backend = None
        print()

    return all_results


def write_json(results, system_info, args, path):
    """Write full results with metadata to JSON."""
    output = {
        "system_info": system_info,
        "parameters": {
            "n_samples": args.n_samples,
            "n_features": args.n_features,
            "n_targets": args.n_targets,
            "n_alphas": args.n_alphas,
            "alpha_max": args.alpha_max,
            "cv": args.cv,
            "n_iter": args.n_iter,
            "n_repetitions": args.n_repetitions,
            "seed": args.seed,
            "float64": args.float64,
            "backends": args.backends,
        },
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"JSON results written to {path}")


def write_csv(results, path):
    """Write a flat CSV table with the key columns."""
    columns = [
        "model",
        "backend",
        "status",
        "n_samples",
        "n_features",
        "n_targets",
        "input_dtype",
        "n_repetitions",
        "fit_time_mean",
        "fit_time_std",
        "fit_time_min",
        "predict_time_mean",
        "predict_time_std",
        "predict_time_min",
        "total_time_mean",
        "total_time_std",
        "reference_backend",
        "predictions_max_abs_diff",
        "predictions_mean_abs_diff",
        "predictions_max_rel_diff",
        "predictions_correlation",
    ]
    with open(path, "w") as f:
        f.write(",".join(columns) + "\n")
        for row in results:
            vals = []
            for col in columns:
                v = row.get(col, "")
                if v is None:
                    v = ""
                vals.append(str(v))
            f.write(",".join(vals) + "\n")
    print(f"CSV results written to {path}")


def print_summary_table(results):
    """Print a Markdown summary table to stdout."""
    # Build a lookup for reference fit/predict times per model so we can
    # compute time multipliers.  The reference backend is the first backend
    # tested for each model (reference_backend is None for that row).
    # model -> (fit_time_mean, predict_time_mean, total_time_mean)
    ref_times = {}
    for rec in results:
        if rec.get("reference_backend") is None and rec.get("status") == "ok":
            ref_times[rec["model"]] = (
                rec["fit_time_mean"],
                rec["predict_time_mean"],
                rec["total_time_mean"],
            )

    def _speedup(rec, time_key):
        """Return speedup string like '1.00x' or '2.00x' vs reference.

        Values > 1 mean faster than reference, < 1 mean slower.
        """
        ref = ref_times.get(rec.get("model"))
        if ref is None:
            return "-"
        key_to_idx = {
            "fit_time_mean": 0,
            "predict_time_mean": 1,
            "total_time_mean": 2,
        }
        idx = key_to_idx.get(time_key, 0)
        ref_t = ref[idx]
        cur_t = rec.get(time_key)
        if ref_t is None or cur_t is None or cur_t == 0:
            return "-"
        return f"{ref_t / cur_t:.2f}x"

    # Column definitions: (header, value_func)
    columns = [
        ("Model", lambda r: str(r.get("model", ""))),
        ("Backend", lambda r: str(r.get("backend", ""))),
        ("Status", lambda r: str(r.get("status", ""))),
        ("Fit (mean)", lambda r: f"{r['fit_time_mean']:.4f}s"
         if r.get("fit_time_mean") is not None else "-"),
        ("Fit (std)", lambda r: f"{r['fit_time_std']:.4f}s"
         if r.get("fit_time_std") is not None else "-"),
        ("Fit (x ref)", lambda r: _speedup(r, "fit_time_mean")),
        ("Predict (mean)", lambda r: f"{r['predict_time_mean']:.4f}s"
         if r.get("predict_time_mean") is not None else "-"),
        ("Predict (std)", lambda r: f"{r['predict_time_std']:.4f}s"
         if r.get("predict_time_std") is not None else "-"),
        ("Predict (x ref)", lambda r: _speedup(r, "predict_time_mean")),
        ("Total (mean)", lambda r: f"{r['total_time_mean']:.4f}s"
         if r.get("total_time_mean") is not None else "-"),
        ("Total (std)", lambda r: f"{r['total_time_std']:.4f}s"
         if r.get("total_time_std") is not None else "-"),
        ("Total (x ref)", lambda r: _speedup(r, "total_time_mean")),
        ("vs Ref", lambda r: r.get("reference_backend") or "-"),
        ("Max |diff|", lambda r: f"{r['predictions_max_abs_diff']:.2e}"
         if r.get("predictions_max_abs_diff") is not None else "-"),
        ("Correlation", lambda r: f"{r['predictions_correlation']:.8f}"
         if r.get("predictions_correlation") is not None else "-"),
    ]

    headers = [c[0] for c in columns]
    # Build rows
    rows = []
    for rec in results:
        rows.append([fmt(rec) for _, fmt in columns])

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Print table
    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"

    print()
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print()


def plot_results(results, output_dir, stamp):
    """Plot a grouped bar chart of total computation time by model and backend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot.")
        return

    ok_results = [r for r in results if r.get("status") == "ok"]
    if not ok_results:
        return

    models = []
    for r in ok_results:
        if r["model"] not in models:
            models.append(r["model"])
    backends = []
    for r in ok_results:
        if r["backend"] not in backends:
            backends.append(r["backend"])

    # Build data: model x backend -> total_time_mean
    data = {}
    for r in ok_results:
        data[(r["model"], r["backend"])] = (
            r["total_time_mean"], r["total_time_std"])

    x = np.arange(len(models))
    n_backends = len(backends)
    width = 0.8 / n_backends

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 3), 5))
    for i, backend in enumerate(backends):
        means = [data.get((m, backend), (0, 0))[0] for m in models]
        stds = [data.get((m, backend), (0, 0))[1] for m in models]
        offset = (i - (n_backends - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, label=backend,
               capsize=3)

    ax.set_ylabel("Total time (s)")
    ax.set_title("Benchmark: Total Computation Time (fit + predict)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, f"benchmark_{stamp}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark himalaya backends for speed and correctness."
    )
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--n_features", type=int, default=1000)
    parser.add_argument("--n_targets", type=int, default=30000)
    parser.add_argument("--n_alphas", type=int, default=20)
    parser.add_argument("--alpha_max", type=float, default=20,
                        help="Upper bound for logspace alphas (default: 20).")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n_iter", type=int, default=5)
    parser.add_argument("--n_repetitions", type=int, default=10,
                        help="Repetitions per benchmark (default: 10).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backends",
        type=str,
        default=None,
        help="Comma-separated backends, e.g. 'numpy,torch,torch_mps'. "
        "Default: auto-detect.",
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for result files. Default: "
                        "benchmarks/benchmark_results (relative to script).")
    parser.add_argument(
        "--float64", action="store_true",
        help="Use float64 input instead of the default float32.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick smoke test with small dimensions "
        "(n_samples=200, n_features=100, n_targets=50, n_alphas=5, "
        "n_repetitions=3).",
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Number of untimed warmup iterations for GPU backends "
        "(torch_cuda, torch_mps). Helps reduce variance from JIT compilation. "
        "Default: 1.",
    )
    parser.add_argument(
        "--n_targets_batch", type=int, default=5000,
        help="Size of target batches during cross-validation. Reduces memory "
        "pressure, especially important for GPU backends. Default: 5000.",
    )
    args = parser.parse_args()

    # --fast overrides dimension/repetition defaults (but not explicit flags)
    if args.fast:
        fast_defaults = {
            "n_samples": 200, "n_features": 100, "n_targets": 50,
            "n_alphas": 5, "alpha_max": 3, "n_repetitions": 3,
        }
        for key, val in fast_defaults.items():
            # Only override if the user didn't explicitly set the flag
            if f"--{key}" not in sys.argv:
                setattr(args, key, val)

    # Resolve output directory (default: benchmark_results/ next to this script)
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, "benchmark_results")

    # Resolve backends
    if args.backends is None:
        args.backends = _detect_backends()
    else:
        args.backends = [b.strip() for b in args.backends.split(",")]

    # System info
    system_info = _get_system_info()
    print("System information:")
    for k, v in system_info.items():
        print(f"  {k}: {v}")
    print()

    # Run benchmarks
    results = run_benchmarks(args)

    # Write output with datetime stamp so successive runs don't overwrite
    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"benchmark_{stamp}.json")
    csv_path = os.path.join(args.output_dir, f"benchmark_{stamp}.csv")
    write_json(results, system_info, args, json_path)
    write_csv(results, csv_path)
    plot_results(results, args.output_dir, stamp)

    # Print markdown summary table
    print_summary_table(results)

    print("Done.")


if __name__ == "__main__":
    main()
