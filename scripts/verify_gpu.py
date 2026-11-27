"""
Verification script for GPU acceleration on NVIDIA RTX 50-series (Blackwell).
Tests PyTorch, XGBoost, CatBoost, LightGBM, and related ecosystem libraries.
Prints device name, training times (CPU vs GPU), memory usage, and operational status.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from sklearn.datasets import make_classification


def run_gpu_verification():
    print("=" * 70)
    print("  FRAUDSHIELD GPU & ENVIRONMENT VERIFICATION (ASUS TUF A16 - RTX 5070)")
    print("=" * 70)

    results = []

    # 1. PyTorch & CUDA Check
    print("\n[1/5] Checking PyTorch & CUDA Acceleration...")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        cuda_ver = torch.version.cuda
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        cap = torch.cuda.get_device_capability(0) if cuda_avail else "N/A"

        print(f"  • PyTorch Version: {torch.__version__}")
        print(f"  • CUDA Available: {cuda_avail} (Compiled with CUDA {cuda_ver})")
        print(f"  • GPU Device: {device_name} (Capability: {cap})")
        print(f"  • Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB" if cuda_avail else "")

        if cuda_avail:
            # Benchmark tensor ops on CPU vs GPU
            N = 4000
            # CPU
            t0 = time.perf_counter()
            x_cpu = torch.randn(N, N, device="cpu")
            y_cpu = torch.matmul(x_cpu, x_cpu)
            t_cpu = (time.perf_counter() - t0) * 1000

            # GPU
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x_gpu = torch.randn(N, N, device="cuda")
            y_gpu = torch.matmul(x_gpu, x_gpu)
            torch.cuda.synchronize()
            t_gpu = (time.perf_counter() - t0) * 1000

            speedup = t_cpu / max(t_gpu, 0.001)
            print(f"  • Matrix Multiply (4000x4000): CPU={t_cpu:.1f}ms | GPU={t_gpu:.1f}ms ({speedup:.1f}x speedup)")
            results.append(("PyTorch", "GPU (CUDA)", f"{speedup:.1f}x speedup over CPU", "WORKING"))
        else:
            print("  [!] PyTorch CUDA is not available on this build.")
            results.append(("PyTorch", "CPU only", "CUDA not available", "WARNING"))
    except (RuntimeError, ImportError, AttributeError) as e:
        print(f"  [!] PyTorch error: {e}")
        results.append(("PyTorch", "FAILED", str(e), "ERROR"))

    # Generate synthetic tabular classification benchmark (100,000 rows, 30 features)
    print("\nGenerating benchmark classification dataset (100,000 rows, 30 features, 1% fraud)...")
    X, y = make_classification(
        n_samples=100000,
        n_features=30,
        n_informative=20,
        weights=[0.99, 0.01],
        random_state=42,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    # 2. XGBoost Check
    print("\n[2/5] Checking XGBoost GPU Acceleration...")
    try:
        import xgboost as xgb
        print(f"  • XGBoost Version: {xgb.__version__}")

        # CPU run (100 trees)
        clf_cpu = xgb.XGBClassifier(n_estimators=100, tree_method="hist", device="cpu", random_state=42)
        t0 = time.perf_counter()
        clf_cpu.fit(X, y)
        t_xgb_cpu = (time.perf_counter() - t0) * 1000

        # GPU run (100 trees)
        try:
            clf_gpu = xgb.XGBClassifier(n_estimators=100, tree_method="hist", device="cuda", random_state=42)
            t0 = time.perf_counter()
            clf_gpu.fit(X, y)
            t_xgb_gpu = (time.perf_counter() - t0) * 1000
            speedup_xgb = t_xgb_cpu / max(t_xgb_gpu, 0.001)
            print(f"  • XGBoost 100k Fit (100 trees): CPU={t_xgb_cpu:.1f}ms | GPU={t_xgb_gpu:.1f}ms ({speedup_xgb:.1f}x speedup)")
            results.append(("XGBoost", "GPU (CUDA)", f"Fit in {t_xgb_gpu:.1f}ms ({speedup_xgb:.1f}x speedup)", "WORKING"))
        except (xgb.core.XGBoostError, RuntimeError) as e_gpu:
            print(f"  [!] XGBoost GPU failed: {e_gpu}. Using CPU fallback ({t_xgb_cpu:.1f}ms).")
            results.append(("XGBoost", "CPU fallback", f"Fit in {t_xgb_cpu:.1f}ms (GPU error: {e_gpu})", "PARTIAL"))
    except (ImportError, RuntimeError) as e:
        print(f"  [!] XGBoost not installed or error: {e}")
        results.append(("XGBoost", "NOT INSTALLED", str(e), "MISSING"))

    # 3. CatBoost Check
    print("\n[3/5] Checking CatBoost GPU Acceleration...")
    try:
        import catboost as cb
        print(f"  • CatBoost Version: {cb.__version__}")

        # CPU run (100 trees)
        cb_cpu = cb.CatBoostClassifier(iterations=100, task_type="CPU", verbose=0, random_seed=42)
        t0 = time.perf_counter()
        cb_cpu.fit(X, y)
        t_cb_cpu = (time.perf_counter() - t0) * 1000

        # GPU run (100 trees)
        try:
            cb_gpu = cb.CatBoostClassifier(iterations=100, task_type="GPU", verbose=0, random_seed=42)
            t0 = time.perf_counter()
            cb_gpu.fit(X, y)
            t_cb_gpu = (time.perf_counter() - t0) * 1000
            speedup_cb = t_cb_cpu / max(t_cb_gpu, 0.001)
            print(f"  • CatBoost 100k Fit (100 trees): CPU={t_cb_cpu:.1f}ms | GPU={t_cb_gpu:.1f}ms ({speedup_cb:.1f}x speedup)")
            results.append(("CatBoost", "GPU (CUDA)", f"Fit in {t_cb_gpu:.1f}ms ({speedup_cb:.1f}x speedup)", "WORKING"))
        except (cb.CatBoostError, RuntimeError) as e_gpu:
            print(f"  [!] CatBoost GPU failed: {e_gpu}. Using CPU fallback ({t_cb_cpu:.1f}ms).")
            results.append(("CatBoost", "CPU fallback", f"Fit in {t_cb_cpu:.1f}ms (GPU error: {e_gpu})", "PARTIAL"))
    except (ImportError, RuntimeError) as e:
        print(f"  [!] CatBoost not installed or error: {e}")
        results.append(("CatBoost", "NOT INSTALLED", str(e), "MISSING"))

    # 4. LightGBM (CPU as specified)
    print("\n[4/5] Checking LightGBM (CPU)...")
    try:
        import lightgbm as lgb
        print(f"  • LightGBM Version: {lgb.__version__}")
        lgb_clf = lgb.LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=42, verbose=-1)
        t0 = time.perf_counter()
        lgb_clf.fit(X, y)
        t_lgb = (time.perf_counter() - t0) * 1000
        print(f"  • LightGBM 100k Fit (100 trees, CPU multithreaded): {t_lgb:.1f}ms")
        results.append(("LightGBM", "CPU", f"Fit in {t_lgb:.1f}ms", "WORKING"))
    except (ImportError, RuntimeError) as e:
        print(f"  [!] LightGBM error: {e}")
        results.append(("LightGBM", "NOT INSTALLED", str(e), "MISSING"))

    # 5. Ecosystem Libraries: Optuna, SHAP, NetworkX, Polars, PyG
    print("\n[5/5] Checking Ecosystem Libraries...")
    eco_libs = ["optuna", "shap", "networkx", "polars", "pyarrow", "torch_geometric"]
    for lib in eco_libs:
        try:
            mod = __import__(lib)
            v = getattr(mod, "__version__", "installed")
            print(f"  • {lib}: {v}")
            results.append((lib, "CPU/Ecosystem", f"v{v}", "WORKING"))
        except ImportError:
            print(f"  • {lib}: Not installed (optional/pending)")
            results.append((lib, "Ecosystem", "Not installed", "MISSING"))
        except (RuntimeError, AttributeError, TypeError) as e:
            print(f"  • {lib}: Error ({e})")
            results.append((lib, "Ecosystem", str(e), "ERROR"))

    # Summary Table
    print("\n" + "=" * 70)
    print("  ENVIRONMENT ACCELERATION SUMMARY")
    print("=" * 70)
    print(f"{'Library':<20} | {'Target Device':<15} | {'Details':<22} | {'Status'}")
    print("-" * 70)
    for lib, dev, detail, status in results:
        print(f"{lib:<20} | {dev:<15} | {detail:<22} | {status}")
    print("=" * 70)


if __name__ == "__main__":
    run_gpu_verification()

