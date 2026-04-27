"""
scripts/benchmark.py — Measure inference latency and throughput.

Runs the full preprocessing + inference pipeline on a batch of synthetic
signals and reports latency percentiles, throughput, model size, and
parameter count. Target: < 500 ms per analysis on CPU (spec requirement).

Usage
-----
    python -m scripts.benchmark                  # uses a synthetic signal
    python -m scripts.benchmark --n 100          # run 100 inferences
    python -m scripts.benchmark --concurrency 4  # parallel threads
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def synth_signal(duration: float = 10.0, fs: int = 500) -> np.ndarray:
    """Build a realistic (12, fs*duration) synthetic ECG."""
    t = np.arange(0, duration, 1 / fs)
    base = np.zeros_like(t)
    for beat_start in np.arange(0.3, duration - 0.3, 1.0):  # 60 BPM
        base += 0.15 * np.exp(-((t - beat_start) ** 2) / 0.04 ** 2)
        base -= 0.10 * np.exp(-((t - (beat_start + 0.15)) ** 2) / 0.01 ** 2)
        base += 1.00 * np.exp(-((t - (beat_start + 0.17)) ** 2) / 0.012 ** 2)
        base -= 0.20 * np.exp(-((t - (beat_start + 0.20)) ** 2) / 0.015 ** 2)
        base += 0.30 * np.exp(-((t - (beat_start + 0.35)) ** 2) / 0.04 ** 2)
    rng = np.random.default_rng(0)
    base += 0.02 * rng.standard_normal(len(t))
    sig = np.tile(base[None, :], (12, 1)) * np.linspace(0.85, 1.15, 12)[:, None]
    return sig.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of runs")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5, help="warmup runs to discard")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    if args.checkpoint:
        os.environ["ECG_CHECKPOINT_PATH"] = args.checkpoint

    from app.model_loader import load_model
    from app.service import get_service, AnalysisService

    print("Loading model...")
    try:
        bundle = load_model()
        print(f"  checkpoint: {bundle.checkpoint_path}")
        print(f"  device:     {bundle.device}")
        counts = bundle.model.parameter_count()
        print(f"  params:     {counts['total']:,} total, {counts['trainable']:,} trainable")
        try:
            import torch
            sd = bundle.model.state_dict()
            size_mb = sum(t.numel() * t.element_size() for t in sd.values()) / (1024 ** 2)
            print(f"  size:       {size_mb:.2f} MB")
        except Exception:
            pass
    except FileNotFoundError as exc:
        print(f"  [!] {exc}")
        print("  benchmark will measure preprocessing only")
        bundle = None

    service: AnalysisService = get_service()
    signal = synth_signal()

    # Warmup
    print(f"\nWarming up ({args.warmup} runs)...")
    for _ in range(args.warmup):
        try:
            service.analyze_array(signal, sample_rate=500)
        except Exception as exc:
            print(f"  warmup failed: {exc}")
            return

    # Benchmark
    print(f"Running {args.n} iterations with concurrency={args.concurrency}...")
    latencies_ms: list[float] = []
    errors = 0

    def one_run():
        t0 = time.perf_counter()
        try:
            service.analyze_array(signal, sample_rate=500)
            return (time.perf_counter() - t0) * 1000
        except Exception as exc:
            return f"ERROR: {exc}"

    t_start = time.perf_counter()
    if args.concurrency <= 1:
        for i in range(args.n):
            r = one_run()
            if isinstance(r, float):
                latencies_ms.append(r)
            else:
                errors += 1
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for r in pool.map(lambda _: one_run(), range(args.n)):
                if isinstance(r, float):
                    latencies_ms.append(r)
                else:
                    errors += 1
    wall_seconds = time.perf_counter() - t_start

    if not latencies_ms:
        print("No successful runs.")
        return

    latencies_ms.sort()
    p50 = statistics.median(latencies_ms)
    p90 = latencies_ms[int(0.90 * len(latencies_ms))]
    p95 = latencies_ms[int(0.95 * len(latencies_ms))]
    p99 = latencies_ms[min(int(0.99 * len(latencies_ms)), len(latencies_ms) - 1)]
    mean = statistics.mean(latencies_ms)

    throughput = args.n / wall_seconds

    print("\n=== Results ===")
    print(f"  Runs:        {args.n} ({errors} errors)")
    print(f"  Wall time:   {wall_seconds:.2f} s")
    print(f"  Throughput:  {throughput:.2f} req/s")
    print(f"  Latency ms:  mean={mean:.1f}  p50={p50:.1f}  p90={p90:.1f}  p95={p95:.1f}  p99={p99:.1f}")
    print(f"  Target:      < 500 ms per analysis")
    status = "OK" if p95 < 500 else "MISS"
    print(f"  [{status}] p95 {'within' if status == 'OK' else 'exceeds'} target")


if __name__ == "__main__":
    main()
