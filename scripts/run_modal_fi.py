"""
FlashInfer-Bench Modal Cloud Benchmark Runner (FlashInfer baseline comparison).

Benchmarks your solution head-to-head against the official FlashInfer baseline
(flashinfer_wrapper_9sdjf3) on B200, using the same tolerances as the contest
evaluator (atol=1, rtol=0.3, required_matched_ratio=0.9).

Both solutions are evaluated against the PyTorch reference for correctness.
Speedup = flashinfer_baseline_latency / your_latency  (matches official scoring).

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench-fi")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# Persists the FlashInfer JIT cache (/root/.cache/flashinfer) across runs so the
# sm_100a CUDA extension (~5-10 min compile) is only built once.
# One-time setup: modal volume create flashinfer-jit-cache
jit_cache_volume = modal.Volume.from_name("flashinfer-jit-cache", create_if_missing=True)
JIT_CACHE_PATH = "/root/.cache/flashinfer"

# CUDA 13.0 devel image is required: the FlashInfer MoE kernel JIT-compiles a
# CUDA extension targeting sm_100a (B200 Blackwell), which needs nvcc 13+.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install("flashinfer-bench", "torch", "triton", "numpy", "helion==1.0.0")
)

# Official MoE evaluation tolerances from EVALUATION.md
_EVAL_CONFIG = dict(atol=1.0, rtol=0.3, required_matched_ratio=0.9)

FI_BASELINE_PATH = (
    PROJECT_ROOT.parent
    / "mlsys26-contest/solutions/baseline/moe"
    / "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    / "flashinfer_wrapper_9sdjf3.json"
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={
        TRACE_SET_PATH: trace_volume,
        JIT_CACHE_PATH: jit_cache_volume,
    },
)
def run_benchmark(
    user_solution: Solution,
    fi_solution: Solution,
    config: BenchmarkConfig = None,
) -> dict:
    """Run both solutions on Modal B200 and return per-workload latencies."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    def_name = user_solution.definition
    if def_name not in trace_set.definitions:
        raise ValueError(f"Definition '{def_name}' not found in trace set")

    definition = trace_set.definitions[def_name]
    workloads = trace_set.workloads.get(def_name, [])
    if not workloads:
        raise ValueError(f"No workloads found for definition '{def_name}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={def_name: definition},
        solutions={def_name: [user_solution, fi_solution]},
        workloads={def_name: workloads},
        traces={def_name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    # Persist any newly compiled FlashInfer JIT artifacts so future runs skip recompilation.
    jit_cache_volume.commit()

    traces = result_trace_set.traces.get(def_name, [])
    # results[workload_uuid][solution_name] = {status, latency_ms, ...}
    results = {}

    for trace in traces:
        if not trace.evaluation:
            continue
        wid = trace.workload.uuid
        sol_name = trace.solution if isinstance(trace.solution, str) else trace.solution.name
        if wid not in results:
            results[wid] = {}

        entry = {"status": trace.evaluation.status.value}
        if trace.evaluation.performance:
            entry["latency_ms"] = trace.evaluation.performance.latency_ms
        if trace.evaluation.correctness:
            entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
            entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
        results[wid][sol_name] = entry

    return results


def print_results(results: dict, user_name: str, fi_name: str):
    """Print per-workload comparison table."""
    speedups = []
    print(f"\n{'Workload':<14} {'Your status':<20} {'Your ms':>9} {'FI ms':>9} {'Speedup':>9}")
    print("-" * 65)
    for wid, sols in sorted(results.items()):
        user = sols.get(user_name, {})
        fi = sols.get(fi_name, {})

        user_status = user.get("status", "N/A")
        user_ms = user.get("latency_ms")
        fi_ms = fi.get("latency_ms")

        speedup_str = "N/A"
        if user_ms and fi_ms:
            sp = fi_ms / user_ms
            speedups.append(sp)
            speedup_str = f"{sp:.3f}x"

        user_ms_str = f"{user_ms:.3f}" if user_ms else "N/A"
        fi_ms_str = f"{fi_ms:.3f}" if fi_ms else "N/A"

        print(
            f"  {wid[:8]}...  {user_status:<20} {user_ms_str:>9} {fi_ms_str:>9} {speedup_str:>9}"
        )

    if speedups:
        avg = sum(speedups) / len(speedups)
        print("-" * 65)
        print(f"  {'Mean speedup vs FlashInfer baseline':>52} {avg:.3f}x")
        print(f"\n  (Official score = {avg:.3f}x — speedup > 1x beats FlashInfer)")


@app.local_entrypoint()
def main(warmup_runs: int = 3, iterations: int = 100, num_trials: int = 5):
    """Pack solution and benchmark it against the FlashInfer baseline on Modal."""
    from scripts.pack_solution import pack_solution

    # Load user solution
    print("Packing solution from source files...")
    solution_path = pack_solution()
    user_solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded user solution: {user_solution.name} ({user_solution.definition})")

    # Load FlashInfer baseline
    if not FI_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"FlashInfer baseline not found at {FI_BASELINE_PATH}\n"
            "Make sure mlsys26-contest is at ../mlsys26-contest relative to this repo."
        )
    fi_solution = Solution.model_validate_json(FI_BASELINE_PATH.read_text())
    print(f"Loaded FlashInfer baseline: {fi_solution.name}")

    config = BenchmarkConfig(
        warmup_runs=warmup_runs,
        iterations=iterations,
        num_trials=num_trials,
        timeout_seconds=1800,  # FlashInfer JIT-compiles sm_100a extension on first run (~5-10 min)
        **_EVAL_CONFIG,
    )

    print(
        f"\nRunning on Modal B200 — official tolerances "
        f"(atol={_EVAL_CONFIG['atol']}, rtol={_EVAL_CONFIG['rtol']}, "
        f"matched_ratio={_EVAL_CONFIG['required_matched_ratio']})"
    )
    print(f"warmup={warmup_runs}, iterations={iterations}, trials={num_trials}\n")

    results = run_benchmark.remote(user_solution, fi_solution, config)

    if not results:
        print("No results returned!")
        return

    print_results(results, user_solution.name, fi_solution.name)
