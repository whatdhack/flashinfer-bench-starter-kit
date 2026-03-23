"""
FlashInfer-Bench Local Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks locally.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from scripts.pack_solution import pack_solution


def get_trace_set_path() -> str:
    """Get trace set path from environment variable."""
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return path


def run_benchmark(solution: Solution, config: BenchmarkConfig = None, workload_indices: list = None) -> dict:
    """Run benchmark locally and return results.

    Args:
        solution: Solution to benchmark
        config: Benchmark configuration
        workload_indices: Optional list of workload indices (1-based) to run. If None, runs all workloads.
    """
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    # Filter workloads if specific indices are provided
    if workload_indices:
        filtered_workloads = []
        for idx in workload_indices:
            if idx < 1 or idx > len(workloads):
                raise ValueError(f"Workload index {idx} out of range (1-{len(workloads)})")
            filtered_workloads.append(workloads[idx - 1])  # Convert to 0-based
        print(f"Running {len(filtered_workloads)} of {len(workloads)} workloads")
        workloads = filtered_workloads
    else:
        print(f"Running all {len(workloads)} workloads")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


def main():
    """Pack solution and run benchmark."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run FlashInfer benchmark locally",
        epilog="Examples:\n"
               "  python scripts/run_local.py --list\n"
               "  python scripts/run_local.py --workloads 1 5 10\n"
               "  python scripts/run_local.py -w 1-3 --iterations 10\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--warmup-runs", type=int, default=3,
                       help="Number of warmup iterations (default: 3)")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of iterations per trial (default: 100)")
    parser.add_argument("--num-trials", type=int, default=5,
                       help="Number of trials to run (default: 5)")
    parser.add_argument("--workloads", "-w", nargs="+", metavar="N",
                       help="Specific workload numbers to run (1-based, space-separated). "
                            "Supports ranges like '1-5'. If not specified, runs all workloads.")
    parser.add_argument("--list", "--list-workloads", "-l", action="store_true",
                       help="List all available workloads and exit")
    args = parser.parse_args()

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    # Handle --list flag
    if args.list:
        trace_set_path = get_trace_set_path()
        trace_set = TraceSet.from_path(trace_set_path)
        workloads = trace_set.workloads.get(solution.definition, [])
        print(f"\nAvailable workloads for {solution.definition}:")
        for i, w in enumerate(workloads, 1):
            # Access workload attributes properly
            if hasattr(w, 'workload'):
                workload = w.workload
            else:
                workload = w
            seq_len = workload.axes.get("seq_len", "N/A") if hasattr(workload, 'axes') else "N/A"
            uuid_str = workload.uuid if hasattr(workload, 'uuid') else str(w)
            print(f"  {i:2d}. {uuid_str} (seq_len={seq_len})")
        print(f"\nTotal: {len(workloads)} workloads")
        return

    # Parse workload indices (support ranges like "1-5")
    workload_indices = None
    if args.workloads:
        workload_indices = []
        for item in args.workloads:
            if "-" in item:
                # Parse range like "1-5"
                start, end = item.split("-", 1)
                workload_indices.extend(range(int(start), int(end) + 1))
            else:
                workload_indices.append(int(item))

    print(f"\nRunning benchmark (warmup={args.warmup_runs}, iterations={args.iterations}, trials={args.num_trials})...")
    config = BenchmarkConfig(
        warmup_runs=args.warmup_runs,
        iterations=args.iterations,
        num_trials=args.num_trials
    )
    results = run_benchmark(solution, config, workload_indices=workload_indices)

    if not results:
        print("No results returned!")
        return

    print_results(results)


if __name__ == "__main__":
    main()
