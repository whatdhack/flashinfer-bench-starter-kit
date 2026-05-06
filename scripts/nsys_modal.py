"""
Run nsys profile of moe_fp8fpX_fused.py on Modal B200.

Usage:
    modal run scripts/nsys_modal.py
    modal run scripts/nsys_modal.py --warmup 10 --iterations 20 --workload-idx 0

Saves the .nsys-rep file locally to profiles/nsys/.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-bench-nsys")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

SOLUTION_DIR = PROJECT_ROOT / "solution" / "triton"
MLSYS_ROOT = Path("/home/sgoswami/ai/mlsys26-contest")
MOE_WORKLOADS_DIR = MLSYS_ROOT / "workloads" / "moe"
MOE_BLOB_DIR = MLSYS_ROOT / "blob" / "workloads" / "moe"

# Base image same as run_modal.py + nsight-systems, with solution + workload files baked in
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-base-ubuntu22.04", add_python="3.12")
    .run_commands(
        "apt-get update -qq && apt-get install -y wget gcc",
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update -qq",
        "apt-get install -y nsight-systems-2025.1.3",
    )
    .pip_install("flashinfer-bench", "torch", "triton", "numpy", "helion==1.0.0", "safetensors")
    .add_local_dir(str(SOLUTION_DIR), "/solution")
    # Workload JSONL at /workloads/traces/moe/ so base_path (parent*3) = /workloads
    # Blob at /workloads/blob/workloads/moe/ to match "./blob/workloads/moe/..." in JSONL
    .add_local_dir(str(MOE_WORKLOADS_DIR), "/workloads/traces/moe")
    .add_local_dir(str(MOE_BLOB_DIR), "/workloads/blob/workloads/moe")
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=600,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_nsys(warmup: int = 5, iterations: int = 20, workload_idx: int = 0, compute_dtype: str = "helion_tf32") -> bytes:
    import subprocess, glob, os, sys

    # Find nsys binary
    candidates = glob.glob("/opt/nvidia/nsight-systems/*/bin/nsys")
    nsys = candidates[0] if candidates else "nsys"
    print(f"nsys: {nsys}")

    r = subprocess.run([nsys, "--version"], capture_output=True, text=True)
    print("nsys version:", r.stdout.strip().splitlines()[0] if r.stdout else r.stderr.strip())

    # Use the baked-in local workload JSONL; base_path is /workloads so that
    # relative paths like ./blob/workloads/moe/... resolve correctly.
    wl_files = sorted(glob.glob("/workloads/traces/moe/*.jsonl"))
    print(f"Found {len(wl_files)} MoE workload file(s)")
    if not wl_files:
        raise RuntimeError("No MoE .jsonl workload files found at /workloads/moe/")

    wl_file = wl_files[workload_idx % len(wl_files)]
    print(f"Using workload: {wl_file}")

    out_path = "/tmp/moe_fp8fpX_fused"
    rep_path = out_path + ".nsys-rep"

    cmd = [
        nsys, "profile",
        "--output", out_path,
        "--trace", "cuda,nvtx,osrt",
        "--force-overwrite", "true",
        "python3", "/solution/moe_fp8fpX_fused.py",
        "--workload", wl_file,
        "--warmup", str(warmup),
        "--iterations", str(iterations),
        "--compute-dtype", compute_dtype,
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True, timeout=540)
    print("nsys returncode:", result.returncode)

    if not os.path.exists(rep_path):
        raise RuntimeError(f"nsys-rep not found at {rep_path}")

    size = os.path.getsize(rep_path)
    print(f"Profile saved: {rep_path} ({size/1024/1024:.1f} MB)")
    with open(rep_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(warmup: int = 5, iterations: int = 20, workload_idx: int = 0, compute_dtype: str = "helion_tf32"):
    from datetime import datetime

    print(f"Running nsys profile on Modal B200 (warmup={warmup}, iters={iterations}, compute_dtype={compute_dtype})...")
    data = run_nsys.remote(warmup=warmup, iterations=iterations, workload_idx=workload_idx, compute_dtype=compute_dtype)

    out_dir = PROJECT_ROOT / "profiles" / "nsys"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"moe_fp8fpX_fused_{compute_dtype}_{ts}.nsys-rep"
    out_path.write_bytes(data)
    print(f"\nProfile saved locally: {out_path} ({len(data)/1024/1024:.1f} MB)")
    print("Open with: nsys-ui " + str(out_path))
