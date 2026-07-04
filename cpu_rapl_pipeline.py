"""
cpu_rapl_pipeline.py
=====================
LOCAL / RAPL-CAPABLE MACHINE SIDE.

Takes *_gpu_results.json files produced by gpu_pipeline.py on Lightning
(pure data: solutions + pass/fail, no profiling) and:

  1. Runs REAL instruction-count profiling (Cirron) + DPS/DPS_norm
     scoring, reusing evalplus.evalperf.perf_worker exactly as
     evalopti.py would — just here, where perf_event_paranoid permits it.
  2. Runs REAL CPU execution energy via RAPL sysfs per sample.
  3. Writes an updated results file per iteration with profiled, dps,
     dps_norm, cpu_time, num_cpu_instructions, e_exec_j all filled in.

Requirements on this machine: Linux, Intel(2012+)/AMD(Zen+), RAPL sysfs
readable, perf_event_paranoid low enough for Cirron, evalplus installed.

Usage:
  python cpu_rapl_pipeline.py profile --results-dir gpu_outputs/gpu_only \
      --min-correct 1 --max-profile 40 --calibration-time 30
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import rich

from evalplus.data import get_evalperf_data, get_human_eval_plus, get_mbpp_plus
from evalplus.evalperf import TaskEvalResult, perf_worker, get_max_workers
from evalplus.perf.profile import simple_test_profiler


# ─────────────────────────────────────────────────────────────────────────
# RAPL sysfs
# ─────────────────────────────────────────────────────────────────────────

def find_rapl_paths():
    base = Path("/sys/class/powercap")
    if not base.exists():
        return []
    paths = []
    for p in sorted(base.glob("*/energy_uj")):
        name_file = p.parent / "name"
        name = name_file.read_text().strip() if name_file.exists() else ""
        if any(k in name.lower() for k in ("core", "uncore", "dram", "psys")):
            continue
        if p.parent.name.count(":") > 1:
            continue
        try:
            int(p.read_text())
            paths.append(p)
        except Exception:
            pass
    return paths


def read_rapl_uj(paths) -> float:
    total = 0
    for p in paths:
        try:
            total += int(p.read_text())
        except Exception:
            pass
    return float(total)


def calibrate_idle_watts(rapl_paths, duration_s=30.0) -> float:
    rich.print(f"[calibrate] measuring idle power for {duration_s:.0f}s...")
    e0 = read_rapl_uj(rapl_paths); t0 = time.perf_counter()
    time.sleep(duration_s)
    e1 = read_rapl_uj(rapl_paths); t1 = time.perf_counter()
    d = e1 - e0
    if d < 0:
        d += 2 ** 32
    w = (d / 1e6) / (t1 - t0)
    rich.print(f"[calibrate] idle = {w:.3f} W")
    return w


def measure_execution_energy_j(solution: str, test_harness: str, idle_w: float,
                               rapl_paths, min_wall_s=1.0, timeout_s=60.0) -> Optional[float]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(solution.rstrip() + "\n\n" + test_harness.rstrip() + "\n")
        tmp = f.name
    try:
        n_reps, total_s = 0, 0.0
        e0 = read_rapl_uj(rapl_paths)
        t0 = time.perf_counter()
        while total_s < min_wall_s:
            if time.perf_counter() - t0 > timeout_s * 4:
                return None
            ts = time.perf_counter()
            res = subprocess.run([sys.executable, tmp], capture_output=True, timeout=timeout_s)
            te = time.perf_counter()
            if res.returncode != 0:
                return None
            n_reps += 1
            total_s += (te - ts)
        elapsed = time.perf_counter() - t0
        e1 = read_rapl_uj(rapl_paths)
        d_uj = e1 - e0
        if d_uj < 0:
            d_uj += 2 ** 32
        raw_j = d_uj / 1e6
        idle_j = idle_w * elapsed
        return max(0.0, raw_j - idle_j) / n_reps
    finally:
        Path(tmp).unlink(missing_ok=True)


def _load_test_harnesses(ptasks_all):
    he = get_human_eval_plus()
    mb = get_mbpp_plus()
    harnesses = {}
    for task_id in ptasks_all:
        if task_id in he:
            prob = he[task_id]
            harnesses[task_id] = prob["test"] + "\n\n" + f'check({prob["entry_point"]})'
        elif task_id in mb:
            harnesses[task_id] = mb[task_id]["assertion"]
    return harnesses


# ─────────────────────────────────────────────────────────────────────────
# Profiling one iteration file
# ─────────────────────────────────────────────────────────────────────────

def profile_iteration_file(path: str, max_profile: int, min_correct: int,
                           max_workers: int, idle_w: float, rapl_paths,
                           lazy_evaluation: bool = True):
    with open(path) as f:
        data = json.load(f)

    task_eval_results = {tid: TaskEvalResult(**r) for tid, r in data["eval"].items()}
    ptasks_all = get_evalperf_data()

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for task_id, ptask in ptasks_all.items():
            if task_id not in task_eval_results:
                continue
            n_pass = len([r for r in task_eval_results[task_id].results if r.passed])
            if n_pass < min_correct:
                continue
            futures.append(executor.submit(
                perf_worker, task_id, ptask, task_eval_results[task_id],
                lazy_evaluation, max_profile,
            ))
        for fut in as_completed(futures):
            result = fut.result()
            task_eval_results[result.task_id] = result

    test_harnesses = _load_test_harnesses(ptasks_all)
    n_energy_measured = 0
    for task_id, task in task_eval_results.items():
        for r in task.results:
            if r.passed and r.profiled:
                e_j = measure_execution_energy_j(
                    r.solution, test_harnesses.get(task_id, ""), idle_w, rapl_paths,
                )
                r.e_exec_j = e_j  # dynamically added attribute, fine on non-slotted dataclass
                if e_j is not None:
                    n_energy_measured += 1

    data["eval"] = task_eval_results
    out_path = path.replace("_gpu_results.json", "_rapl_results.json")
    with open(out_path, "w") as f:
        json.dump(data, f, default=vars)
    rich.print(f"[green]Wrote {out_path}[/]  (energy measured for {n_energy_measured} samples)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("profile")
    p.add_argument("--results-dir", required=True,
                   help="dir with *_gpu_results.json copied from Lightning")
    p.add_argument("--min-correct", type=int, default=1)
    p.add_argument("--max-profile", type=int, default=40)
    p.add_argument("--calibration-time", type=float, default=30.0)
    p.add_argument("--nb-workers", type=int, default=None)

    args = ap.parse_args()
    if args.cmd == "profile":
        simple_test_profiler()  # verify Cirron actually works HERE — should not raise
        rapl_paths = find_rapl_paths()
        if not rapl_paths:
            rich.print("[red]No RAPL sysfs found on this machine — cannot measure execution energy.[/]")
        idle_w = calibrate_idle_watts(rapl_paths, args.calibration_time) if rapl_paths else 0.0
        max_workers = get_max_workers(args.nb_workers)

        files = sorted(glob.glob(os.path.join(args.results_dir, "*_gpu_results.json")))
        rich.print(f"Found {len(files)} iteration files to profile")
        for path in files:
            rich.print(f"\n--- {path} ---")
            profile_iteration_file(path, args.max_profile, args.min_correct,
                                   max_workers, idle_w, rapl_paths)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()