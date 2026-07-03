"""
strategy_a_lightning.py
=======================
LIGHTNING AI SIDE of Strategy A (GPU optimization only).
CPU/RAPL measurement happens on your local hardware (see rapl_runner.py,
written out by `export`, and run on your own Linux/Intel/AMD machine).

Why this differs from the old version
--------------------------------------
`evalplus.evalopti` is NOT a single-iteration tool. Given `--iterations N`,
it internally loops iter=0..N *itself*: iteration 0 generates from scratch,
iteration k>0 loads iteration k-1's `*_evalopti_results.json`, feeds the
passing/profiled samples into the chosen optimizer, and writes a *new*
`*_evalopti_results.json` for iteration k. So there is exactly ONE process
to launch per (model, optimizer) pair — not one per iteration.

This script:
  1. Launches `evalplus.evalopti` ONCE for the full `--iterations` range,
     with a GPU logger (nvidia-smi polling) running for the whole duration.
  2. `evalopti.py` writes a small `*_gpu_phase.json` sidecar the moment each
     iteration's generation/optimization call finishes (before correctness
     checking/profiling even start). We watch for those sidecars and use
     their `gpu_phase_start`/`gpu_phase_end` timestamps to slice the
     continuous GPU power log into one energy segment per iteration —
     GENERATION-ONLY, no idle/correctness/profiling time included. We
     report CUMULATIVE GPU energy per iteration (matching the schema
     compute_metrics.py expects: e_optim_wh is the cumulative cost of
     getting to iteration N).
  3. Parses each iteration's `evalopti_results.json` (the TaskEvalResult /
     CodeEvalResult schema evalopti.py actually produces) into the flat
     sample schema used by export/finalize/compute_metrics.py.
  4. `export` bundles passing (or all) samples as standalone .py scripts +
     manifest + a self-contained `rapl_runner.py` for your local machine.
  5. `finalize` (after you copy `exec_energy_rapl.csv` back) merges real
     CPU energy into the samples and computes BEP.

Run order
---------
  Step 1 (Lightning, GPU):
    python strategy_a_lightning.py generate \
        --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
        --optimizer simple --iterations 4

    (repeat for each optimizer you want: simple, cot, cod,
     self-refine-nl-feedback, self-refine-exec-feedback, llm4effi, eoh,
     simple10 — and for each model)

  Step 2 (Lightning): bundle everything for your local machine
    python strategy_a_lightning.py export \
        --samples_dir bep_outputs/samples \
        --out_bundle rapl_bundle.tar.gz

  Step 3 (your local Linux/Intel/AMD machine):
    tar -xzf rapl_bundle.tar.gz && cd rapl_package
    sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'   # if needed
    python rapl_runner.py --input scripts/ --output exec_energy_rapl.csv

  Step 4 (Lightning, after copying exec_energy_rapl.csv back):
    python strategy_a_lightning.py finalize \
        --exec_csv exec_energy_rapl.csv \
        --samples_dir bep_outputs/samples \
        --out bep_final.json
"""

import os
import sys
import time
import json
import csv
import glob
import math
import tarfile
import argparse
import threading
import subprocess
import statistics
from pathlib import Path
from collections import defaultdict

from evalplus.data import get_human_eval_plus, get_mbpp_plus

OUTPUT_ROOT = "bep_outputs"

# Optimizers whose iteration-0 *is itself* the method (they generate
# their own scratch samples rather than reusing plain codegen). Mirrors
# OptimizerBase.can_generate_samples_from_scratch() in evalopti.py.
SCRATCH_GENERATORS = {"llm4effi", "eoh"}


# ─────────────────────────────────────────────────────────────────────────────
# GPU LOGGER  (runs for the whole evalopti invocation; we slice it afterwards)
# ─────────────────────────────────────────────────────────────────────────────

class GpuLogger:
    def __init__(self, csv_path: str, interval_s: float = 0.5):
        self.csv_path = csv_path
        self.interval_s = interval_s
        self._rows: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self._save()

    def _poll(self):
        while not self._stop.is_set():
            ts = time.time()
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=index,power.draw,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                )
                with self._lock:
                    for line in out.strip().splitlines():
                        p = [x.strip() for x in line.split(",")]
                        if len(p) >= 3:
                            self._rows.append(
                                {"ts": ts, "gpu": p[0], "power_w": p[1], "util": p[2]}
                            )
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def _save(self):
        with self._lock:
            if not self._rows:
                return
            Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
                w.writeheader()
                w.writerows(self._rows)

    def energy_wh_between(self, t_start: float, t_end: float) -> float:
        """Trapezoidal-integrate power over [t_start, t_end], per GPU, summed."""
        with self._lock:
            rows = [r for r in self._rows if t_start <= r["ts"] <= t_end]
        by_gpu: dict[str, list] = {}
        for r in rows:
            by_gpu.setdefault(r["gpu"], []).append(r)
        ws = 0.0
        for gpu_rows in by_gpu.values():
            gpu_rows.sort(key=lambda x: x["ts"])
            for i in range(1, len(gpu_rows)):
                try:
                    p1 = float(gpu_rows[i - 1]["power_w"])
                    p2 = float(gpu_rows[i]["power_w"])
                    dt = gpu_rows[i]["ts"] - gpu_rows[i - 1]["ts"]
                    ws += 0.5 * (p1 + p2) * dt
                except (ValueError, KeyError):
                    pass
        return ws / 3600.0


# ─────────────────────────────────────────────────────────────────────────────
# ITERATION RESULT-FILE / GPU-PHASE-SIDECAR WATCHING
# ─────────────────────────────────────────────────────────────────────────────

def _identifier_for_iteration(model: str, optimizer: str, iteration: int,
                               backend: str, temperature: float) -> str:
    """Mirrors evalplus.codegen.get_result_file_identifier +
    OptimizerBase.get_result_file_identifier (evalopti.py's optimizer.py)."""
    base = model.strip("./").replace("/", "--") + f"_{backend}_temp_{temperature}"
    if iteration == 0 and optimizer not in SCRATCH_GENERATORS:
        return base  # plain codegen baseline
    return base + f"_{optimizer}_{iteration}"


def expected_result_path(results_dir: str, model: str, optimizer: str,
                          iteration: int, backend: str, temperature: float) -> str:
    identifier = _identifier_for_iteration(model, optimizer, iteration, backend, temperature)
    return os.path.join(results_dir, "evalperf", f"{identifier}.jsonl").replace(
        ".jsonl", "_evalopti_results.json"
    )


def expected_gpu_phase_path(results_dir: str, model: str, optimizer: str,
                             iteration: int, backend: str, temperature: float) -> str:
    return expected_result_path(results_dir, model, optimizer, iteration,
                                backend, temperature).replace(
        "_evalopti_results.json", "_gpu_phase.json"
    )


def watch_gpu_phase_files(paths: list[str], stop_event: threading.Event, on_ready: "callable"):
    """Poll for each gpu_phase.json sidecar; these are written atomically in one
    open()+write() call by evalopti.py right after generation/optimization finishes,
    well before correctness/profiling complete — so no size-settling wait is needed."""
    seen = set()
    while not stop_event.is_set() and len(seen) < len(paths):
        for idx, p in enumerate(paths):
            if idx in seen:
                continue
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        phase = json.load(f)
                    seen.add(idx)
                    on_ready(idx, phase)
                except (json.JSONDecodeError, OSError):
                    pass  # not fully written yet, retry next poll
        time.sleep(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# PARSING evalopti's real TaskEvalResult/CodeEvalResult schema into flat samples
# ─────────────────────────────────────────────────────────────────────────────

def _test_harness_for(task_id: str, humaneval_tasks: dict, mbpp_tasks: dict) -> str:
    if task_id.startswith("HumanEval/"):
        prob = humaneval_tasks[task_id]
        return prob["test"] + "\n\n" + f'check({prob["entry_point"]})'
    if task_id.startswith("Mbpp/"):
        prob = mbpp_tasks[task_id]
        return prob["assertion"]
    return ""


def parse_evalopti_results(result_path: str, model: str, method: str,
                           iteration: int, e_optim_wh_cumulative: float) -> list[dict]:
    """Flattens one iteration's evalopti_results.json (dict[task_id] ->
    TaskEvalResult{results: [CodeEvalResult...]}) into the sample schema
    used by export/finalize/compute_metrics.py."""
    with open(result_path) as f:
        data = json.load(f)

    humaneval_tasks = get_human_eval_plus()
    mbpp_tasks = get_mbpp_plus()

    samples = []
    for task_id, task in data.get("eval", {}).items():
        for r in task.get("results", []):
            code = r.get("solution", "") or ""
            samples.append({
                "task_id":          task_id,
                "model":            model,
                "method":           method,
                "iteration":        iteration,
                "sample_idx":       r.get("sample_id") or len(samples),
                "batch_size":       1,
                "batch_id":         r.get("sample_id") or len(samples),
                "is_selected":      True,
                "passed_tests":     bool(r.get("passed", False)),
                "cpu_instructions": r.get("num_cpu_instructions"),
                "cpu_time":         r.get("cpu_time"),
                "dps":              r.get("dps"),
                "dps_norm":         r.get("dps_norm"),
                "e_exec_j":         None,        # filled in by `finalize` from RAPL CSV
                "e_optim_wh":       e_optim_wh_cumulative,
                "_code":            code,
                "_test_code":       _test_harness_for(task_id, humaneval_tasks, mbpp_tasks),
            })
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: GENERATE — run evalopti (all iterations, one process) + slice GPU energy
#         via evalopti.py's gpu_phase.json sidecars (generation-only windows)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_generate(args):
    Path(OUTPUT_ROOT).mkdir(exist_ok=True)
    samples_dir = Path(OUTPUT_ROOT) / "samples"
    gpu_dir = Path(OUTPUT_ROOT) / "gpu_logs"
    samples_dir.mkdir(exist_ok=True)
    gpu_dir.mkdir(exist_ok=True)

    model = args.model
    optimizer = args.optimizer
    iterations = args.iterations
    backend = args.backend
    temperature = args.temperature

    expected_paths = [
        expected_result_path(args.results_dir, model, optimizer, it, backend, temperature)
        for it in range(iterations + 1)
    ]
    expected_phase_paths = [
        expected_gpu_phase_path(args.results_dir, model, optimizer, it, backend, temperature)
        for it in range(iterations + 1)
    ]

    print(f"\n{'='*70}")
    print(f"GENERATE: model={model.split('/')[-1]}  optimizer={optimizer}  "
          f"iterations=0..{iterations}")
    print(f"{'='*70}")
    for p in expected_paths:
        print(f"  expect: {p}")

    gpu_csv = str(gpu_dir / f"gpu_{optimizer}_{model.replace('/', '--')}.csv")
    logger = GpuLogger(gpu_csv)
    logger.start()
    t_run_start = time.time()

    gpu_phases: dict[int, dict] = {}   # iteration -> {"gpu_phase_start", "gpu_phase_end", ...}
    watch_stop = threading.Event()

    def on_ready(idx, phase):
        # Guard against stale sidecars from a prior partial run predating this launch
        if phase["gpu_phase_start"] < t_run_start - 5:
            print(f"  [watch] iteration {idx} gpu_phase.json predates this run "
                  f"(stale from a previous attempt) — ignoring, will re-check for a fresh one")
            return
        gpu_phases[idx] = phase
        print(f"  [watch] iteration {idx} GPU phase: "
              f"{phase['gpu_phase_duration_s']:.1f}s "
              f"@ {time.strftime('%H:%M:%S', time.localtime(phase['gpu_phase_start']))}")

    watcher = threading.Thread(
        target=watch_gpu_phase_files,
        args=(expected_phase_paths, watch_stop, on_ready),
        daemon=True,
    )
    watcher.start()

    evalopti_cmd = [
        sys.executable, "-m", "evalplus.evalopti",
        "--optimizer", optimizer,
        "--iterations", str(iterations),
        "--model", model,
        "--backend", backend,
        "--temperature", str(temperature),
        "--base-url", args.base_url,
        "--root", args.results_dir,
        "--n-samples", str(args.n_samples),
        "--calibration-time", str(args.calibration_time),
    ]
    if args.skip_prev_config_check:
        evalopti_cmd.append("--skip-prev-config-check")

    log_path = Path(OUTPUT_ROOT) / f"evalopti_{optimizer}_{model.replace('/', '--')}.log"
    print(f"\n  Running: {' '.join(evalopti_cmd)}")
    print(f"  Log -> {log_path}\n")

    with open(log_path, "w") as logf:
        ret = subprocess.call(evalopti_cmd, stdout=logf, stderr=subprocess.STDOUT)

    watch_stop.set()
    watcher.join(timeout=5)
    logger.stop()

    if ret != 0:
        print(f"  [WARN] evalopti exited with code {ret} — check {log_path}")

    # Re-check once more for any sidecar the watcher missed (e.g. written during final poll gap)
    for idx, p in enumerate(expected_phase_paths):
        if idx not in gpu_phases and os.path.exists(p):
            with open(p) as f:
                phase = json.load(f)
            if phase["gpu_phase_start"] >= t_run_start - 5:
                gpu_phases[idx] = phase

    if not gpu_phases:
        print("  [ERROR] No gpu_phase.json sidecars were produced. Aborting parse.")
        return

    # Slice GPU energy from the sidecar-defined windows only: iteration k's segment
    # is exactly [gpu_phase_start, gpu_phase_end] for that iteration's generation/
    # optimization call — no idle/correctness/profiling time included. We report
    # the CUMULATIVE energy up to and including iteration k, matching the schema
    # compute_metrics.py expects (e_optim_wh = cumulative Wh).
    cumulative_wh = 0.0
    for idx in sorted(gpu_phases):
        phase = gpu_phases[idx]
        seg_wh = logger.energy_wh_between(phase["gpu_phase_start"], phase["gpu_phase_end"])
        cumulative_wh += seg_wh   # cumulative = sum of GENERATION-ONLY segments, idle time excluded

        result_path = expected_paths[idx]
        if not os.path.exists(result_path):
            # The evalopti subprocess has already finished by this point (we waited
            # on subprocess.call above), so a missing result file here means
            # correctness/profiling for this iteration genuinely hasn't produced
            # output yet in a rare race, or something went wrong upstream — check
            # log_path in that case. We poll indefinitely (with periodic status
            # pings) rather than a fixed timeout, since profiling duration scales
            # with --max-profile/--n-samples and we don't want to bail early.
            wait_start = time.time()
            last_status = wait_start
            print(f"  [WARN] iteration {idx}: gpu_phase.json exists but no evalopti_results.json yet "
                  f"(correctness/profiling still running) — waiting indefinitely")
            while not os.path.exists(result_path):
                time.sleep(2.0)
                now = time.time()
                if now - last_status >= 60:  # status ping every minute, doesn't block
                    print(f"    [waiting] iteration {idx}: still no result file after "
                          f"{now - wait_start:.0f}s ...")
                    last_status = now
            print(f"    iteration {idx}: result file appeared after {time.time() - wait_start:.0f}s")

        print(f"\n  Parsing iteration {idx}: {Path(result_path).name}")
        print(f"    GENERATION-ONLY GPU energy: {seg_wh:.4f} Wh   "
              f"cumulative: {cumulative_wh:.4f} Wh ({cumulative_wh*3600:.1f} J)   "
              f"[phase duration: {phase['gpu_phase_duration_s']:.1f}s]")

        samples = parse_evalopti_results(
            result_path=result_path,
            model=model,
            method=optimizer,
            iteration=idx,
            e_optim_wh_cumulative=cumulative_wh,
        )
        n_pass = sum(1 for s in samples if s["passed_tests"])
        print(f"    parsed {len(samples)} samples, {n_pass} passed")

        out_file = samples_dir / f"samples_{optimizer}_{model.replace('/', '--')}_iter{idx}.json"
        with open(out_file, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"    saved -> {out_file}")

    print(f"\nNext: repeat `generate` for other optimizers/models, then:")
    print(f"  python strategy_a_lightning.py export")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: EXPORT — bundle all samples for your local RAPL machine
# ─────────────────────────────────────────────────────────────────────────────

def cmd_export(args):
    samples_dir = Path(args.samples_dir)
    out_bundle = args.out_bundle

    all_samples: list[dict] = []
    for f in sorted(samples_dir.glob("samples_*.json")):
        with open(f) as fh:
            all_samples.extend(json.load(fh))

    has_code = [s for s in all_samples if s.get("_code", "").strip()]
    n_passing = sum(1 for s in has_code if s.get("passed_tests"))
    n_failing = len(has_code) - n_passing

    if args.passing_only:
        export_set = [s for s in has_code if s.get("passed_tests")]
        print(f"\n[export] Total samples: {len(all_samples)}  |  "
              f"Exporting PASSING ONLY: {len(export_set)} (skipping {n_failing} failures)")
    else:
        export_set = has_code
        print(f"\n[export] Total samples: {len(all_samples)}  |  "
              f"Exporting ALL: {len(export_set)} ({n_passing} passed, {n_failing} failed)")
        print(f"[export] Failed samples are tagged 'passed': false in the manifest "
              f"so you can analyze energy wasted on bad optimizations.")

    pkg_dir = Path(OUTPUT_ROOT) / "rapl_package"
    pkg_dir.mkdir(exist_ok=True)
    scripts_dir = pkg_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    manifest = []
    for s in export_set:
        fname = (
            f"{str(s['task_id']).replace('/', '_').replace(':', '_')}"
            f"__{s['method']}__iter{s['iteration']}__sidx{s['sample_idx']}.py"
        )
        code = s["_code"].strip()
        if code.startswith("```python"):
            code = code[len("```python"):].lstrip()
        elif code.startswith("```"):
            code = code[len("```"):].lstrip()
        lines = code.splitlines()
        if lines and lines[0].strip().lower() == "python":
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

        fpath = scripts_dir / fname
        fpath.write_text(code.rstrip() + "\n\n" + s.get("_test_code", "").rstrip() + "\n")

        manifest.append({
            "file":        fname,
            "task_id":     s["task_id"],
            "model":       s["model"],
            "method":      s["method"],
            "iteration":   s["iteration"],
            "sample_idx":  s["sample_idx"],
            "e_optim_wh":  s.get("e_optim_wh", 0.0),
            "batch_size":  s.get("batch_size", 1),
            "is_selected": s.get("is_selected", True),
            "passed":      bool(s.get("passed_tests", False)),
        })

    (pkg_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_rapl_runner(pkg_dir / "rapl_runner.py")
    _write_rapl_readme(pkg_dir / "README.txt", out_bundle)

    with tarfile.open(out_bundle, "w:gz") as tar:
        tar.add(pkg_dir, arcname="rapl_package")

    size_mb = Path(out_bundle).stat().st_size / 1e6
    print(f"\n[export] Bundle: {out_bundle}  ({size_mb:.1f} MB)  Scripts: {len(manifest)}")
    print(f"\nNEXT STEPS")
    print(f"  1. Copy {out_bundle} to your local Linux (Intel/AMD) machine")
    print(f"  2. tar -xzf {out_bundle} && cd rapl_package")
    print(f"     sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'  # if needed")
    print(f"     python rapl_runner.py --input scripts/ --output exec_energy_rapl.csv")
    print(f"  3. Copy exec_energy_rapl.csv back here, then:")
    print(f"     python strategy_a_lightning.py finalize \\")
    print(f"         --exec_csv exec_energy_rapl.csv \\")
    print(f"         --samples_dir {args.samples_dir} --out bep_final.json")


def _write_rapl_runner(dest: Path):
    dest.write_text('''\
#!/usr/bin/env python3
"""
rapl_runner.py — run on YOUR local Linux machine (Intel 2012+ / AMD Zen+).
No extra pip packages needed (Python 3.8+ stdlib only).

If "Permission denied" on RAPL:
  sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
  (or run this whole script with sudo)

Usage:
  python rapl_runner.py --input scripts/ --output exec_energy_rapl.csv
"""
import os, sys, time, json, csv, subprocess, argparse, statistics
from pathlib import Path


def perf_available() -> bool:
    try:
        r = subprocess.run(
            ["perf", "stat", "-e", "instructions", "--", sys.executable, "-c", "pass"],
            capture_output=True, text=True, timeout=10,
        )
        return "instructions" in r.stderr and "<not supported>" not in r.stderr \\
            and "Permission" not in r.stderr
    except Exception:
        return False


def run_with_perf_instructions(script_path, timeout=60.0):
    cmd = ["perf", "stat", "-e", "instructions", "--field-separator", ",",
           sys.executable, str(script_path)]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    t1 = time.perf_counter()
    instructions = None
    for line in res.stderr.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 3 and "instructions" in parts[2]:
            try:
                instructions = int(parts[0].strip().replace(",", ""))
            except ValueError:
                pass
    return instructions, (t1 - t0), res.returncode


def find_rapl_paths():
    base = Path("/sys/class/powercap")
    if not base.exists():
        return []
    paths = []
    for p in sorted(base.glob("*/energy_uj")):
        name_f = p.parent / "name"
        name = name_f.read_text().strip() if name_f.exists() else ""
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


def read_uj(paths) -> float:
    total = 0
    for p in paths:
        try:
            total += int(p.read_text())
        except Exception:
            pass
    return float(total)


def check_live(paths, wait=2.0):
    if not paths:
        return False, 0.0
    e1 = read_uj(paths); t1 = time.perf_counter()
    time.sleep(wait)
    e2 = read_uj(paths); t2 = time.perf_counter()
    d = e2 - e1
    if d < 0: d += 2**32
    return d > 0, (d / 1e6) / (t2 - t1)


def calibrate(paths, duration=30.0) -> float:
    print(f"[calibrate] idle power ({duration:.0f}s) ...", flush=True)
    e1 = read_uj(paths); t1 = time.perf_counter()
    time.sleep(duration)
    e2 = read_uj(paths); t2 = time.perf_counter()
    d = max(0.0, (e2 - e1 + (2**32 if e2 < e1 else 0))) / 1e6
    w = d / (t2 - t1)
    print(f"[calibrate] idle = {w:.3f} W")
    return w


def measure_one(script_path, paths, idle_w, use_perf, min_s=1.0, timeout=60.0):
    n_reps, total_s = 0, 0.0
    instr_samples = []
    e_before = read_uj(paths)
    t_start = time.perf_counter()
    while total_s < min_s:
        if time.perf_counter() - t_start > timeout * 4:
            return {"e_raw_j": None, "e_idle_j": None, "e_exec_j": None,
                    "cpu_time_ns": None, "instructions": None,
                    "n_reps": n_reps, "status": "timeout"}
        if use_perf:
            instr, dt, rc = run_with_perf_instructions(script_path, timeout)
            if rc != 0:
                return {"e_raw_j": None, "e_idle_j": None, "e_exec_j": None,
                        "cpu_time_ns": None, "instructions": None,
                        "n_reps": n_reps, "status": "error"}
            if instr is not None:
                instr_samples.append(instr)
        else:
            t0 = time.perf_counter()
            res = subprocess.run([sys.executable, str(script_path)],
                                 capture_output=True, timeout=timeout)
            dt = time.perf_counter() - t0
            if res.returncode != 0:
                return {"e_raw_j": None, "e_idle_j": None, "e_exec_j": None,
                        "cpu_time_ns": None, "instructions": None,
                        "n_reps": n_reps, "status": "error"}
        n_reps += 1
        total_s += dt

    elapsed = time.perf_counter() - t_start
    e_after = read_uj(paths)
    d_uj = e_after - e_before
    if d_uj < 0: d_uj += 2**32
    e_raw_j = d_uj / 1e6
    e_idle_j = idle_w * elapsed
    e_marginal_j = e_raw_j - e_idle_j
    cpu_time_ns = (total_s / n_reps) * 1e9
    instructions_per_rep = sum(instr_samples) / len(instr_samples) if instr_samples else None
    return {
        "e_raw_j": e_raw_j / n_reps, "e_idle_j": e_idle_j / n_reps,
        "e_exec_j": e_marginal_j / n_reps, "cpu_time_ns": cpu_time_ns,
        "instructions": instructions_per_rep, "n_reps": n_reps, "status": "ok",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="scripts/")
    ap.add_argument("--output", default="exec_energy_rapl.csv")
    ap.add_argument("--calibrate", type=float, default=30.0)
    ap.add_argument("--min-wall", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--recal-every", type=int, default=50)
    ap.add_argument("--recal-minutes", type=float, default=20.0)
    ap.add_argument("--no-perf", action="store_true")
    args = ap.parse_args()

    print("\\n[rapl_runner] Checking RAPL sysfs ...")
    paths = find_rapl_paths()
    if not paths:
        print("[ERROR] No RAPL sysfs paths found (need Linux + Intel 2012+/AMD Zen+).")
        sys.exit(1)
    live, idle_test = check_live(paths, wait=2.0)
    if not live:
        print("[ERROR] RAPL counter not changing (VM without passthrough?).")
        sys.exit(1)
    print(f"  Live check OK (idle ~{idle_test:.1f} W)")

    use_perf = (not args.no_perf) and perf_available()
    print(f"  perf stat: {'available' if use_perf else 'unavailable (wall time only)'}")

    input_dir = Path(args.input)
    manifest_path = input_dir.parent / "manifest.json"
    if not manifest_path.exists():
        manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] manifest.json not found near {input_dir}")
        sys.exit(1)
    manifest = json.load(open(manifest_path))
    print(f"[rapl_runner] {len(manifest)} samples to measure")

    idle_w = calibrate(paths, args.calibrate)
    last_calibrate_t = time.perf_counter()
    results = []
    for i, entry in enumerate(manifest):
        elapsed_min = (time.perf_counter() - last_calibrate_t) / 60.0
        if (i > 0 and i % args.recal_every == 0) or elapsed_min >= args.recal_minutes:
            idle_w = calibrate(paths, min(10.0, args.calibrate))
            last_calibrate_t = time.perf_counter()

        script = input_dir / entry["file"]
        if not script.exists():
            print(f"  [{i+1}/{len(manifest)}] MISSING {entry['file']}")
            continue
        print(f"  [{i+1:>4}/{len(manifest)}] {entry['task_id']} {entry['method']} "
              f"iter{entry['iteration']} sidx{entry['sample_idx']} ...", end=" ", flush=True)
        m = measure_one(script, paths, idle_w, use_perf, args.min_wall, args.timeout)
        print("OK" if m["status"] == "ok" else f"FAIL [{m['status']}]")
        results.append({
            "task_id": entry["task_id"], "model": entry["model"], "method": entry["method"],
            "iteration": entry["iteration"], "sample_idx": entry["sample_idx"],
            "passed": entry.get("passed", True),
            "e_raw_j": m["e_raw_j"], "e_idle_j": m["e_idle_j"], "e_exec_j": m["e_exec_j"],
            "cpu_time_ns": m["cpu_time_ns"], "instructions": m["instructions"],
            "n_reps": m["n_reps"], "status": m["status"], "backend": "rapl_sysfs",
            "instr_backend": "perf_stat" if use_perf else "unavailable",
        })

    with open(args.output, "w", newline="") as f:
        if results:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\\n[rapl_runner] Done: {ok}/{len(results)} OK. Output: {args.output}")
    print("Copy this file back to Lightning and run `strategy_a_lightning.py finalize`.")


if __name__ == "__main__":
    main()
''')


def _write_rapl_readme(dest: Path, bundle_name: str):
    dest.write_text(f"""\
RAPL Energy Measurement Package
================================
Bundle: {bundle_name}

REQUIREMENTS
------------
- Linux, Intel (2012+) or AMD (Zen+) CPU
- Python 3.8+ (stdlib only)
- cat /sys/class/powercap/intel-rapl*/energy_uj  must work

If permission denied:
  sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'

STEPS
-----
1. tar -xzf {bundle_name} && cd rapl_package
2. Sanity check:
     python rapl_runner.py --calibrate 5 --min-wall 0.1 --input scripts/ --output test.csv
     head test.csv
3. Full run:
     python rapl_runner.py --input scripts/ --output exec_energy_rapl.csv
4. Copy exec_energy_rapl.csv back to Lightning, then:
     python strategy_a_lightning.py finalize \\
         --exec_csv exec_energy_rapl.csv \\
         --samples_dir bep_outputs/samples --out bep_final.json
""")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: FINALIZE — merge real CPU energy + compute BEP
# ─────────────────────────────────────────────────────────────────────────────

def cmd_finalize(args):
    exec_map: dict[tuple, dict] = {}
    with open(args.exec_csv) as f:
        for row in csv.DictReader(f):
            if row.get("e_exec_j") in (None, "None", ""):
                continue
            key = (row["task_id"], row["model"], row["method"],
                  int(row["iteration"]), row["sample_idx"])

            def fnum(v):
                try:
                    return float(v) if v not in (None, "None", "") else None
                except (ValueError, TypeError):
                    return None

            exec_map[key] = {
                "e_exec_j": fnum(row.get("e_exec_j")),
                "e_raw_j": fnum(row.get("e_raw_j")),
                "e_idle_j": fnum(row.get("e_idle_j")),
                "cpu_time_ns": fnum(row.get("cpu_time_ns")),
                "instructions": fnum(row.get("instructions")),
            }
    print(f"[finalize] Loaded {len(exec_map)} exec energy entries")

    samples_dir = Path(args.samples_dir)
    all_samples: list[dict] = []
    for f in sorted(samples_dir.glob("samples_*.json")):
        with open(f) as fh:
            all_samples.extend(json.load(fh))
    print(f"[finalize] Loaded {len(all_samples)} samples")

    attached = 0
    for s in all_samples:
        key = (s["task_id"], s["model"], s["method"], int(s["iteration"]), s["sample_idx"])
        if key in exec_map:
            e = exec_map[key]
            s["e_exec_j"] = e["e_exec_j"]
            s["e_raw_j"] = e["e_raw_j"]
            s["e_idle_j"] = e["e_idle_j"]
            s["cpu_time_ns"] = e["cpu_time_ns"]
            s["instructions"] = e["instructions"]
            attached += 1
    print(f"[finalize] Attached energy to {attached}/{len(all_samples)} samples")

    # Baseline sanity check (iteration 0 must exist per method for BEP to be defined)
    iters_by_method = defaultdict(set)
    for s in all_samples:
        iters_by_method[s["method"]].add(s["iteration"])
    for method, iters in sorted(iters_by_method.items()):
        if 0 not in iters:
            print(f"  [WARN] method={method}: no iteration=0 → BEP undefined for it")

    baseline: dict[tuple, list] = defaultdict(list)
    for s in all_samples:
        if s["iteration"] == 0 and s.get("passed_tests") and s.get("e_exec_j") is not None:
            baseline[(s["model"], s["task_id"])].append(max(0.0, s["e_exec_j"]))
    baseline_mean = {k: statistics.mean(v) for k, v in baseline.items() if v}

    def trimmed_mean(vals, pct=0.05):
        arr = sorted(v for v in vals if v is not None and not math.isnan(v))
        if not arr:
            return float("nan")
        cut = max(1, int(len(arr) * pct))
        trimmed = arr[cut:-cut] if len(arr) - 2 * cut > 0 else arr
        return statistics.mean(trimmed) if trimmed else float("nan")

    configs = defaultdict(list)
    for s in all_samples:
        configs[(s["model"], s["method"], s["iteration"])].append(s)

    rows = []
    print(f"\n{'='*85}")
    print(f"{'Model':30s} {'Method':15s} {'Iter':4s}  {'Pass@1':7s}  {'E_red':6s}  {'BEP':>12s}  {'n':6s}")
    print(f"{'='*85}")
    for (model, method, iteration) in sorted(configs):
        group = configs[(model, method, iteration)]
        e_optim_j = (group[0].get("e_optim_wh") or 0.0) * 3600.0
        n_pass = sum(1 for s in group if s.get("passed_tests"))
        pass_at_1 = n_pass / len(group) * 100 if group else 0.0

        bep_vals, ered_vals = [], []
        for s in group:
            if not (s.get("passed_tests") and s.get("e_exec_j") is not None and s.get("is_selected")):
                continue
            e_orig = baseline_mean.get((model, s["task_id"]))
            if not e_orig:
                continue
            e_opt = max(0.0, s["e_exec_j"])
            ered_vals.append(e_opt / e_orig)
            delta = e_orig - e_opt
            if delta > 0 and e_optim_j > 0:
                bep_vals.append(e_optim_j / delta)

        bep = trimmed_mean(bep_vals)
        ered = trimmed_mean(ered_vals)
        bep_str = f"{bep:>12,.0f}" if not math.isnan(bep) else "       undef"
        ered_str = f"{ered:.3f}" if not math.isnan(ered) else " N/A"
        print(f"  {model.split('/')[-1]:30s} {method:15s} {iteration:4d}  "
              f"{pass_at_1:6.1f}%  {ered_str}  {bep_str}  {len(bep_vals):6d}")

        rows.append({
            "model": model.split("/")[-1], "method": method, "iteration": iteration,
            "pass_at_1_pct": round(pass_at_1, 2),
            "energy_reduction": None if math.isnan(ered) else round(ered, 4),
            "BEP": None if math.isnan(bep) else round(bep),
            "e_optim_wh": round(group[0].get("e_optim_wh") or 0, 4),
            "n_bep_samples": len(bep_vals),
        })
    print(f"{'='*85}")

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[finalize] Saved -> {args.out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Strategy A: GPU optimization on Lightning + RAPL CPU measurement locally"
    )
    sub = ap.add_subparsers(dest="cmd")

    g = sub.add_parser("generate", help="Run evalopti (all iterations, one process) + log GPU energy")
    g.add_argument("--model", required=True)
    g.add_argument("--optimizer", required=True,
                   choices=["simple", "cot", "cod", "self-refine-nl-feedback",
                            "self-refine-exec-feedback", "llm4effi", "eoh", "simple10"])
    g.add_argument("--iterations", type=int, default=4)
    g.add_argument("--n_samples", type=int, default=40)
    g.add_argument("--results_dir", default="evalplus_results")
    g.add_argument("--backend", default="openai")
    g.add_argument("--base_url", default="http://localhost:8000/v1")
    g.add_argument("--temperature", type=float, default=0.2)
    g.add_argument("--calibration_time", type=float, default=30)
    g.add_argument("--skip_prev_config_check", action="store_true")

    e = sub.add_parser("export", help="Bundle all samples for your local RAPL machine")
    e.add_argument("--samples_dir", default=f"{OUTPUT_ROOT}/samples")
    e.add_argument("--out_bundle", default="rapl_bundle.tar.gz")
    e.add_argument("--passing-only", action="store_true")

    f = sub.add_parser("finalize", help="Merge real RAPL energy + compute BEP")
    f.add_argument("--exec_csv", required=True)
    f.add_argument("--samples_dir", default=f"{OUTPUT_ROOT}/samples")
    f.add_argument("--out", default="bep_final.json")

    args = ap.parse_args()
    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "export":
        cmd_export(args)
    elif args.cmd == "finalize":
        cmd_finalize(args)
    else:
        ap.print_help()