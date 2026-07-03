"""
strategy_a_lightning.py
=======================
LIGHTNING AI SIDE of Strategy A.

What this script does:
  1. Runs evalplus.evaluate with GPU energy logging (nvidia-smi)
  2. Parses evalplus outputs into a flat sample schema
  3. Exports passing samples as standalone Python scripts
  4. Bundles everything into a single tar.gz for transfer to RAPL machine
  5. After you copy exec_energy_rapl.csv back, computes final BEP

Run order on Lightning:
  Step 1 — generate + log GPU energy for each method/iteration:
    python strategy_a_lightning.py generate \
        --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
        --method simple --iteration 0

  Step 2 — export all samples for RAPL machine:
    python strategy_a_lightning.py export \
        --samples_dir bep_outputs/samples \
        --out_bundle rapl_bundle.tar.gz

  Step 3 (after copying exec_energy_rapl.csv back from RAPL machine):
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
import tarfile
import argparse
import threading
import subprocess
import tempfile
import statistics
import math
from pathlib import Path
from evalplus.data import get_human_eval_plus, get_mbpp_plus


OUTPUT_ROOT = "bep_outputs"


# ─────────────────────────────────────────────────────────────────────────────
# GPU LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class GpuLogger:
    def __init__(self, csv_path: str, interval_s: float = 0.1):
        self.csv_path   = csv_path
        self.interval_s = interval_s
        self._rows: list[dict] = []
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._t0     = 0.0

    def start(self):
        self._t0 = time.time()
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=5)
        wh = self._integrate()
        self._save()
        wall = time.time() - self._t0
        print(f"  [GPU] wall={wall:.1f}s  samples={len(self._rows)}"
              f"  energy={wh:.4f} Wh ({wh*3600:.1f} J)")
        return wh

    def _poll(self):
        while not self._stop.is_set():
            ts = time.time()
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=index,power.draw,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5
                )
                for line in out.strip().splitlines():
                    p = [x.strip() for x in line.split(",")]
                    if len(p) >= 3:
                        self._rows.append(
                            {"ts": ts, "gpu": p[0],
                             "power_w": p[1], "util": p[2]}
                        )
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def _integrate(self) -> float:
        by_gpu: dict[str, list] = {}
        for r in self._rows:
            by_gpu.setdefault(r["gpu"], []).append(r)
        ws = 0.0
        for rows in by_gpu.values():
            rows.sort(key=lambda x: x["ts"])
            for i in range(1, len(rows)):
                try:
                    p1 = float(rows[i-1]["power_w"])
                    p2 = float(rows[i]["power_w"])
                    dt = rows[i]["ts"] - rows[i-1]["ts"]
                    ws += 0.5 * (p1 + p2) * dt
                except (ValueError, KeyError):
                    pass
        return ws / 3600.0

    def _save(self):
        if not self._rows:
            return
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._rows)


# ─────────────────────────────────────────────────────────────────────────────
# EVALPLUS PARSER  (inline, no dependency on parse_evalplus_outputs.py)
# ─────────────────────────────────────────────────────────────────────────────

def find_evalplus_files(results_dir: str, model: str, dataset: str):
    import glob
    slug = model.replace("/", "--")
    base = Path(results_dir) / dataset
    raws  = glob.glob(str(base / f"{slug}*.raw.jsonl"))
    # evals = glob.glob(str(base / "*.jsonl"))
    # evals = [f for f in evals if not f.endswith(".raw.jsonl")]
    # fallback: any file
    if not raws:  raws  = glob.glob(str(base / "*.raw.jsonl"))
    slug = model.replace("/", "--")

    evals = glob.glob(
        str(base / f"{slug}*_evalperf_results.json")
    )

    if not evals:
        evals = glob.glob(str(base / "*_evalperf_results.json"))
    
    return (
        raws[0] if raws else None,
        evals[0] if evals else None,
    )

def parse_evalplus_to_samples(
    raw_path: str,
    eval_path: str,
    model: str,
    method: str,
    iteration: int,
    e_optim_wh: float,
    dataset: str,
) -> list[dict]:
    # Load eval results
        # EvalPlus codegen does not produce evaluation results
        # Load EvalPerf results
    with open(eval_path) as f:
        eval_data = json.load(f)

    results_map = {}

    for task_id, task in eval_data["eval"].items():
        results_map[task_id] = {
            r["sample_id"]: r["passed"]
            for r in task.get("results", [])
        }
    # Load task metadata (used to export executable test harnesses)
    humaneval_tasks = get_human_eval_plus()
    mbpp_tasks = get_mbpp_plus()
    # Load raw generations
    samples = []
    with open(raw_path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_id    = row.get("task_id", row.get("problem", f"task_{idx}"))
            code       = row.get("solution", row.get("completion", ""))
            sample_idx = row.get("_index", idx)
            sample_id = row["sample_id"]

            passed = results_map.get(task_id, {}).get(sample_id, False)

            # Build executable test harness
            test_code = ""

            if task_id.startswith("HumanEval/"):
                prob = humaneval_tasks[task_id]
                test_code = (
                    prob["test"]
                    + "\n\n"
                    + f'check({prob["entry_point"]})'
                )

            elif task_id.startswith("Mbpp/"):
                prob = mbpp_tasks[task_id]
                test_code = prob["assertion"]

            samples.append({
                "task_id":      f"{dataset.upper()}/{task_id}" if "/" not in task_id else task_id,
                "model":        model,
                "method":       method,
                "iteration":    iteration,
                "sample_idx":   sample_idx,
                "batch_size":   1,
                "batch_id":     sample_idx,
                "is_selected":  True,
                "passed_tests": passed,
                "cpu_instructions": None,
                "e_exec_j":     None,
                "e_optim_wh":   e_optim_wh,
                "_code": code,
                "_test_code": test_code,  
            })
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: GENERATE — run evalplus + log GPU energy
# ─────────────────────────────────────────────────────────────────────────────

def cmd_generate(args):
    Path(OUTPUT_ROOT).mkdir(exist_ok=True)
    samples_dir = Path(OUTPUT_ROOT) / "samples"
    gpu_dir     = Path(OUTPUT_ROOT) / "gpu_logs"
    samples_dir.mkdir(exist_ok=True)
    gpu_dir.mkdir(exist_ok=True)

    method    = args.method
    iteration = args.iteration
    model     = args.model
    dataset   = args.dataset

    gpu_csv = str(gpu_dir / f"gpu_{method}_iter{iteration}.csv")

    print(f"\n{'='*60}")
    print(f"GENERATE: model={model.split('/')[-1]}  "
          f"method={method}  iter={iteration}  dataset={dataset}")
    print(f"{'='*60}")
    print(f"  GPU log → {gpu_csv}")

    evalplus_cmd = (
    f"evalplus.codegen "
    f"{model} "
    f"{dataset} "
    f"--backend openai "
    f"--base-url http://localhost:8000/v1 "
    f"--greedy "
    f"--root {args.results_dir} "
    f"2>&1 | tee {OUTPUT_ROOT}/evalplus_{method}_iter{iteration}.log"
    )

    logger = GpuLogger(gpu_csv)
    logger.start()
    print(f"\n  Running: {evalplus_cmd}\n")
    ret = os.system(evalplus_cmd)
    wh  = logger.stop()

    if ret != 0:
        print(f"  [WARN] evalplus exited with code {ret}")

    # Parse evalplus outputs
    raw_path, eval_path = find_evalplus_files(
        args.results_dir, model, dataset
    )
    if not raw_path or not eval_path:
        print(f"  [ERROR] Could not find evalplus output files in "
              f"{args.results_dir}/{dataset}/")
        print(f"  Expected: {model.replace('/', '--')}*.raw.jsonl")
        return

    print(f"\n  Parsing: {Path(raw_path).name}")
    samples = parse_evalplus_to_samples(
        raw_path, eval_path,
        model=model, method=method,
        iteration=iteration, e_optim_wh=wh,
        dataset=dataset,
    )

    n_pass = sum(1 for s in samples if s["passed_tests"])
    print(f"  Parsed {len(samples)} samples, {n_pass} passed")

    # Save samples JSON
    samples_file = samples_dir / f"samples_{method}_iter{iteration}.json"
    with open(samples_file, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  Saved → {samples_file}")
    print(f"\n  GPU energy: {wh:.4f} Wh ({wh*3600:.1f} J)")
    print(f"\nNext: run for other iterations/methods, then:")
    print(f"  python strategy_a_lightning.py export")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: EXPORT — bundle all samples for RAPL machine
# ─────────────────────────────────────────────────────────────────────────────

def cmd_export(args):
    samples_dir = Path(args.samples_dir)
    out_bundle  = args.out_bundle

    # Collect all samples JSON files
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
              f"Exporting PASSING ONLY: {len(export_set)} "
              f"(skipping {n_failing} failures)")
    else:
        export_set = has_code
        print(f"\n[export] Total samples: {len(all_samples)}  |  "
              f"Exporting ALL: {len(export_set)} "
              f"({n_passing} passed, {n_failing} failed)")
        print(f"[export] Failed samples are tagged 'passed': false in the manifest")
        print(f"[export] so you can later analyze energy wasted on bad optimizations.")
        print(f"[export] Use --passing-only to export only correct samples instead.")

    # Build rapl_package directory in temp
    pkg_dir = Path(OUTPUT_ROOT) / "rapl_package"
    pkg_dir.mkdir(exist_ok=True)
    scripts_dir = pkg_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    manifest = []
    for s in export_set:
        # Sanitise filename
        fname = (
            f"{s['task_id'].replace('/', '_').replace(':', '_')}"
            f"__{s['method']}"
            f"__iter{s['iteration']}"
            f"__sidx{s['sample_idx']}.py"
        )
        import re

        code = s["_code"].strip()

        # Remove Markdown opening fence
        if code.startswith("```python"):
            code = code[len("```python"):].lstrip()
        elif code.startswith("```"):
            code = code[len("```"):].lstrip()

        # Remove stray 'python' line if present
        lines = code.splitlines()
        if lines and lines[0].strip().lower() == "python":
            lines = lines[1:]

        # Remove Markdown closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        code = "\n".join(lines)

        fpath = scripts_dir / fname
        fpath.write_text(
            code.rstrip() + "\n\n" + s.get("_test_code", "").rstrip() + "\n"
        )
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
            # Tag correctness so failed-but-optimized samples can still be
            # analyzed for "energy wasted on incorrect optimizations" (review pt.4)
            "passed":      bool(s.get("passed_tests", False)),
        })

    # Write manifest
    manifest_path = pkg_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Write rapl_runner.py (self-contained, runs on RAPL machine)
    _write_rapl_runner(pkg_dir / "rapl_runner.py")

    # Write README
    _write_rapl_readme(pkg_dir / "README.txt", out_bundle)

    # Bundle into tar.gz
    with tarfile.open(out_bundle, "w:gz") as tar:
        tar.add(pkg_dir, arcname="rapl_package")

    size_mb = Path(out_bundle).stat().st_size / 1e6
    print(f"\n[export] Bundle: {out_bundle}  ({size_mb:.1f} MB)")
    print(f"         Scripts: {len(manifest)}")
    print(f"\n{'='*60}")
    print(f"NEXT STEPS (Strategy A)")
    print(f"{'='*60}")
    print(f"\n1. Download {out_bundle} from Lightning")
    print(f"\n2. On your RAPL machine (Linux with Intel/AMD CPU):")
    print(f"     tar -xzf {out_bundle}")
    print(f"     cd rapl_package")
    print(f"     # Optional: unlock perf if needed")
    print(f"     sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'")
    print(f"     python rapl_runner.py --input scripts/ --output exec_energy_rapl.csv")
    print(f"\n3. Upload exec_energy_rapl.csv back to Lightning")
    print(f"\n4. On Lightning:")
    print(f"     python strategy_a_lightning.py finalize \\")
    print(f"         --exec_csv exec_energy_rapl.csv \\")
    print(f"         --samples_dir {args.samples_dir} \\")
    print(f"         --out bep_final.json")


def _write_rapl_runner(dest: Path):
    dest.write_text('''\
#!/usr/bin/env python3
"""
rapl_runner.py  —  Run this on your RAPL-capable Linux machine.

Requirements:
  - Linux with Intel (2012+) or AMD (Zen+) CPU
  - Python 3.8+  (no extra packages needed)
  - /sys/class/powercap/*/energy_uj  must be readable

If you get "Permission denied":
  sudo sh -c \'echo 1 > /proc/sys/kernel/perf_event_paranoid\'
  OR run with: sudo python rapl_runner.py ...

Usage:
  python rapl_runner.py --input scripts/ --output exec_energy_rapl.csv
  python rapl_runner.py --input scripts/ --output exec_energy_rapl.csv --calibrate 30 --min-wall 1.0

Output:
  exec_energy_rapl.csv  — copy this back to Lightning AI
"""

import os, sys, time, json, csv, subprocess, argparse, statistics
from pathlib import Path


# ── INSTRUCTION COUNTING (perf stat, fix review pt.1/pt.2) ───────────────────

def perf_available() -> bool:
    """Check if `perf stat -e instructions` works without elevated paranoid."""
    try:
        r = subprocess.run(
            ["perf", "stat", "-e", "instructions", "--",
             sys.executable, "-c", "pass"],
            capture_output=True, text=True, timeout=10,
        )
        # perf writes stats to stderr; "instructions" event line confirms access
        return "instructions" in r.stderr and "<not supported>" not in r.stderr \
            and "Permission" not in r.stderr
    except FileNotFoundError:
        return False
    except Exception:
        return False


def run_with_perf_instructions(script_path, timeout=60.0):
    """
    Run the script once under `perf stat -e instructions` and parse the
    real retired-instruction count (NOT wall time). This matches EvalPerf's
    Cirron-based instruction counting more closely than a time proxy.

    Returns (instructions: int | None, wall_s: float, returncode: int)
    """
    cmd = [
        "perf", "stat", "-e", "instructions", "--field-separator", ",",
        sys.executable, str(script_path),
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    t1 = time.perf_counter()

    instructions = None
    for line in res.stderr.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 3 and "instructions" in parts[2]:
            raw = parts[0].strip().replace(",", "")
            try:
                instructions = int(raw)
            except ValueError:
                pass

    return instructions, (t1 - t0), res.returncode


# ── RAPL SYSFS ───────────────────────────────────────────────────────────────

def find_rapl_paths():
    base = Path("/sys/class/powercap")
    if not base.exists():
        return []
    paths = []
    for p in sorted(base.glob("*/energy_uj")):
        name_f = p.parent / "name"
        name   = name_f.read_text().strip() if name_f.exists() else ""
        # Skip sub-package domains (core, uncore, dram) — package already
        # includes them on most Intel CPUs (review pt.8: confirmed reasonable).
        if any(k in name.lower() for k in ("core", "uncore", "dram", "psys")):
            continue
        if p.parent.name.count(":") > 1:   # nested sub-domain
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


def check_live(paths, wait=2.0) -> tuple[bool, float]:
    """Returns (is_live, idle_power_w)."""
    if not paths:
        return False, 0.0
    e1 = read_uj(paths); t1 = time.perf_counter()
    time.sleep(wait)
    e2 = read_uj(paths); t2 = time.perf_counter()
    d  = e2 - e1
    if d < 0: d += 2**32
    w  = (d / 1e6) / (t2 - t1)
    return d > 0, w


# ── CALIBRATION ───────────────────────────────────────────────────────────────

def calibrate(paths, duration=30.0) -> float:
    print(f"[calibrate] idle power measurement ({duration:.0f}s) ...", flush=True)
    e1 = read_uj(paths); t1 = time.perf_counter()
    time.sleep(duration)
    e2 = read_uj(paths); t2 = time.perf_counter()
    d  = max(0.0, (e2 - e1 + (2**32 if e2 < e1 else 0))) / 1e6
    w  = d / (t2 - t1)
    print(f"[calibrate] idle = {w:.3f} W  (delta={d:.2f} J over {t2-t1:.1f}s)")
    return w


# ── SINGLE SAMPLE MEASUREMENT ────────────────────────────────────────────────

def measure_one(script_path, paths, idle_w, use_perf, min_s=1.0, timeout=60.0):
    """
    Executes the script repeatedly until >= min_s total wall time.

    Paper method:
      "we repeated each sample's execution until at least one second
       was elapsed" then subtracted idle power baseline.

    Returns a dict with DIAGNOSTIC fields (review pt.7 — don't silently
    clip to zero, keep raw/idle/marginal separately):
      e_raw_j        : total measured energy before idle subtraction
      e_idle_j       : idle baseline subtracted
      e_exec_j       : marginal energy (raw - idle), CAN be negative
                       (clip only at BEP-computation time, not here)
      cpu_time_ns    : wall-clock time per rep in ns (renamed from the
                       misleading 'cpu_instructions', review pt.1)
      instructions   : REAL retired-instruction count per rep via
                       `perf stat -e instructions`, or None if perf
                       is unavailable (review pt.2)
      n_reps         : repetitions performed
      status         : 'ok' | 'error' | 'timeout'
    """
    n_reps   = 0
    total_s  = 0.0
    instr_samples: list[int] = []

    e_before = read_uj(paths)
    t_start  = time.perf_counter()

    while total_s < min_s:
        if time.perf_counter() - t_start > timeout * 4:
            return _measure_err("timeout", n_reps)

        if use_perf:
            instr, dt, rc = run_with_perf_instructions(script_path, timeout)
            if rc != 0:
                return _measure_err("error", n_reps)
            if instr is not None:
                instr_samples.append(instr)
        else:
            t0  = time.perf_counter()
            res = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True, timeout=timeout,
            )
            t1  = time.perf_counter()
            if res.returncode != 0:
                return _measure_err("error", n_reps)
            dt = t1 - t0

        n_reps  += 1
        total_s += dt

    elapsed  = time.perf_counter() - t_start
    e_after  = read_uj(paths)

    d_uj = e_after - e_before
    if d_uj < 0: d_uj += 2**32          # 32-bit counter wraparound
    e_raw_j  = d_uj / 1e6
    e_idle_j = idle_w * elapsed
    # Diagnostic: do NOT clip here (review pt.7). Clip only when computing BEP.
    e_marginal_j = e_raw_j - e_idle_j

    cpu_time_ns = (total_s / n_reps) * 1e9
    instructions_per_rep = (
        sum(instr_samples) / len(instr_samples) if instr_samples else None
    )

    return {
        "e_raw_j":      e_raw_j / n_reps,
        "e_idle_j":     e_idle_j / n_reps,
        "e_exec_j":     e_marginal_j / n_reps,
        "cpu_time_ns":  cpu_time_ns,
        "instructions": instructions_per_rep,
        "n_reps":       n_reps,
        "status":       "ok",
    }


def _measure_err(status: str, n_reps: int) -> dict:
    return {
        "e_raw_j": None, "e_idle_j": None, "e_exec_j": None,
        "cpu_time_ns": None, "instructions": None,
        "n_reps": n_reps, "status": status,
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",     default="scripts/",
                    help="Directory of generated .py scripts")
    ap.add_argument("--output",    default="exec_energy_rapl.csv")
    ap.add_argument("--calibrate", type=float, default=30.0,
                    help="Idle calibration seconds (paper uses 30)")
    ap.add_argument("--min-wall",  type=float, default=1.0,
                    help="Min wall time per sample in seconds (paper uses 1.0)")
    ap.add_argument("--timeout",   type=float, default=60.0)
    ap.add_argument("--recal-every", type=int, default=50,
                    help="Re-calibrate every N samples")
    ap.add_argument("--recal-minutes", type=float, default=20.0,
                    help="Also re-calibrate if this many minutes have "
                         "elapsed since the last calibration, to account "
                         "for CPU thermal drift (review pt.6)")
    ap.add_argument("--no-perf", action="store_true",
                    help="Skip perf-based instruction counting even if "
                         "available (falls back to wall-time-only timing)")
    args = ap.parse_args()

    # ── Check RAPL ────────────────────────────────────────────────────────────
    print("\\n[rapl_runner] Checking RAPL sysfs ...")
    paths = find_rapl_paths()

    if not paths:
        print("\\n[ERROR] No RAPL sysfs paths found.")
        print("  Check: find /sys/class/powercap -name energy_uj 2>/dev/null")
        print("  You need: Linux + Intel (2012+) or AMD (Zen+) CPU")
        print("  Try:  sudo sh -c \\'echo 1 > /proc/sys/kernel/perf_event_paranoid\\'")
        sys.exit(1)

    print(f"  RAPL paths ({len(paths)}):")
    for p in paths:
        name = (p.parent/"name").read_text().strip() if (p.parent/"name").exists() else "?"
        print(f"    [{name}]  {p}")

    live, idle_test = check_live(paths, wait=2.0)
    if not live:
        print("\\n[ERROR] RAPL counter is not changing (virtualised/blocked).")
        print("  This machine may also be a VM without RAPL passthrough.")
        sys.exit(1)
    print(f"  Live check: OK  (implied idle ≈ {idle_test:.1f} W)")

    # ── Check perf instruction counting (review pt.1/pt.2) ──────────────────
    use_perf = (not args.no_perf) and perf_available()
    if use_perf:
        print(f"  perf stat: available  → recording REAL retired "
              f"instruction counts (matches EvalPerf/Cirron methodology)")
    else:
        reason = "--no-perf set" if args.no_perf else \
            "perf not found or perf_event_paranoid blocks it"
        print(f"  perf stat: unavailable ({reason})")
        print(f"  → 'instructions' column will be empty; "
              f"'cpu_time_ns' (wall time) will be the only timing signal.")
        print(f"  → To enable: sudo sh -c "
              f"'echo 1 > /proc/sys/kernel/perf_event_paranoid'")

    # ── Load manifest ─────────────────────────────────────────────────────────
    input_dir = Path(args.input)
    manifest_path = input_dir.parent / "manifest.json"
    if not manifest_path.exists():
        manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] manifest.json not found near {input_dir}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)
    n_passed = sum(1 for e in manifest if e.get("passed", True))
    n_failed = len(manifest) - n_passed
    print(f"\\n[rapl_runner] {len(manifest)} samples to measure "
          f"({n_passed} passed, {n_failed} failed-but-exported)")

    # ── Initial calibration ───────────────────────────────────────────────────
    idle_w = calibrate(paths, args.calibrate)
    last_calibrate_t = time.perf_counter()

    # ── Measure each sample ───────────────────────────────────────────────────
    results = []
    for i, entry in enumerate(manifest):
        # Re-calibrate periodically by sample count (paper re-calibrates
        # per config) OR by elapsed wall time to catch thermal drift
        # (review pt.6).
        elapsed_min = (time.perf_counter() - last_calibrate_t) / 60.0
        need_recal = (i > 0 and i % args.recal_every == 0) or \
                     (elapsed_min >= args.recal_minutes)
        if need_recal:
            trigger = "sample count" if (i % args.recal_every == 0) else \
                f"{elapsed_min:.1f} min elapsed (thermal drift)"
            print(f"\\n[re-calibrate] at sample {i}  (trigger: {trigger}) ...")
            idle_w = calibrate(paths, min(10.0, args.calibrate))
            last_calibrate_t = time.perf_counter()

        script = input_dir / entry["file"]
        if not script.exists():
            print(f"  [{i+1}/{len(manifest)}] MISSING {entry['file']}")
            continue

        passed_tag = "" if entry.get("passed", True) else "[FAILED-OPT] "
        print(f"  [{i+1:>4}/{len(manifest)}] {passed_tag}"
              f"{entry['task_id']} "
              f"{entry['method']} iter{entry['iteration']} "
              f"sidx{entry['sample_idx']} ...",
              end=" ", flush=True)

        m = measure_one(
            script, paths, idle_w, use_perf, args.min_wall, args.timeout
        )

        if m["status"] == "ok":
            instr_str = f"{m['instructions']:.0f} instr" if m["instructions"] else "no-instr"
            print(f"OK  e={m['e_exec_j']:.5f}J  "
                  f"t={m['cpu_time_ns']:.0f}ns  {instr_str}  "
                  f"reps={m['n_reps']}")
        else:
            print(f"FAIL [{m['status']}]  reps={m['n_reps']}")

        results.append({
            "task_id":          entry["task_id"],
            "model":            entry["model"],
            "method":           entry["method"],
            "iteration":        entry["iteration"],
            "sample_idx":       entry["sample_idx"],
            "passed":           entry.get("passed", True),
            # Diagnostic energy breakdown (review pt.7)
            "e_raw_j":          m["e_raw_j"],
            "e_idle_j":         m["e_idle_j"],
            "e_exec_j":         m["e_exec_j"],      # marginal, can be negative
            # Correctly named timing/instruction fields (review pt.1/pt.2)
            "cpu_time_ns":      m["cpu_time_ns"],
            "instructions":     m["instructions"],   # real perf count, or None
            "n_reps":           m["n_reps"],
            "status":           m["status"],
            "backend":          "rapl_sysfs",
            "instr_backend":    "perf_stat" if use_perf else "unavailable",
        })

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(args.output, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    ok   = sum(1 for r in results if r["status"] == "ok")
    fail = len(results) - ok
    vals = [r["e_exec_j"] for r in results if r.get("e_exec_j") is not None]
    neg  = sum(1 for v in vals if v < 0)

    print(f"\\n[rapl_runner] Done: {ok} OK, {fail} failed")
    if vals:
        print(f"  e_exec_j marginal (J): "
              f"min={min(vals):.4f}  "
              f"median={statistics.median(vals):.4f}  "
              f"max={max(vals):.4f}")
        if neg:
            print(f"  [NOTE] {neg} samples have NEGATIVE marginal energy "
                  f"(execution below idle baseline noise floor). These are "
                  f"preserved as-is in the CSV; clip to 0 only at BEP time.")
    print(f"\\n  Output: {args.output}")
    print(f"  Copy this file back to Lightning AI, then run:")
    print(f"    python strategy_a_lightning.py finalize \\\\")
    print(f"        --exec_csv {args.output} \\\\")
    print(f"        --samples_dir bep_outputs/samples \\\\")
    print(f"        --out bep_final.json")


if __name__ == "__main__":
    main()
''')


def _write_rapl_readme(dest: Path, bundle_name: str):
    dest.write_text(f"""\
RAPL Energy Measurement Package
================================
Generated by: strategy_a_lightning.py (Lightning AI side)
Bundle: {bundle_name}

REQUIREMENTS
------------
- Linux OS (Ubuntu, Debian, Fedora, Arch, etc.)
- Intel CPU (Sandy Bridge 2012+) OR AMD CPU (Zen+ 2017+)
- Python 3.8+  (no extra pip packages needed)
- RAPL readable:  cat /sys/class/powercap/intel-rapl*/energy_uj

If permission denied on RAPL:
  sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
  # OR run rapl_runner.py with sudo

FREE LINUX MACHINES THAT WORK
------------------------------
A) Your local Linux laptop/desktop
   Check RAPL:  find /sys/class/powercap -name energy_uj

B) Hetzner CX22 VPS  (€3.79/mo, bare metal, full RAPL)
   https://www.hetzner.com/cloud
   Create Ubuntu 24.04 server, SSH in, runs rapl_runner.py directly.
   Takes ~1 hour, cancel immediately after → €0.06 total cost.

C) Oracle Cloud Always Free (2 OCPU AMD, usually has RAPL)
   https://www.oracle.com/cloud/free/

D) GitHub Codespaces (sometimes works, 60h free/month)
   Check: cat /sys/class/powercap/intel-rapl/energy_uj

STEPS
-----
1. Extract:
     tar -xzf {bundle_name}
     cd rapl_package

2. Verify RAPL works:
     python rapl_runner.py --calibrate 5 --min-wall 0.1 --input scripts/ --output test.csv
     head test.csv   # should show non-zero e_exec_j values

3. Full measurement:
     python rapl_runner.py --input scripts/ --output exec_energy_rapl.csv
     # Takes roughly: N_scripts × 2s  (1s execution + 1s overhead)
     # For 100 scripts ≈ 3-5 minutes

4. Copy back to Lightning:
     scp exec_energy_rapl.csv <lightning_user>@<lightning_host>:~/evalplus_Zenodo/

5. On Lightning:
     python strategy_a_lightning.py finalize \\
         --exec_csv exec_energy_rapl.csv \\
         --samples_dir bep_outputs/samples \\
         --out bep_final.json

OUTPUT
------
exec_energy_rapl.csv columns:
  task_id, model, method, iteration, sample_idx,
  e_exec_j (Joules), cpu_instructions (ns/rep proxy),
  n_reps, status, backend
""")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: FINALIZE — compute BEP after receiving exec_energy_rapl.csv
# ─────────────────────────────────────────────────────────────────────────────

def cmd_finalize(args):
    from collections import defaultdict

    # ── Load exec energy CSV ──────────────────────────────────────────────────
    # Support both old column name (cpu_instructions) and new names
    # (cpu_time_ns + instructions) for backward compatibility.
    exec_map: dict[tuple, dict] = {}
    with open(args.exec_csv) as f:
        for row in csv.DictReader(f):
            # Accept negative e_exec_j here — only clip at BEP time (fix #7)
            if row.get("e_exec_j") not in (None, "None", ""):
                key = (
                    row["task_id"], row["model"], row["method"],
                    int(row["iteration"]), int(row["sample_idx"]),
                )

                def _float_or_none(v):
                    try:
                        return float(v) if v not in (None, "None", "") else None
                    except (ValueError, TypeError):
                        return None

                exec_map[key] = {
                    "e_exec_j":     _float_or_none(row.get("e_exec_j")),
                    "e_raw_j":      _float_or_none(row.get("e_raw_j")),
                    "e_idle_j":     _float_or_none(row.get("e_idle_j")),
                    # New column name (fix #1): prefer cpu_time_ns, fall back
                    # to old cpu_instructions if reading a legacy CSV
                    "cpu_time_ns":  _float_or_none(
                        row.get("cpu_time_ns") or row.get("cpu_instructions")
                    ),
                    # Real perf instruction count (fix #2); None if unavailable
                    "instructions": _float_or_none(row.get("instructions")),
                    "passed":       row.get("passed", "True").lower() == "true",
                    "backend":      row.get("backend", "rapl_sysfs"),
                    "instr_backend": row.get("instr_backend", "unknown"),
                }

    has_instr  = sum(1 for v in exec_map.values() if v.get("instructions"))
    has_negE   = sum(1 for v in exec_map.values()
                     if v.get("e_exec_j") is not None and v["e_exec_j"] < 0)
    backends   = set(v["instr_backend"] for v in exec_map.values())

    print(f"[finalize] Loaded {len(exec_map)} exec energy entries")
    print(f"  energy backend  : rapl_sysfs")
    print(f"  instr backends  : {backends}")
    print(f"  with real instr : {has_instr}/{len(exec_map)}")
    print(f"  negative e_exec : {has_negE}  "
          f"(kept as-is; clipped only at BEP computation)")

    # ── Load all samples ──────────────────────────────────────────────────────
    samples_dir = Path(args.samples_dir)
    all_samples: list[dict] = []
    for f in sorted(samples_dir.glob("samples_*.json")):
        with open(f) as fh:
            all_samples.extend(json.load(fh))
    print(f"[finalize] Loaded {len(all_samples)} samples from {samples_dir}")

    # ── Baseline validation (fix #3) ─────────────────────────────────────────
    # Confirm that iteration 0 == "original generated code" (not pre-optimized).
    # If iter-0 is supposed to be the baseline but already shows significant
    # improvement over iter-1/2, warn the user — BEP would be meaningless.
    iter0_methods = set(
        s["method"] for s in all_samples if s["iteration"] == 0
    )
    iter_counts = defaultdict(set)
    for s in all_samples:
        iter_counts[s["method"]].add(s["iteration"])

    print(f"\n[finalize] Baseline validation (review pt.3):")
    for method in sorted(iter0_methods):
        iters = sorted(iter_counts[method])
        print(f"  method={method:15s}  iterations present: {iters}")
        if 0 not in iters:
            print(f"  [WARN] method={method} has no iteration=0 samples!")
            print(f"         BEP is UNDEFINED without a true baseline.")
            print(f"         Ensure iteration 0 = plain generation, not "
                  f"an already-optimized pass.")
        elif iters[0] != 0:
            print(f"  [WARN] Lowest iteration for {method} is {iters[0]}, "
                  f"not 0. BEP baseline may be wrong.")
        else:
            print(f"  [OK]  iter=0 present as baseline")

    # ── Attach real energy to samples ─────────────────────────────────────────
    attached = 0
    for s in all_samples:
        key = (
            s["task_id"], s["model"], s["method"],
            int(s["iteration"]), int(s["sample_idx"]),
        )
        if key in exec_map:
            e = exec_map[key]
            s["e_exec_j"]     = e["e_exec_j"]      # marginal (may be negative)
            s["e_raw_j"]      = e["e_raw_j"]
            s["e_idle_j"]     = e["e_idle_j"]
            s["cpu_time_ns"]  = e["cpu_time_ns"]   # correctly named (fix #1)
            s["instructions"] = e["instructions"]  # real count or None (fix #2)
            attached += 1

    print(f"\n[finalize] Attached energy to {attached}/{len(all_samples)} samples")

    # ── Load GPU energy logs → e_optim_wh per (method, iteration) ────────────
    gpu_dir = Path(OUTPUT_ROOT) / "gpu_logs"
    gpu_energy: dict[tuple, float] = {}
    for gpu_csv in sorted(gpu_dir.glob("gpu_*.csv")):
        stem  = gpu_csv.stem           # e.g. gpu_simple_iter0
        parts = stem.split("_")        # ['gpu', 'simple', 'iter0']
        if len(parts) >= 3:
            method_name = parts[1]
            iter_str    = parts[2].replace("iter", "")
            try:
                it = int(iter_str)
            except ValueError:
                continue
            wh = _integrate_gpu_csv(str(gpu_csv))
            gpu_energy[(method_name, it)] = wh
            print(f"  GPU energy: method={method_name} iter={it} "
                  f"→ {wh:.4f} Wh ({wh*3600:.1f} J)")

    for s in all_samples:
        key = (s["method"], int(s["iteration"]))
        if key in gpu_energy:
            s["e_optim_wh"] = gpu_energy[key]

    # ── BEP computation ───────────────────────────────────────────────────────

    def trimmed_mean(vals, pct=0.05):
        arr = sorted(v for v in vals if v is not None and not math.isnan(v))
        if not arr:
            return float("nan")
        cut = max(1, int(len(arr) * pct))
        trimmed = arr[cut:-cut] if len(arr) - 2 * cut > 0 else arr
        return statistics.mean(trimmed) if trimmed else float("nan")

    # Build baseline e_exec per task from iter=0 passing samples.
    # Clip negative values to 0 HERE (first clip point — only for baseline).
    baseline: dict[tuple, list[float]] = defaultdict(list)
    for s in all_samples:
        if (s["iteration"] == 0
                and s.get("passed_tests")
                and s.get("e_exec_j") is not None):
            # Clip to 0 for baseline to avoid a negative baseline energy
            # making BEP meaningless.
            baseline[(s["model"], s["task_id"])].append(
                max(0.0, s["e_exec_j"])
            )
    baseline_mean = {k: statistics.mean(v) for k, v in baseline.items() if v}

    if not baseline_mean:
        print("\n[WARN] No baseline energy values found (iteration=0 with "
              "passed_tests=True and e_exec_j). BEP cannot be computed.")
        print("  Check that iteration=0 samples are passing and have energy attached.")

    # Group by config
    configs: dict[tuple, list] = defaultdict(list)
    for s in all_samples:
        configs[(s["model"], s["method"], s["iteration"])].append(s)

    rows = []
    print(f"\n{'='*85}")
    print(f"{'Model':30s} {'Method':15s} {'Iter':4s}  "
          f"{'Pass@1':7s}  {'E_red':6s}  {'BEP':>12s}  "
          f"{'n_BEP':6s}  {'instr?':6s}")
    print(f"{'='*85}")

    for (model, method, iteration) in sorted(configs):
        group     = configs[(model, method, iteration)]
        e_optim_j = (group[0].get("e_optim_wh") or 0.0) * 3600.0
        n_pass    = sum(1 for s in group if s.get("passed_tests"))
        pass_at_1 = n_pass / len(group) * 100 if group else 0.0
        has_instr_g = any(s.get("instructions") for s in group)

        bep_vals  = []
        ered_vals = []

        for s in group:
            if not (s.get("passed_tests")
                    and s.get("e_exec_j") is not None
                    and s.get("is_selected")):
                continue
            e_orig = baseline_mean.get((model, s["task_id"]))
            if not e_orig:
                continue

            # Clip e_opt to 0 only here at BEP computation time (fix #7).
            # This is the second (and only other) clip point.
            e_opt = max(0.0, s["e_exec_j"])
            ered_vals.append(e_opt / e_orig)

            delta = e_orig - e_opt
            # BEP undefined when no savings (delta <= 0) or no optim cost
            if delta > 0 and e_optim_j > 0:
                bep_vals.append(e_optim_j / delta)

        bep  = trimmed_mean(bep_vals)
        ered = trimmed_mean(ered_vals)

        bep_str   = f"{bep:>12,.0f}" if not math.isnan(bep) else "       undef"
        ered_str  = f"{ered:.3f}"    if not math.isnan(ered) else " N/A"
        instr_str = "yes" if has_instr_g else "no"

        print(f"  {model.split('/')[-1]:30s} {method:15s} {iteration:4d}  "
              f"{pass_at_1:6.1f}%  {ered_str}  {bep_str}  "
              f"{len(bep_vals):6d}  {instr_str:6s}")

        rows.append({
            "model":            model.split("/")[-1],
            "method":           method,
            "iteration":        iteration,
            "pass_at_1_pct":    round(pass_at_1, 2),
            "energy_reduction": None if math.isnan(ered) else round(ered, 4),
            "BEP":              None if math.isnan(bep)  else round(bep),
            "e_optim_wh":       round(group[0].get("e_optim_wh") or 0, 4),
            "e_optim_j":        round(e_optim_j, 2),
            "n_bep_samples":    len(bep_vals),
            "has_real_instructions": has_instr_g,
            "energy_backend":   "rapl_sysfs",
        })

    print(f"{'='*85}")

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[finalize] Saved → {args.out}")
    print(f"\nFeed into compute_metrics.py for full Table II/III:")
    print(f"  python compute_metrics.py --real --input <merged_samples.json>")


def _integrate_gpu_csv(csv_path: str) -> float:
    rows = []
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                rows.append(row)
    except Exception:
        return 0.0
    by_gpu: dict[str, list] = {}
    for r in rows:
        by_gpu.setdefault(r.get("gpu", "0"), []).append(r)
    ws = 0.0
    for gpu_rows in by_gpu.values():
        gpu_rows.sort(key=lambda x: float(x.get("ts", 0)))
        for i in range(1, len(gpu_rows)):
            try:
                p1 = float(gpu_rows[i-1]["power_w"])
                p2 = float(gpu_rows[i]["power_w"])
                dt = float(gpu_rows[i]["ts"]) - float(gpu_rows[i-1]["ts"])
                ws += 0.5 * (p1 + p2) * dt
            except Exception:
                pass
    return ws / 3600.0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Strategy A: GPU generation on Lightning + RAPL on CPU machine"
    )
    sub = ap.add_subparsers(dest="cmd")

    # generate
    g = sub.add_parser("generate",
                       help="Run evalplus + log GPU energy for one method/iteration")
    g.add_argument("--model",       default="Qwen/Qwen2.5-Coder-7B-Instruct")
    g.add_argument("--dataset",     default="humaneval")
    g.add_argument("--method",      default="simple")
    g.add_argument("--iteration",   type=int, default=0)
    g.add_argument("--results_dir", default="evalplus_results")

    # export
    e = sub.add_parser("export",
                       help="Bundle all samples for RAPL machine")
    e.add_argument("--samples_dir", default=f"{OUTPUT_ROOT}/samples")
    e.add_argument("--out_bundle",  default="rapl_bundle.tar.gz")
    e.add_argument("--passing-only", action="store_true",
                   help="Export only passing samples (old behavior). "
                        "Default: export ALL samples (passed + failed) "
                        "so energy wasted on incorrect optimizations can "
                        "also be analyzed.")

    # finalize
    f = sub.add_parser("finalize",
                       help="Compute BEP after receiving exec_energy_rapl.csv")
    f.add_argument("--exec_csv",    required=True)
    f.add_argument("--samples_dir", default=f"{OUTPUT_ROOT}/samples")
    f.add_argument("--out",         default="bep_final.json")

    args = ap.parse_args()

    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "export":
        cmd_export(args)
    elif args.cmd == "finalize":
        cmd_finalize(args)
    else:
        ap.print_help()
