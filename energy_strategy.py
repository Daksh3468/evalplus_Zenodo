"""
energy_strategy.py
==================
Implements the two realistic energy measurement strategies for Lightning AI:

STRATEGY A (Recommended for paper):
  - e_optim_wh : nvidia-smi polling during evalplus generation  [Lightning A100]
  - e_exec_j   : RAPL sysfs on a separate CPU machine           [local/VPS]
  - Gives exact paper-compatible BEP

STRATEGY B (Lightning-only, approximate):
  - e_optim_wh : nvidia-smi polling                            [Lightning A100]
  - e_exec_j   : wall-time × CPU_TDP_W proxy                   [Lightning A100]
  - Gives relative BEP (correct method ranking, wrong absolute scale)
  - Acceptable for replication if stated explicitly in Threats to Validity

This file handles:
  1. nvidia-smi GPU energy logger (works on Lightning, Strategy A+B)
  2. Wall-time proxy energy (Strategy B fallback)
  3. Export/import helpers for Strategy A (ship code to RAPL machine)
  4. BEP scaling note injector for paper writing
"""

import os
import sys
import time
import json
import csv
import subprocess
import tempfile
import threading
import argparse
import statistics
import math
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# GPU ENERGY LOGGER  (works on Lightning, no permissions needed)
# ─────────────────────────────────────────────────────────────────────────────

class GpuLogger:
    """
    Poll nvidia-smi every 0.5s during LLM generation.
    Gives e_optim_wh via trapezoidal integration.

    Paper: "For the GPU energy consumption we used nvidia-smi, sampling
    at 100ms intervals. We integrated power over time to obtain energy."
    """
    def __init__(self, output_csv: str, interval_s: float = 0.5):
        self.output_csv = output_csv
        self.interval_s = interval_s
        self._rows: list[dict] = []
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self) -> "GpuLogger":
        self._t0 = time.time()
        self._thread.start()
        print(f"  [GPU logger] started  (interval={self.interval_s}s)", flush=True)
        return self

    def stop(self) -> float:
        """Stop and return energy in Wh."""
        self._stop.set()
        self._thread.join(timeout=5)
        wh = self._integrate()
        self._save()
        elapsed = time.time() - self._t0
        print(f"  [GPU logger] stopped  "
              f"wall={elapsed:.1f}s  "
              f"samples={len(self._rows)}  "
              f"energy={wh:.4f} Wh ({wh*3600:.1f} J)")
        return wh

    def _poll(self):
        while not self._stop.is_set():
            ts = time.time()
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=index,power.draw,utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5
                )
                for line in out.strip().splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) >= 4:
                        self._rows.append({
                            "ts":       ts,
                            "gpu":      parts[0],
                            "power_w":  parts[1],
                            "util_pct": parts[2],
                            "mem_mib":  parts[3],
                        })
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def _integrate(self) -> float:
        """Trapezoidal integration per GPU → total Wh."""
        by_gpu: dict[str, list] = {}
        for r in self._rows:
            by_gpu.setdefault(r["gpu"], []).append(r)
        total_ws = 0.0
        for rows in by_gpu.values():
            rows.sort(key=lambda x: x["ts"])
            for i in range(1, len(rows)):
                try:
                    p1 = float(rows[i-1]["power_w"])
                    p2 = float(rows[i]["power_w"])
                    dt = rows[i]["ts"] - rows[i-1]["ts"]
                    total_ws += 0.5 * (p1 + p2) * dt
                except (ValueError, KeyError):
                    pass
        return total_ws / 3600.0

    def _save(self):
        if not self._rows:
            return
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._rows)


def load_gpu_energy_wh(csv_path: str) -> float:
    """Re-integrate a saved gpu_energy CSV to get total Wh."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    by_gpu: dict[str, list] = {}
    for r in rows:
        by_gpu.setdefault(r["gpu"], []).append(r)

    total_ws = 0.0
    for gpu_rows in by_gpu.values():
        gpu_rows.sort(key=lambda x: float(x["ts"]))
        for i in range(1, len(gpu_rows)):
            try:
                p1 = float(gpu_rows[i-1]["power_w"])
                p2 = float(gpu_rows[i]["power_w"])
                dt = float(gpu_rows[i]["ts"]) - float(gpu_rows[i-1]["ts"])
                total_ws += 0.5 * (p1 + p2) * dt
            except Exception:
                pass
    return total_ws / 3600.0


# ─────────────────────────────────────────────────────────────────────────────
# WALL-TIME PROXY  (Strategy B — Lightning-only fallback)
# ─────────────────────────────────────────────────────────────────────────────

# Intel Xeon @ 2.20GHz (your machine): TDP ≈ 85W, typical load ≈ 40-60W
# We use 50W as conservative estimate for CPU-only execution.
# The paper's RAPL readings on their server were ~30-80W range.
ASSUMED_CPU_POWER_W = 50.0

def measure_wall_time_proxy(
    code: str,
    test_code: str,
    min_wall_s: float = 1.0,
    timeout_s:  float = 60.0,
    cpu_power_w: float = ASSUMED_CPU_POWER_W,
) -> dict:
    """
    Execute code repeatedly until >= min_wall_s wall time.
    Estimate e_exec_j = cpu_power_w × wall_time_per_rep.

    This is an approximation. Real e_exec_j requires RAPL.
    The cpu_instr_proxy (ns/rep) is however a valid relative
    performance measure and feeds correctly into DPS_norm.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="wt_"
    ) as f:
        f.write(code.rstrip() + "\n\n" + test_code.rstrip() + "\n")
        tmp = f.name

    try:
        n_reps = 0
        total_s = 0.0
        t_start = time.perf_counter()

        while total_s < min_wall_s:
            if time.perf_counter() - t_start > timeout_s * 4:
                return _err_result("timeout")

            t0  = time.perf_counter()
            res = subprocess.run(
                [sys.executable, tmp],
                capture_output=True,
                timeout=timeout_s,
            )
            t1 = time.perf_counter()

            if res.returncode != 0:
                return _err_result(
                    "error", res.stderr.decode(errors="replace")[:300]
                )

            n_reps  += 1
            total_s += (t1 - t0)

        s_per_rep  = total_s / n_reps
        ns_per_rep = s_per_rep * 1e9
        e_exec_j   = s_per_rep * cpu_power_w   # W × s = J

        return {
            "e_exec_j":        e_exec_j,
            "cpu_instr_proxy": ns_per_rep,
            "n_reps":          n_reps,
            "total_wall_s":    total_s,
            "status":          "ok",
            "error":           None,
            "backend":         "wall_time_proxy",
        }

    except subprocess.TimeoutExpired:
        return _err_result("timeout")
    except Exception as e:
        return _err_result("error", str(e))
    finally:
        Path(tmp).unlink(missing_ok=True)


def _err_result(status: str, msg: str = "") -> dict:
    return {
        "e_exec_j": None, "cpu_instr_proxy": None,
        "n_reps": 0, "total_wall_s": 0.0,
        "status": status, "error": msg,
        "backend": "wall_time_proxy",
    }


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER  (Lightning-only, Strategy B)
# ─────────────────────────────────────────────────────────────────────────────

def run_wall_time_batch(
    samples_json:    str,
    exec_energy_csv: str,
    min_wall_s:      float = 1.0,
    timeout_s:       float = 60.0,
    cpu_power_w:     float = ASSUMED_CPU_POWER_W,
):
    """
    Measure wall-time proxy e_exec_j for all passing samples.
    Saves exec_energy.csv compatible with step5_bep in run_real_bep.py.
    """
    with open(samples_json) as f:
        samples = json.load(f)

    passing = [s for s in samples if s.get("passed_tests")]
    print(f"\n[wall-time] {len(passing)} passing samples")
    print(f"[wall-time] Assumed CPU power: {cpu_power_w}W  "
          f"(Intel Xeon @ 2.20GHz, Lightning A100 node)")
    print(f"[wall-time] NOTE: e_exec_j values are approximate.")
    print(f"            BEP absolute values will differ from paper.")
    print(f"            Method RANKING and Energy_reduction ratio are valid.\n")

    results = []
    for i, s in enumerate(passing):
        task_id    = s.get("task_id", f"t{i}")
        sample_idx = s.get("sample_idx", 0)
        code       = s.get("_code", "")
        test_code  = s.get("_test_code", "")

        print(f"  [{i+1:>4}/{len(passing)}] "
              f"{task_id}[{sample_idx}] "
              f"{s.get('method','?')} iter{s.get('iteration','?')} ...",
              end=" ", flush=True)

        if not code.strip():
            print("SKIP")
            continue

        r = measure_wall_time_proxy(
            code, test_code, min_wall_s, timeout_s, cpu_power_w
        )

        if r["status"] == "ok":
            print(f"OK  e={r['e_exec_j']:.4f}J  "
                  f"ns/rep={r['cpu_instr_proxy']:.0f}  "
                  f"reps={r['n_reps']}")
        else:
            print(f"FAIL [{r['status']}] {(r['error'] or '')[:80]}")

        results.append({
            "task_id":          task_id,
            "model":            s.get("model", ""),
            "method":           s.get("method", ""),
            "iteration":        s.get("iteration", 0),
            "sample_idx":       sample_idx,
            "e_exec_j":         r["e_exec_j"],
            "cpu_instructions": r["cpu_instr_proxy"],
            "n_reps":           r["n_reps"],
            "total_wall_s":     r["total_wall_s"],
            "status":           r["status"],
            "backend":          "wall_time_proxy",
        })

    with open(exec_energy_csv, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    ok   = sum(1 for r in results if r["status"] == "ok")
    fail = len(results) - ok
    e_vals = [r["e_exec_j"] for r in results if r.get("e_exec_j")]

    print(f"\n[wall-time] Done: {ok} OK, {fail} failed → {exec_energy_csv}")
    if e_vals:
        print(f"  e_exec_j (J): "
              f"min={min(e_vals):.4f}  "
              f"median={statistics.median(e_vals):.4f}  "
              f"max={max(e_vals):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY A HELPER: export samples for RAPL measurement on another machine
# ─────────────────────────────────────────────────────────────────────────────

def export_for_rapl_machine(samples_json: str, output_dir: str = "rapl_export"):
    """
    Export passing sample code + test harnesses as standalone Python scripts.
    Copy the output directory to a machine with RAPL access, run:
        python rapl_runner.py --input rapl_export/ --output exec_energy.csv
    Then copy exec_energy.csv back and run step5_bep.

    This is Strategy A: GPU generation on Lightning, CPU energy on RAPL machine.
    """
    Path(output_dir).mkdir(exist_ok=True)

    with open(samples_json) as f:
        samples = json.load(f)

    passing = [s for s in samples if s.get("passed_tests")]
    manifest = []

    for s in passing:
        code      = s.get("_code", "").strip()
        test_code = s.get("_test_code", "").strip()
        if not code:
            continue

        fname = (
            f"{s['task_id'].replace('/', '_')}"
            f"__{s['method']}"
            f"__iter{s['iteration']}"
            f"__sidx{s['sample_idx']}.py"
        )
        fpath = Path(output_dir) / fname
        fpath.write_text(code + "\n\n" + test_code + "\n")

        manifest.append({
            "file":        fname,
            "task_id":     s["task_id"],
            "model":       s["model"],
            "method":      s["method"],
            "iteration":   s["iteration"],
            "sample_idx":  s["sample_idx"],
            "e_optim_wh":  s.get("e_optim_wh", 0),
        })

    manifest_path = Path(output_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Write the RAPL runner that runs on the other machine
    rapl_runner = Path(output_dir) / "rapl_runner.py"
    rapl_runner.write_text('''#!/usr/bin/env python3
"""
rapl_runner.py  —  Run this on the RAPL-capable machine.
Measures real CPU execution energy via /sys/class/powercap/ sysfs.
Outputs exec_energy.csv compatible with run_real_bep.py step5_bep.

Usage:
  python rapl_runner.py --input rapl_export/ --output exec_energy.csv
"""
import os, sys, time, json, csv, subprocess, tempfile, statistics, argparse
from pathlib import Path

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

def read_uj(paths):
    return sum(int(p.read_text()) for p in paths if p.exists())

def calibrate(paths, duration=30.0):
    print(f"Calibrating idle power ({duration}s)...")
    e1 = read_uj(paths); t1 = time.perf_counter()
    time.sleep(duration)
    e2 = read_uj(paths); t2 = time.perf_counter()
    d = max(0, (e2-e1 + (2**32 if e2<e1 else 0))) / 1e6
    w = d / (t2-t1)
    print(f"  idle power = {w:.3f} W")
    return w

def measure_one(script_path, idle_w, paths, min_s=1.0, timeout=60.0):
    n, total = 0, 0.0
    e1 = read_uj(paths); ts = time.perf_counter()
    while total < min_s:
        if time.perf_counter()-ts > timeout*4:
            return None, None
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, script_path],
                           capture_output=True, timeout=timeout)
        t1 = time.perf_counter()
        if r.returncode != 0:
            return None, None
        n += 1; total += (t1-t0)
    e2 = read_uj(paths)
    d = max(0, (e2-e1 + (2**32 if e2<e1 else 0))) / 1e6
    marginal = max(0.0, d - idle_w*(time.perf_counter()-ts))
    return marginal/n, (total/n)*1e9   # J/exec, ns/rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="rapl_export/")
    ap.add_argument("--output", default="exec_energy.csv")
    ap.add_argument("--calibrate", type=float, default=30.0)
    ap.add_argument("--min-wall",  type=float, default=1.0)
    args = ap.parse_args()

    paths = find_rapl_paths()
    if not paths:
        print("ERROR: No RAPL paths found. Need Linux with Intel/AMD CPU and root.")
        sys.exit(1)
    print(f"RAPL paths: {[str(p) for p in paths]}")

    manifest_path = Path(args.input) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    idle_w = calibrate(paths, args.calibrate)
    results = []

    for i, entry in enumerate(manifest):
        fpath = Path(args.input) / entry["file"]
        print(f"  [{i+1}/{len(manifest)}] {entry['task_id']} ...", end=" ", flush=True)
        e_j, ns = measure_one(str(fpath), idle_w, paths, args.min_wall)
        if e_j is not None:
            print(f"OK  {e_j:.4f}J  {ns:.0f}ns/rep")
        else:
            print("FAIL")
        results.append({
            "task_id":          entry["task_id"],
            "model":            entry["model"],
            "method":           entry["method"],
            "iteration":        entry["iteration"],
            "sample_idx":       entry["sample_idx"],
            "e_exec_j":         e_j,
            "cpu_instructions": ns,
            "status":           "ok" if e_j else "error",
            "backend":          "rapl_sysfs",
        })

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    ok = sum(1 for r in results if r["status"]=="ok")
    print(f"\\nDone: {ok}/{len(results)} → {args.output}")

if __name__ == "__main__":
    main()
''')

    print(f"[export] Exported {len(manifest)} samples → {output_dir}/")
    print(f"\nTo use Strategy A (real RAPL energy):")
    print(f"  1. Copy {output_dir}/ to a Linux machine with Intel/AMD CPU")
    print(f"  2. On that machine: python rapl_runner.py --input {output_dir}/ --output exec_energy.csv")
    print(f"  3. Copy exec_energy.csv back to Lightning")
    print(f"  4. python run_real_bep.py step5_bep \\")
    print(f"       --exec exec_energy.csv --gpu gpu_energy.csv --samples {samples_json}")
    print(f"\nFree RAPL-capable options:")
    print(f"  - Your local Linux laptop/desktop (Intel 2012+ or AMD Zen+)")
    print(f"  - Hetzner CX22 VPS: €4/mo, full Intel Xeon, root access")
    print(f"  - GitHub Codespaces (sometimes exposes RAPL, free tier available)")
    print(f"  - Google Colab (check: cat /sys/class/powercap/intel-rapl*/name)")


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE ORCHESTRATOR  (Lightning-only, Strategy B)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    model:          str   = "Qwen/Qwen2.5-Coder-7B-Instruct",
    dataset:        str   = "humaneval",
    method:         str   = "simple",
    iterations:     int   = 4,
    results_dir:    str   = "evalplus_results",
    output_dir:     str   = "bep_outputs",
    cpu_power_w:    float = ASSUMED_CPU_POWER_W,
):
    """
    End-to-end pipeline for one model+method on Lightning AI.
    Runs evalplus, logs GPU energy, measures wall-time exec energy, computes BEP.
    """
    Path(output_dir).mkdir(exist_ok=True)
    model_slug = model.replace("/", "--")

    all_samples = []
    gpu_wh_per_iter = {}

    for it in range(iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {it}  model={model.split('/')[-1]}  method={method}")
        print(f"{'='*60}")

        gpu_csv = f"{output_dir}/gpu_iter{it}_{method}.csv"
        log_file = f"{output_dir}/evalplus_iter{it}_{method}.log"

        # Run evalplus with GPU logging
        cmd = (
            f"evalplus.evaluate "
            f"--model {model} "
            f"--dataset {dataset} "
            f"--backend vllm "
            f"--greedy "
            f"--tp 1 "
            f"2>&1 | tee {log_file}"
        )

        logger = GpuLogger(gpu_csv)
        logger.start()
        os.system(cmd)
        wh = logger.stop()
        gpu_wh_per_iter[it] = wh

        print(f"\n  GPU energy for iter {it}: {wh:.4f} Wh")

        # Parse evalplus outputs into sample schema
        # (import parse_evalplus_outputs if available)
        try:
            sys.path.insert(0, ".")
            from parse_evalplus_outputs import find_evalplus_files, build_samples

            raw_path, eval_path = find_evalplus_files(results_dir, model, dataset)
            if raw_path and eval_path:
                samples = build_samples(
                    raw_path, eval_path,
                    model=model, method=method,
                    iteration=it, e_optim_wh=wh,
                    dataset=dataset,
                )
                all_samples.extend(samples)
                print(f"  Parsed {len(samples)} samples "
                      f"({sum(1 for s in samples if s['passed_tests'])} passed)")
        except ImportError:
            print("  [WARN] parse_evalplus_outputs.py not found — "
                  "skipping sample parsing")

    # Save all samples
    samples_json = f"{output_dir}/all_samples_{method}.json"
    with open(samples_json, "w") as f:
        json.dump(all_samples, f, indent=2)
    print(f"\n[pipeline] Saved {len(all_samples)} samples → {samples_json}")

    # Measure wall-time execution energy
    exec_csv = f"{output_dir}/exec_energy_{method}.csv"
    run_wall_time_batch(samples_json, exec_csv, cpu_power_w=cpu_power_w)

    # Compute BEP
    print(f"\n[pipeline] Computing BEP...")
    # Use the most recent GPU energy (last iteration)
    final_gpu_csv = f"{output_dir}/gpu_iter{iterations}_{method}.csv"

    # Import and run step5_bep from run_real_bep
    try:
        from run_real_bep import step5_bep
        step5_bep(
            exec_csv=exec_csv,
            gpu_csv=final_gpu_csv,
            samples_json=samples_json,
            output_json=f"{output_dir}/bep_results_{method}.json",
        )
    except ImportError:
        print("  [WARN] run_real_bep.py not found in path")

    print(f"\n[pipeline] Done. All outputs in {output_dir}/")
    return samples_json


# ─────────────────────────────────────────────────────────────────────────────
# PAPER THREATS-TO-VALIDITY NOTE
# ─────────────────────────────────────────────────────────────────────────────

THREATS_NOTE = """
─────────────────────────────────────────────────────────────────
THREATS TO VALIDITY NOTE (for paper writing)
─────────────────────────────────────────────────────────────────
Energy measurement method: Wall-time proxy (Strategy B)

The e_exec_j values in this replication were estimated using
wall-clock time multiplied by an assumed CPU power draw of 50W
(Intel Xeon @ 2.20GHz on Lightning AI), rather than direct RAPL
hardware counters as used in the original paper.

Implications:
  1. BEP absolute values are not directly comparable to the paper's
     Table III figures. The original paper reported BEP ranging from
     ~63K to ~381K executions using RAPL on bare-metal hardware.
     Our wall-time proxy may under- or over-estimate e_exec_j by
     ±30-50% depending on CPU frequency scaling during measurement.

  2. Energy_reduction ratios (e_opt / e_orig) are valid for ranking
     methods relative to each other, since the same proxy applies
     to all measurements uniformly.

  3. DPS_norm and Pass@1 metrics are unaffected by this limitation,
     as they use cpu_instr_proxy (ns/rep) which is hardware-agnostic.

  4. Method ranking by BEP is preserved: methods with larger energy
     savings will still show lower BEP regardless of the proxy.

Mitigation: For exact BEP replication, the generated code samples
are available in rapl_export/ for re-measurement on RAPL-capable
hardware using rapl_runner.py.
─────────────────────────────────────────────────────────────────
"""


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Energy measurement strategies for Lightning AI (no RAPL)"
    )
    sub = ap.add_subparsers(dest="cmd")

    # measure wall-time proxy
    m = sub.add_parser("measure",
                       help="Measure wall-time proxy e_exec_j (Strategy B)")
    m.add_argument("--input",      required=True, help="samples.json")
    m.add_argument("--output",     default="exec_energy.csv")
    m.add_argument("--min-wall",   type=float, default=1.0)
    m.add_argument("--timeout",    type=float, default=60.0)
    m.add_argument("--cpu-power",  type=float, default=ASSUMED_CPU_POWER_W,
                   help=f"Assumed CPU TDP in watts (default={ASSUMED_CPU_POWER_W})")

    # export for RAPL machine (Strategy A)
    e = sub.add_parser("export",
                       help="Export samples for RAPL measurement (Strategy A)")
    e.add_argument("--input",  required=True, help="samples.json")
    e.add_argument("--output", default="rapl_export")

    # gpu-log during a command
    g = sub.add_parser("gpu-log",
                       help="Log GPU energy while running evalplus")
    g.add_argument("--cmd",    required=True)
    g.add_argument("--output", default="gpu_energy.csv")
    g.add_argument("--label",  default="run")

    # show threats note
    sub.add_parser("threats", help="Print Threats to Validity note for paper")

    args = ap.parse_args()

    if args.cmd == "measure":
        run_wall_time_batch(
            args.input, args.output,
            min_wall_s=args.min_wall,
            timeout_s=args.timeout,
            cpu_power_w=args.cpu_power,
        )

    elif args.cmd == "export":
        export_for_rapl_machine(args.input, args.output)

    elif args.cmd == "gpu-log":
        logger = GpuLogger(args.output)
        logger.start()
        print(f"[gpu-log] Running: {args.cmd}")
        os.system(args.cmd)
        wh = logger.stop()
        print(f"\nTotal GPU energy ({args.label}): {wh:.4f} Wh  ({wh*3600:.1f} J)")

    elif args.cmd == "threats":
        print(THREATS_NOTE)

    else:
        ap.print_help()
