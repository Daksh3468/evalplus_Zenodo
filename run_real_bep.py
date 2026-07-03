"""
run_real_bep.py
===============
Complete pipeline to get REAL BEP energy numbers on Lightning AI.

Your setup:
  - Intel Xeon @ 2.20GHz  (RAPL sysfs at /sys/class/powercap/)
  - perf_event_paranoid = 4  (perf CLI blocked, but sysfs MAY be readable)
  - NVIDIA A100 80GB  (nvidia-smi works for e_optim_wh)

Handles 3 scenarios automatically:
  A) RAPL sysfs readable  → real e_exec_j in Joules (exact paper method)
  B) RAPL returns zeros   → wall-time proxy (scaled to plausible Joules)
  C) RAPL unreadable      → wall-time proxy with warning

Run order:
  python run_real_bep.py step1_check
  python run_real_bep.py step2_calibrate
  python run_real_bep.py step3_measure --input samples.json
  python run_real_bep.py step4_gpu_log --cmd "evalplus.evaluate ..."
  python run_real_bep.py step5_bep     --exec exec_energy.csv --gpu gpu_energy.csv --samples samples.json
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
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# RAPL SYSFS  — with virtualisation detection
# ─────────────────────────────────────────────────────────────────────────────

def find_rapl_paths() -> list[Path]:
    base = Path("/sys/class/powercap")
    if not base.exists():
        return []
    paths = []
    for p in sorted(base.glob("*/energy_uj")):
        name_file = p.parent / "name"
        name = name_file.read_text().strip() if name_file.exists() else ""
        # Skip sub-package domains to avoid double-counting
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


def read_rapl_uj(paths: list[Path]) -> float:
    total = 0
    for p in paths:
        try:
            total += int(p.read_text())
        except Exception:
            pass
    return float(total)


def check_rapl_live(paths: list[Path], wait_s: float = 2.0) -> dict:
    """
    Returns dict with:
      readable    : bool  (paths exist and return integers)
      live        : bool  (counter actually changes over wait_s seconds)
      power_w     : float (implied idle power W)
      virtualised : bool  (counter exists but never changes — VM issue)
    """
    if not paths:
        return {"readable": False, "live": False, "power_w": 0.0, "virtualised": False}

    e1 = read_rapl_uj(paths)
    t1 = time.perf_counter()
    time.sleep(wait_s)
    e2 = read_rapl_uj(paths)
    t2 = time.perf_counter()

    delta = e2 - e1
    if delta < 0:
        delta += 2**32   # wraparound

    elapsed  = t2 - t1
    power_w  = (delta / 1e6) / elapsed   # µJ → J → W

    live         = delta > 0
    virtualised  = (e1 > 0 and delta == 0)   # counter exists but frozen

    return {
        "readable":    True,
        "live":        live,
        "power_w":     power_w,
        "virtualised": virtualised,
        "delta_j":     delta / 1e6,
        "elapsed_s":   elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CHECK
# ─────────────────────────────────────────────────────────────────────────────

def step1_check():
    print("=" * 60)
    print("STEP 1: Energy measurement availability check")
    print("=" * 60)

    # RAPL paths
    paths = find_rapl_paths()
    print(f"\nRAPL sysfs paths found: {len(paths)}")
    for p in paths:
        name = (p.parent / "name").read_text().strip() if (p.parent / "name").exists() else "?"
        val  = p.read_text().strip()
        print(f"  [{name:20s}]  {val} µJ  →  {p}")

    if not paths:
        print("\n[RESULT] No RAPL paths found.")
        print("  → Will use WALL-TIME PROXY for e_exec_j (approximate BEP).")
        return "wall_time"

    print(f"\nChecking if counter is live (2s test) ...")
    status = check_rapl_live(paths, wait_s=2.0)

    print(f"  delta_j     = {status['delta_j']:.4f} J")
    print(f"  power_w     = {status['power_w']:.2f} W")
    print(f"  live        = {status['live']}")
    print(f"  virtualised = {status['virtualised']}")

    if status["live"] and 2.0 < status["power_w"] < 500:
        print("\n[RESULT] ✓ RAPL sysfs is LIVE and readable.")
        print(f"  Idle power ≈ {status['power_w']:.1f} W")
        print("  → Will use RAPL SYSFS for real e_exec_j measurements.")
        return "rapl_sysfs"
    elif status["virtualised"] or not status["live"]:
        print("\n[RESULT] RAPL counter exists but is FROZEN (virtualised/blocked).")
        print("  Lightning AI likely intercepts sysfs reads and returns a fixed value.")
        print("  → Will use WALL-TIME PROXY for e_exec_j (approximate BEP).")
        print("\n  To get real RAPL readings you need one of:")
        print("  A) A bare-metal Linux machine (local or cheap VPS like Hetzner CX22)")
        print("  B) A VM with RAPL passthrough enabled (rare in cloud)")
        print("  C) Use nvidia-smi GPU energy as the sole energy proxy (GPU-only BEP)")
        return "wall_time"
    else:
        print(f"\n[RESULT] RAPL readable but power={status['power_w']:.2f}W looks odd.")
        print("  Proceeding with RAPL sysfs — verify results make sense.")
        return "rapl_sysfs"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CALIBRATE IDLE
# ─────────────────────────────────────────────────────────────────────────────

def step2_calibrate(backend: str, duration_s: float = 30.0) -> float:
    print(f"\n[calibrate] backend={backend}  duration={duration_s}s")
    if backend == "rapl_sysfs":
        paths = find_rapl_paths()
        e1 = read_rapl_uj(paths); t1 = time.perf_counter()
        print(f"  Sleeping {duration_s}s ...", flush=True)
        time.sleep(duration_s)
        e2 = read_rapl_uj(paths); t2 = time.perf_counter()
        delta = max(0.0, (e2 - e1 + (2**32 if e2 < e1 else 0)) / 1e6)
        idle_w = delta / (t2 - t1)
        print(f"  idle power = {idle_w:.3f} W")
        return idle_w
    else:
        print("  Wall-time backend: idle_power_w = 0 (not needed)")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CORE: measure one sample's execution energy
# ─────────────────────────────────────────────────────────────────────────────

def _measure_one(
    code: str,
    test_code: str,
    idle_w: float,
    backend: str,
    min_wall_s: float = 1.0,
    timeout_s:  float = 60.0,
) -> dict:
    """
    Returns: {e_exec_j, cpu_instr_proxy, n_reps, status, error}
    cpu_instr_proxy = wall-time per rep in ns (monotone proxy for instructions)
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="bep_"
    ) as f:
        f.write(code.rstrip() + "\n\n" + test_code.rstrip() + "\n")
        tmp = f.name

    paths = find_rapl_paths() if backend == "rapl_sysfs" else []

    try:
        n_reps = 0
        total_wall_s = 0.0

        # Snapshot before
        e_start = read_rapl_uj(paths) if paths else 0.0
        t_start = time.perf_counter()

        while total_wall_s < min_wall_s:
            if time.perf_counter() - t_start > timeout_s * 4:
                return {"e_exec_j": None, "cpu_instr_proxy": None,
                        "n_reps": n_reps, "status": "timeout",
                        "error": "global timeout"}
            t0  = time.perf_counter()
            res = subprocess.run([sys.executable, tmp],
                                 capture_output=True, timeout=timeout_s)
            t1  = time.perf_counter()
            if res.returncode != 0:
                return {"e_exec_j": None, "cpu_instr_proxy": None,
                        "n_reps": n_reps, "status": "error",
                        "error": res.stderr.decode(errors="replace")[:300]}
            n_reps += 1
            total_wall_s += (t1 - t0)

        e_end   = read_rapl_uj(paths) if paths else 0.0
        elapsed = time.perf_counter() - t_start

        if backend == "rapl_sysfs" and paths:
            delta_uj = e_end - e_start
            if delta_uj < 0: delta_uj += 2**32
            total_j  = delta_uj / 1e6
            idle_j   = idle_w * elapsed
            e_per_exec = max(0.0, total_j - idle_j) / n_reps
        else:
            # Wall-time proxy: scale ns/rep to plausible Joules
            # Using power proxy: assume 50W average CPU during execution
            # e_exec_j ≈ power × time_per_rep
            ns_per_rep = (total_wall_s / n_reps) * 1e9
            e_per_exec = (total_wall_s / n_reps) * 50.0   # 50W assumed CPU TDP

        ns_per_rep = (total_wall_s / n_reps) * 1e9
        return {
            "e_exec_j":        e_per_exec,
            "cpu_instr_proxy": ns_per_rep,
            "n_reps":          n_reps,
            "status":          "ok",
            "error":           None,
        }

    except subprocess.TimeoutExpired:
        return {"e_exec_j": None, "cpu_instr_proxy": None,
                "n_reps": n_reps, "status": "timeout", "error": "subprocess timeout"}
    except Exception as exc:
        return {"e_exec_j": None, "cpu_instr_proxy": None,
                "n_reps": 0, "status": "error", "error": str(exc)}
    finally:
        Path(tmp).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — MEASURE e_exec_j FOR ALL PASSING SAMPLES
# ─────────────────────────────────────────────────────────────────────────────

def step3_measure(
    samples_json: str,
    output_csv:   str,
    backend:      str,
    calibrate_s:  float = 30.0,
    min_wall_s:   float = 1.0,
    timeout_s:    float = 60.0,
):
    with open(samples_json) as f:
        samples = json.load(f)

    passing = [s for s in samples if s.get("passed_tests")]
    print(f"\n[measure] {len(passing)} passing samples  (backend={backend})")

    idle_w = step2_calibrate(backend, calibrate_s)

    results = []
    for i, s in enumerate(passing):
        task_id    = s.get("task_id", f"t{i}")
        sample_idx = s.get("sample_idx", 0)
        code       = s.get("_code", "")
        test_code  = s.get("_test_code", "")

        print(f"  [{i+1:>4}/{len(passing)}] {task_id}[{sample_idx}] "
              f"{s.get('method','?')} iter{s.get('iteration','?')} ...",
              end=" ", flush=True)

        if not code.strip():
            print("SKIP (no code)")
            continue

        r = _measure_one(code, test_code, idle_w, backend, min_wall_s, timeout_s)

        if r["status"] == "ok":
            print(f"OK  e={r['e_exec_j']:.4f}J  "
                  f"ns/rep={r['cpu_instr_proxy']:.0f}  reps={r['n_reps']}")
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
            "status":           r["status"],
            "backend":          backend,
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        writer.writeheader()
        writer.writerows(results)

    ok = sum(1 for r in results if r["status"] == "ok")
    e_vals = [r["e_exec_j"] for r in results if r.get("e_exec_j")]
    print(f"\n[measure] {ok}/{len(results)} OK  →  {output_csv}")
    if e_vals:
        print(f"  e_exec_j: min={min(e_vals):.4f}  "
              f"median={statistics.median(e_vals):.4f}  "
              f"max={max(e_vals):.4f}  J")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — LOG GPU ENERGY DURING GENERATION (e_optim_wh)
# ─────────────────────────────────────────────────────────────────────────────

def step4_gpu_log(cmd: str, output_csv: str, label: str = "run") -> float:
    rows: list[dict] = []
    stop  = threading.Event()

    def _poll():
        while not stop.is_set():
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
                        rows.append({"ts": ts, "gpu": p[0],
                                     "power_w": p[1], "util": p[2]})
            except Exception:
                pass
            stop.wait(0.5)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    print(f"[gpu-log] Starting: {cmd}")
    t0 = time.time()
    os.system(cmd)
    t1 = time.time()
    stop.set()
    t.join(timeout=3)

    # Integrate power → Wh (trapezoidal per GPU)
    gpus: dict[str, list] = {}
    for r in rows:
        gpus.setdefault(r["gpu"], []).append(r)

    total_ws = 0.0
    for g_rows in gpus.values():
        g_rows.sort(key=lambda x: x["ts"])
        for i in range(1, len(g_rows)):
            try:
                p1 = float(g_rows[i-1]["power_w"])
                p2 = float(g_rows[i]["power_w"])
                dt = g_rows[i]["ts"] - g_rows[i-1]["ts"]
                total_ws += 0.5 * (p1 + p2) * dt
            except (ValueError, KeyError):
                pass
    wh = total_ws / 3600.0

    with open(output_csv, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"[gpu-log] label={label}  wall={t1-t0:.1f}s  "
          f"energy={wh:.4f} Wh ({wh*3600:.1f} J)  → {output_csv}")
    return wh


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — COMPUTE REAL BEP
# ─────────────────────────────────────────────────────────────────────────────

def step5_bep(
    exec_csv:    str,
    gpu_csv:     str,
    samples_json: str,
    output_json: str = "real_bep_results.json",
):
    """
    Merge exec_energy.csv + gpu_energy.csv into samples.json,
    then compute BEP per (model, method, iteration).

    BEP = E_optim_J / (e_orig_J - e_opt_J)
    """
    import math

    # Load exec energies
    exec_map: dict[tuple, float] = {}
    instr_map: dict[tuple, float] = {}
    with open(exec_csv) as f:
        for row in csv.DictReader(f):
            if row.get("e_exec_j") and row["e_exec_j"] != "None":
                key = (row["task_id"], row["model"],
                       row["method"], int(row["iteration"]),
                       int(row["sample_idx"]))
                exec_map[key]  = float(row["e_exec_j"])
                if row.get("cpu_instructions") and row["cpu_instructions"] != "None":
                    instr_map[key] = float(row["cpu_instructions"])

    # Load GPU energy (single value per generation run)
    gpu_energy_wh = 0.0
    if Path(gpu_csv).exists():
        rows = []
        with open(gpu_csv) as f:
            for row in csv.DictReader(f):
                rows.append(row)
        if rows:
            total_ws = 0.0
            for i in range(1, len(rows)):
                try:
                    p1 = float(rows[i-1]["power_w"])
                    p2 = float(rows[i]["power_w"])
                    dt = float(rows[i]["ts"]) - float(rows[i-1]["ts"])
                    total_ws += 0.5 * (p1 + p2) * dt
                except Exception:
                    pass
            gpu_energy_wh = total_ws / 3600.0
        print(f"[bep] GPU energy from {gpu_csv}: {gpu_energy_wh:.4f} Wh")

    # Load samples and attach energies
    with open(samples_json) as f:
        samples = json.load(f)

    for s in samples:
        key = (s["task_id"], s["model"], s["method"],
                int(s["iteration"]), int(s["sample_idx"]))
        s["e_exec_j"]         = exec_map.get(key)
        s["cpu_instructions"]  = instr_map.get(key)
        # e_optim_wh: use GPU-logged value if available, else use stored value
        if gpu_energy_wh > 0:
            s["e_optim_wh"] = gpu_energy_wh   # apply to all (same generation run)

    # Compute BEP grouped by (model, method, iteration)
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for s in samples:
        groups[(s["model"], s["method"], s["iteration"])].append(s)

    # Build baseline energy per task (iter=0)
    baseline_e: dict[tuple, list[float]] = defaultdict(list)
    for (model, method, iteration), group in groups.items():
        if iteration == 0:
            for s in group:
                if s.get("passed_tests") and s.get("e_exec_j"):
                    baseline_e[(model, s["task_id"])].append(s["e_exec_j"])
    baseline_mean = {k: statistics.mean(v) for k, v in baseline_e.items() if v}

    results = []
    for (model, method, iteration), group in sorted(groups.items()):
        e_optim_j = (group[0].get("e_optim_wh") or 0.0) * 3600.0

        bep_vals = []
        e_red_vals = []
        for s in group:
            if not (s.get("passed_tests") and s.get("e_exec_j") and s.get("is_selected")):
                continue
            e_orig = baseline_mean.get((model, s["task_id"]))
            if not e_orig:
                continue
            e_opt = s["e_exec_j"]
            e_red_vals.append(e_opt / e_orig)
            delta = e_orig - e_opt
            if delta > 0 and e_optim_j > 0:
                bep_vals.append(e_optim_j / delta)

        def _tmean(vals):
            if not vals: return float("nan")
            arr = sorted(vals)
            cut = max(1, int(len(arr) * 0.05))
            trimmed = arr[cut:-cut] if len(arr) - 2*cut > 0 else arr
            return statistics.mean(trimmed) if trimmed else float("nan")

        bep  = _tmean(bep_vals)
        ered = _tmean(e_red_vals)

        row = {
            "model": model.split("/")[-1],
            "method": method,
            "iteration": iteration,
            "e_optim_wh": group[0].get("e_optim_wh", 0),
            "e_optim_j":  e_optim_j,
            "energy_reduction": round(ered, 4) if not math.isnan(ered) else None,
            "BEP": round(bep) if not math.isnan(bep) else None,
            "n_bep_samples": len(bep_vals),
        }
        results.append(row)

        bep_str  = f"{bep:>12,.0f}" if not math.isnan(bep) else "        undef"
        ered_str = f"{ered:.3f}" if not math.isnan(ered) else " N/A"
        print(f"  {model.split('/')[-1]:30s} {method:15s} iter={iteration}"
              f"  E_red={ered_str}  BEP={bep_str}  (n={len(bep_vals)})")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[bep] Results saved → {output_json}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real BEP pipeline: RAPL sysfs + nvidia-smi"
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("step1_check", help="Check RAPL availability + detect backend")

    c2 = sub.add_parser("step2_calibrate", help="Measure idle CPU power")
    c2.add_argument("--backend", default="auto")
    c2.add_argument("--duration", type=float, default=30.0)

    c3 = sub.add_parser("step3_measure", help="Measure e_exec_j for all samples")
    c3.add_argument("--input",     required=True)
    c3.add_argument("--output",    default="exec_energy.csv")
    c3.add_argument("--backend",   default="auto")
    c3.add_argument("--calibrate", type=float, default=30.0)
    c3.add_argument("--min-wall",  type=float, default=1.0)
    c3.add_argument("--timeout",   type=float, default=60.0)

    c4 = sub.add_parser("step4_gpu_log", help="Log GPU energy during a command")
    c4.add_argument("--cmd",    required=True)
    c4.add_argument("--output", default="gpu_energy.csv")
    c4.add_argument("--label",  default="run")

    c5 = sub.add_parser("step5_bep", help="Compute real BEP from measured energies")
    c5.add_argument("--exec",    required=True, help="exec_energy.csv")
    c5.add_argument("--gpu",     required=True, help="gpu_energy.csv")
    c5.add_argument("--samples", required=True, help="samples.json")
    c5.add_argument("--output",  default="real_bep_results.json")

    args = parser.parse_args()

    if args.cmd == "step1_check":
        backend = step1_check()
        print(f"\n→ Save this for next steps: backend={backend}")

    elif args.cmd == "step2_calibrate":
        backend = args.backend
        if backend == "auto":
            paths = find_rapl_paths()
            s = check_rapl_live(paths, 2.0)
            backend = "rapl_sysfs" if s.get("live") else "wall_time"
        step2_calibrate(backend, args.duration)

    elif args.cmd == "step3_measure":
        backend = args.backend
        if backend == "auto":
            paths = find_rapl_paths()
            s = check_rapl_live(paths, 2.0) if paths else {"live": False}
            backend = "rapl_sysfs" if s.get("live") else "wall_time"
            print(f"[auto] detected backend: {backend}")
        step3_measure(
            args.input, args.output, backend,
            args.calibrate, args.min_wall, args.timeout,
        )

    elif args.cmd == "step4_gpu_log":
        step4_gpu_log(args.cmd_arg if hasattr(args, "cmd_arg") else args.cmd,
                      args.output, args.label)

    elif args.cmd == "step5_bep":
        step5_bep(args.exec, args.gpu, args.samples, args.output)

    else:
        parser.print_help()
