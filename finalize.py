"""
finalize.py
============
Merge *_rapl_results.json across all (model, optimizer, iteration)
combos and compute DPS/DPS_norm/Pass@1 plus BEP, using real RAPL
execution energy and real per-iteration GPU energy recorded by
gpu_pipeline.py.

Usage:
  python finalize.py --results-dir . --out bep_final.json
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict
from statistics import mean


def load_all(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_rapl_results.json"))):
        with open(path) as f:
            data = json.load(f)
        rows.append((path, data["config"], data["gpu_energy_wh"], data["eval"]))
    return rows


def trimmed_mean(vals, pct=0.05):
    arr = sorted(v for v in vals if v is not None and not math.isnan(v))
    if not arr:
        return float("nan")
    cut = max(1, int(len(arr) * pct))
    trimmed = arr[cut:-cut] if len(arr) - 2 * cut > 0 else arr
    return mean(trimmed) if trimmed else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", default="bep_final.json")
    args = ap.parse_args()

    rows = load_all(args.results_dir)
    print(f"Loaded {len(rows)} iteration files")

    # Group by (model, optimizer); within each, sort by iteration for cumulative GPU energy.
    grouped = defaultdict(list)
    for path, config, gpu_wh, ev in rows:
        grouped[(config["model"], config["optimizer"])].append((config["iteration"], gpu_wh, ev))

    baseline_exec: dict[tuple, list] = defaultdict(list)   # (model, task_id) -> [e_exec_j]
    per_config = {}  # (model, optimizer, iteration) -> {dps, dps_norm, pass1, gpu_wh, samples:[...]}

    for (model, optimizer), entries in grouped.items():
        entries.sort(key=lambda x: x[0])
        for iteration, gpu_wh, ev in entries:
            samples = []
            n_pass, n_total = 0, 0
            dps_vals, dps_norm_vals = [], []
            for task_id, task in ev.items():
                for r in task["results"]:
                    n_total += 1
                    if r.get("passed"):
                        n_pass += 1
                    if r.get("dps") is not None:
                        dps_vals.append(r["dps"])
                    if r.get("dps_norm") is not None:
                        dps_norm_vals.append(r["dps_norm"])
                    samples.append({
                        "task_id": task_id, "passed": r.get("passed"),
                        "profiled": r.get("profiled"), "e_exec_j": r.get("e_exec_j"),
                    })
                    if iteration == 0 and r.get("passed") and r.get("e_exec_j") is not None:
                        baseline_exec[(model, task_id)].append(max(0.0, r["e_exec_j"]))

            per_config[(model, optimizer, iteration)] = {
                "gpu_energy_wh": gpu_wh,
                "pass_at_1": 100 * n_pass / n_total if n_total else 0.0,
                "dps": mean(dps_vals) if dps_vals else 0.0,
                "dps_norm": mean(dps_norm_vals) if dps_norm_vals else 0.0,
                "samples": samples,
            }

    baseline_mean = {k: mean(v) for k, v in baseline_exec.items() if v}

    results = []
    print(f"\n{'Model':30s} {'Method':15s} {'Iter':4s}  {'Pass@1':7s}  {'DPS':6s}  {'DPS_norm':8s}  {'E_red':6s}  {'BEP':>12s}")
    for (model, optimizer, iteration), cfg in sorted(per_config.items()):
        e_optim_j = cfg["gpu_energy_wh"] * 3600.0
        bep_vals, ered_vals = [], []
        for s in cfg["samples"]:
            if not (s["passed"] and s["e_exec_j"] is not None):
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
        print(f"{model.split('/')[-1]:30s} {optimizer:15s} {iteration:4d}  "
              f"{cfg['pass_at_1']:6.1f}%  {cfg['dps']:6.1f}  {cfg['dps_norm']:8.1f}  {ered_str}  {bep_str}")

        results.append({
            "model": model.split("/")[-1], "method": optimizer, "iteration": iteration,
            "pass_at_1": round(cfg["pass_at_1"], 2),
            "dps": round(cfg["dps"], 2), "dps_norm": round(cfg["dps_norm"], 2),
            "gpu_energy_wh_cumulative": round(cfg["gpu_energy_wh"], 4),
            "energy_reduction": None if math.isnan(ered) else round(ered, 4),
            "BEP": None if math.isnan(bep) else round(bep),
            "n_bep_samples": len(bep_vals),
        })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()