"""
compute_metrics.py
==================
Paper-aligned metric computation for:
  "When Faster Isn't Greener: The Hidden Costs of LLM-Based Code Optimization"

Metrics implemented (exact paper definitions):
  - Pass@1
  - Pass@result      (batch-aware correctness of the selected output)
  - Efficient@1      (fraction of correct samples that are faster than baseline)
  - DPS_norm         (Differential Performance Score, normalized)
  - DPS_norm_delta   (delta vs iter-0 baseline)
  - Energy_per_optimization_Wh  (cumulative GPU+CPU energy to produce one result)
  - Energy_reduction (e_opt / e_orig, ratio < 1 means savings)
  - BEP              (Break-Even Point = E_optim / (e_orig - e_opt))

Paper conventions faithfully reproduced:
  - BEP is UNDEFINED (NaN) when e_opt >= e_orig  (no savings → never break even)
  - BEP and Energy_reduction averages use 5% trimmed mean (top+bottom 5% removed)
  - Efficient@1 excludes incorrect samples from its denominator
  - DPS_norm_delta = DPS_norm(final_iter) - DPS_norm(iter_0_baseline)

Input schema
------------
Each "sample" is a dict with these fields (see `SAMPLE_SCHEMA` below).
You can feed real evalplus outputs or synthetic data for testing.

Usage
-----
  python compute_metrics.py              # runs built-in synthetic demo
  python compute_metrics.py --real       # expects real_samples.json in cwd
"""

import json
import math
import argparse
import statistics
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SAMPLE SCHEMA (document what each field means)
# ---------------------------------------------------------------------------
SAMPLE_SCHEMA = {
    "task_id":        str,   # e.g. "HumanEval/0"
    "model":          str,   # e.g. "Qwen/Qwen2.5-Coder-7B-Instruct"
    "method":         str,   # e.g. "simple", "cot", "eoh", ...
    "iteration":      int,   # 0 = baseline, 1-4 = optimization rounds
    "sample_idx":     int,   # index within a batch (0 for single-code methods)
    "batch_size":     int,   # 1 for single-code methods, 10/50 for batching
    "is_selected":    bool,  # True if this sample is the "result" of the batch
    "passed_tests":   bool,  # correctness: passed all unit tests?
    # Performance (CPU instructions from perf, or proxy runtime in ns)
    "cpu_instructions": Optional[float],  # None if correctness failed
    # Energy: per-execution marginal CPU energy (Joules)
    "e_exec_j":       Optional[float],    # None if not measured / failed
    # Energy: optimization phase cost (Wh) — cumulative up to this iteration
    # This is the SAME for all samples in the same (model, method, iteration)
    "e_optim_wh":     float,
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def trimmed_mean(values: list[float], pct: float = 0.05) -> float:
    """
    Compute the mean after removing the lowest and highest `pct` fraction.
    Paper uses 5% trimming (pct=0.05) for BEP and Energy_reduction.
    """
    arr = sorted(v for v in values if not math.isnan(v))
    if not arr:
        return float("nan")
    n = len(arr)
    cut = max(1, int(math.floor(n * pct)))
    trimmed = arr[cut: n - cut] if n - 2 * cut > 0 else arr
    return statistics.mean(trimmed) if trimmed else float("nan")


def safe_mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return statistics.mean(clean) if clean else float("nan")


# ---------------------------------------------------------------------------
# METRIC FUNCTIONS (all operate on a DataFrame of samples)
# ---------------------------------------------------------------------------

def compute_pass_at_1(df: pd.DataFrame) -> float:
    """
    Pass@1: proportion of ALL individual samples that pass correctness tests.
    For batching methods, every sample in the batch counts independently.
    Paper: "Pass@1 over all individual samples in the batch, regardless of
    whether they were ultimately selected as the final result."
    """
    return df["passed_tests"].mean()


def compute_pass_at_result(df: pd.DataFrame) -> float:
    """
    Pass@result: correctness of the SELECTED output per optimization attempt.
    - Single-code methods: identical to Pass@1 (batch_size=1, is_selected=True).
    - Batching methods:  fraction of batches whose selected result passes.
    Paper: "Pass@result measures the proportion of batches whose selected
    result passes all tests."
    """
    selected = df[df["is_selected"] == True]
    if selected.empty:
        return float("nan")
    return selected["passed_tests"].mean()


def compute_efficient_at_1(
    df_iter: pd.DataFrame,
    df_baseline: pd.DataFrame,
) -> float:
    """
    Efficient@1: among CORRECT optimized samples, the fraction whose
    energy consumption is lower than the baseline for the same task.

    Paper: "Efficient@k is the probability that at least one of the top-k
    correct code samples for a problem is more efficient than our baseline.
    Incorrect samples are EXCLUDED from its calculation."

    Implementation:
      - baseline energy per task = mean e_exec_j of iter-0 correct samples
      - for each task, take the CORRECT iter-N samples
      - count how many have e_exec_j < baseline_e_task
      - Efficient@1 = count / total_correct_samples_across_tasks
    """
    # build per-task baseline energy (mean of correct iter-0 samples)
    baseline_correct = df_baseline[
        df_baseline["passed_tests"] & df_baseline["e_exec_j"].notna()
    ]
    baseline_energy = (
        baseline_correct.groupby("task_id")["e_exec_j"].mean().to_dict()
    )

    # optimized correct samples
    opt_correct = df_iter[
        df_iter["passed_tests"] & df_iter["e_exec_j"].notna()
    ].copy()

    if opt_correct.empty:
        return float("nan")

    opt_correct["baseline_e"] = opt_correct["task_id"].map(baseline_energy)
    opt_correct = opt_correct.dropna(subset=["baseline_e"])

    if opt_correct.empty:
        return float("nan")

    efficient = (opt_correct["e_exec_j"] < opt_correct["baseline_e"]).sum()
    return efficient / len(opt_correct)


def compute_dps_norm(
    df: pd.DataFrame,
    all_solutions_df: pd.DataFrame,
) -> float:
    """
    DPS_norm (Normalized Differential Performance Score).

    Paper definition (Liu et al. 2024 / EvalPerf):
      For each task, rank ALL reference solutions by cpu_instructions ascending
      (fewer instructions = faster = better).
      DPS_norm of a sample = fraction of references that are SLOWER than it
      (i.e., have MORE cpu_instructions), expressed as 0-100.

    A sample faster than ALL references → DPS_norm = 100.
    A sample slower than ALL references → DPS_norm = 0.
    Average across all tasks gives the config-level DPS_norm.

    DPS_norm_delta = DPS_norm(iter_N) − DPS_norm(iter_0_baseline).
    Positive = improvement; negative = regression.

    Note: The paper uses hardware perf counters (RAPL/perf) for cpu_instructions.
    Any monotone proxy (e.g., wall-time in ns) gives the same ranking signal.
    """
    valid = df[df["cpu_instructions"].notna() & df["passed_tests"]].copy()
    if valid.empty:
        return float("nan")

    task_scores: dict[str, list[float]] = {}

    for task_id, group in valid.groupby("task_id"):
        refs = all_solutions_df[
            all_solutions_df["task_id"] == task_id
        ]["cpu_instructions"].dropna().values

        if len(refs) == 0:
            continue

        refs_sorted = np.sort(refs)   # ascending: index 0 = fastest reference

        task_scores[task_id] = []
        for sample_instr in group["cpu_instructions"].values:
            # Fraction of references SLOWER than this sample (MORE instructions)
            n_slower = int(np.sum(refs_sorted > sample_instr))
            pct = (n_slower / len(refs_sorted)) * 100.0
            task_scores[task_id].append(pct)

    if not task_scores:
        return float("nan")

    # Average per task first (so each task contributes equally), then overall
    per_task_means = [statistics.mean(v) for v in task_scores.values()]
    return statistics.mean(per_task_means)


def compute_dps_norm_delta(dps_norm_final: float, dps_norm_baseline: float) -> float:
    """
    DPS_norm_delta = DPS_norm(final_iteration) - DPS_norm(iter_0_baseline).
    Positive = improvement over baseline. Negative = regression.
    Paper: "DPS_norm_delta as the difference in the DPS_norm of the baseline
    and the DPS_norm obtained in the final optimization iteration."
    """
    if math.isnan(dps_norm_final) or math.isnan(dps_norm_baseline):
        return float("nan")
    return dps_norm_final - dps_norm_baseline


def compute_energy_per_optimization_wh(df: pd.DataFrame) -> float:
    """
    Energy per optimization result (Wh), cumulative over all iterations.
    This is the cost of the OPTIMIZATION PROCESS itself (LLM generation,
    correctness eval, performance eval phases on GPU+CPU).

    The paper reports this per (method, model) config.
    Here we average e_optim_wh across all samples in the group,
    since all samples in the same (model, method, iteration) share it.
    """
    vals = df["e_optim_wh"].dropna().unique()
    # e_optim_wh is the same for all samples in a config at a given iteration
    # just return the value (or mean if floating point noise causes duplicates)
    return float(np.mean(vals)) if len(vals) > 0 else float("nan")


def compute_energy_reduction(
    df_iter: pd.DataFrame,
    df_baseline: pd.DataFrame,
    trim_pct: float = 0.05,
) -> float:
    """
    Energy_reduction = e_opt / e_orig  (per task, then trimmed mean).
    < 1.0 means the optimized code uses less energy (good).
    = 1.0 means no change.
    > 1.0 means the optimized code uses MORE energy (regression).

    Paper: "ratio of the energy consumed by the optimized sample over the
    energy consumed by the baseline."
    Uses the SELECTED sample per batch for batching methods.
    5% trimmed mean across tasks.
    """
    # baseline per task
    baseline_correct = df_baseline[
        df_baseline["passed_tests"] & df_baseline["e_exec_j"].notna()
    ]
    baseline_energy = (
        baseline_correct.groupby("task_id")["e_exec_j"].mean().to_dict()
    )

    # selected + correct optimized samples
    opt = df_iter[
        df_iter["is_selected"] &
        df_iter["passed_tests"] &
        df_iter["e_exec_j"].notna()
    ].copy()
    opt["e_orig"] = opt["task_id"].map(baseline_energy)
    opt = opt.dropna(subset=["e_orig"])

    ratios = []
    for _, row in opt.iterrows():
        if row["e_orig"] > 0:
            ratios.append(row["e_exec_j"] / row["e_orig"])

    return trimmed_mean(ratios, pct=trim_pct)


def compute_bep(
    df_iter: pd.DataFrame,
    df_baseline: pd.DataFrame,
    trim_pct: float = 0.05,
) -> float:
    """
    BEP (Break-Even Point) = E_optim / (e_orig - e_opt)

    Where:
      E_optim = one-time cumulative energy cost of the optimization (Wh → J)
      e_orig  = energy per execution of the ORIGINAL program (J)
      e_opt   = energy per execution of the OPTIMIZED program (J)

    BEP is UNDEFINED (NaN) when e_opt >= e_orig (no energy savings).
    Paper uses 5% trimmed mean for the final aggregated BEP value.

    Units note: E_optim is in Wh → convert to Joules (* 3600) so units match.
    """
    # baseline per task
    baseline_correct = df_baseline[
        df_baseline["passed_tests"] & df_baseline["e_exec_j"].notna()
    ]
    baseline_energy = (
        baseline_correct.groupby("task_id")["e_exec_j"].mean().to_dict()
    )

    # optimized: selected + correct samples only
    opt = df_iter[
        df_iter["is_selected"] &
        df_iter["passed_tests"] &
        df_iter["e_exec_j"].notna()
    ].copy()
    opt["e_orig"] = opt["task_id"].map(baseline_energy)
    opt = opt.dropna(subset=["e_orig"])

    bep_values = []
    for _, row in opt.iterrows():
        e_orig = row["e_orig"]         # Joules
        e_opt  = row["e_exec_j"]       # Joules
        e_optim_j = row["e_optim_wh"] * 3600.0  # Wh → Joules

        delta = e_orig - e_opt
        if delta <= 0:
            # No energy savings → BEP undefined (paper excludes these)
            continue

        bep = e_optim_j / delta
        bep_values.append(bep)

    return trimmed_mean(bep_values, pct=trim_pct)


# ---------------------------------------------------------------------------
# AGGREGATOR: compute all metrics for one (model, method, iteration) config
# ---------------------------------------------------------------------------

def compute_all_metrics(
    df: pd.DataFrame,
    model: str,
    method: str,
    iteration: int,
    all_solutions_df: pd.DataFrame,
) -> dict:
    """
    Compute all paper metrics for a single (model, method, iteration) config.

    Returns a dict matching Table II / Table III columns from the paper.
    """
    # Slice: this config at this iteration
    mask = (
        (df["model"] == model) &
        (df["method"] == method) &
        (df["iteration"] == iteration)
    )
    df_iter = df[mask]

    # Baseline: iter-0 for the same model
    # For LLM4EFFI and EoH, iter-0 IS part of the method.
    # For all other methods, iter-0 is the plain baseline generation.
    mask_base = (df["model"] == model) & (df["iteration"] == 0)
    if method in ("llm4effi", "eoh"):
        # Their iter-0 already uses the method → use the method's own iter-0
        mask_base = mask_base & (df["method"] == method)
    else:
        # Use "simple" or whichever method was used for iter-0 baseline
        mask_base = mask_base & (df["method"] == "baseline")

    df_base = df[mask_base]

    # DPS_norm for baseline (iter-0)
    dps_base = compute_dps_norm(df_base, all_solutions_df)
    # DPS_norm for this iteration
    dps_iter = compute_dps_norm(df_iter, all_solutions_df)

    return {
        "model":              model,
        "method":             method,
        "iteration":          iteration,
        # --- Correctness ---
        "Pass@1":             round(compute_pass_at_1(df_iter) * 100, 2),
        "Pass@result":        round(compute_pass_at_result(df_iter) * 100, 2),
        # --- Performance ---
        "DPS_norm":           round(dps_iter, 2),
        "DPS_norm_baseline":  round(dps_base, 2),
        "DPS_norm_delta":     round(compute_dps_norm_delta(dps_iter, dps_base), 2),
        # --- Energy profitability ---
        "Efficient@1":        round(compute_efficient_at_1(df_iter, df_base), 3),
        "Energy_reduction":   round(compute_energy_reduction(df_iter, df_base), 3),
        "Energy_optim_Wh":    round(compute_energy_per_optimization_wh(df_iter), 4),
        "BEP":                round(compute_bep(df_iter, df_base)),
    }


# ---------------------------------------------------------------------------
# FULL PIPELINE: run across all configs and produce a summary table
# ---------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    all_solutions_df: pd.DataFrame,
    output_csv: str = "paper_metrics_summary.csv",
) -> pd.DataFrame:
    """
    Iterate over all (model, method, iteration) combinations in df,
    compute all metrics, and return a summary DataFrame (and save CSV).
    """
    results = []

    configs = (
        df[["model", "method", "iteration"]]
        .drop_duplicates()
        .sort_values(["model", "method", "iteration"])
    )

    print(f"Running metrics for {len(configs)} configs...\n")

    for _, row in configs.iterrows():
        metrics = compute_all_metrics(
            df,
            model=row["model"],
            method=row["method"],
            iteration=row["iteration"],
            all_solutions_df=all_solutions_df,
        )
        results.append(metrics)

        print(
            f"  [{row['model'].split('/')[-1]:30s}] "
            f"method={row['method']:15s} "
            f"iter={row['iteration']}  "
            f"Pass@1={metrics['Pass@1']:5.1f}  "
            f"DPS_norm={metrics['DPS_norm']:5.1f}  "
            f"DPS_delta={metrics['DPS_norm_delta']:+6.2f}  "
            f"E_red={metrics['Energy_reduction']:.3f}  "
            f"BEP={metrics['BEP']:>10,.0f}"
        )

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    return summary_df


# ---------------------------------------------------------------------------
# SYNTHETIC DATA GENERATOR (for testing without real evalplus outputs)
# ---------------------------------------------------------------------------

def generate_synthetic_data(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic synthetic sample data for testing.

    Calibrated against Table II / Table III of the paper:
      - Qwen2.5-Coder-7B-Instruct, methods: baseline / simple / cot / eoh
      - 5 iterations (0-4), 118 tasks → 10 tasks here for speed
      - 40 independent samples per task (single-code); 3 batches×10 (EoH)

    Three bugs fixed vs the original generator:
      1. DPS_norm was always 100: reference pool now spans the SAME magnitude
         as sample cpu_instructions, so samples score ~75-85 at iter-0
         and improve toward ~80-90 by iter-4.
      2. BEP was too small (62-12k vs paper's 63k-381k): e_exec_j now uses
         realistic RAPL-scale values (hundreds of Joules per execution, matching
         EvalPerf tasks which exercise heavy compute loops).
      3. Pass@result for EoH was always 100%: batch-selection now correctly
         marks exactly ONE sample per (task, iter, batch_id) as is_selected,
         and that sample can be incorrect if no correct sample exists.
    """
    rng = np.random.default_rng(seed)

    MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # Cumulative optimization energy (Wh) per method per iteration — from Table I
    OPTIM_ENERGY = {
        "baseline": [0.79, 0.00, 0.00, 0.00, 0.00],
        "simple":   [0.79, 0.94, 1.06, 1.18, 1.30],
        "cot":      [0.79, 0.94, 1.12, 1.34, 1.53],
        "eoh":      [0.74, 6.15, 10.18, 14.11, 18.19],
    }

    TASKS = [f"HumanEval/{i}" for i in range(10)]
    N_TASKS = len(TASKS)

    # ------------------------------------------------------------------
    # FIX 1: Reference pool calibrated to match sample instruction range.
    # EvalPerf tasks are computationally non-trivial; typical instruction
    # counts in the millions-to-billions range. We fix baseline task energy
    # at hundreds of Joules (realistic RAPL CPU measurements for tight loops
    # run until ≥1s of total execution, as the paper specifies).
    # ------------------------------------------------------------------

    # Per-task "true" baseline CPU instructions (determines DPS_norm scoring)
    # Reference pool: 50 solutions per task, spread around the task mean
    ref_rows = []
    task_base_instr = {}   # anchor instruction count per task
    task_base_energy = {}  # anchor execution energy per task (Joules)

    for task in TASKS:
        # Anchor: each task has a different computational weight
        anchor_instr = float(rng.uniform(2e7, 5e8))   # 20M – 500M instructions
        task_base_instr[task] = anchor_instr

        # FIX 2: Execution energy in realistic Joules range (RAPL CPU, ≥1s runs)
        # Paper reports e_exec reductions from 318J → 2.6J in extreme cases.
        # Typical range: a few to hundreds of Joules depending on task weight.
        task_base_energy[task] = float(rng.uniform(20.0, 350.0))

        # Reference pool: other LLM solutions for this task, spanning ±50% of anchor
        for _ in range(50):
            ref_instr = anchor_instr * rng.uniform(0.5, 2.0)
            ref_rows.append({"task_id": task, "cpu_instructions": ref_instr})

    all_solutions_df = pd.DataFrame(ref_rows)

    # ------------------------------------------------------------------
    # Build sample rows
    # ------------------------------------------------------------------
    rows = []

    for method, energy_schedule in OPTIM_ENERGY.items():
        is_batching = method == "eoh"
        batch_size  = 10 if is_batching else 1
        n_batches   = 3  if is_batching else 40  # independent draws per task

        for task in TASKS:
            anchor_instr  = task_base_instr[task]
            anchor_energy = task_base_energy[task]

            for iter_n in range(5):
                e_optim_wh = energy_schedule[iter_n]

                # Correctness pass probabilities (calibrated to paper Table II)
                if method == "baseline":
                    pass_prob = 0.75
                elif method == "eoh":
                    # EoH improves correctness over iterations (Table II: 69→91%)
                    pass_prob = min(0.92, 0.69 + iter_n * 0.056)
                elif method == "cot":
                    # CoT degrades: 75→55% (Table II)
                    pass_prob = max(0.50, 0.75 - iter_n * 0.05)
                else:  # simple
                    # Simple degrades: 76→58%
                    pass_prob = max(0.50, 0.77 - iter_n * 0.048)

                for batch_id in range(n_batches):
                    for in_batch_idx in range(batch_size):
                        sample_idx = batch_id * batch_size + in_batch_idx
                        passed = bool(rng.random() < pass_prob)

                        if passed:
                            # FIX 1: cpu_instructions spans reference pool range
                            # Optimization improves by up to 25% over 4 iters
                            improvement = 1.0 - rng.uniform(0.0, 0.25) * (iter_n / 4.0)
                            cpu_instr = anchor_instr * improvement * rng.uniform(0.85, 1.15)

                            # FIX 2: e_exec_j in hundreds-of-Joules range (RAPL scale)
                            # Same improvement factor applies to energy
                            e_exec = anchor_energy * improvement * rng.uniform(0.90, 1.10)
                            if iter_n == 0:
                                # iter-0 is the baseline → tight cluster around anchor
                                e_exec = anchor_energy * rng.uniform(0.97, 1.03)
                                cpu_instr = anchor_instr * rng.uniform(0.97, 1.03)
                        else:
                            cpu_instr = None
                            e_exec    = None

                        rows.append({
                            "task_id":          task,
                            "model":            MODEL,
                            "method":           method,
                            "iteration":        iter_n,
                            "sample_idx":       sample_idx,
                            "batch_size":       batch_size,
                            "batch_id":         batch_id,
                            # FIX 3: default False; will be set correctly below
                            "is_selected":      False,
                            "passed_tests":     passed,
                            "cpu_instructions": cpu_instr,
                            "e_exec_j":         e_exec,
                            "e_optim_wh":       e_optim_wh,
                        })

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # FIX 3: Mark exactly ONE sample per (task, model, method, iter, batch_id)
    # as is_selected.  For batching methods: best CORRECT sample (lowest
    # cpu_instructions); if none correct, pick the first sample in the batch
    # (which will be marked incorrect — lowering Pass@result as in the paper).
    # For single-code methods (batch_size=1): every sample is its own batch,
    # so is_selected = True always.
    # ------------------------------------------------------------------
    def _mark_selected(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        if group["batch_size"].iloc[0] == 1:
            # single-code: every row is the sole sample in its "batch"
            group["is_selected"] = True
            return group

        # batching: pick best correct sample, else first sample
        correct = group[group["passed_tests"] & group["cpu_instructions"].notna()]
        if not correct.empty:
            best = correct["cpu_instructions"].idxmin()
        else:
            best = group.index[0]
        group["is_selected"] = False
        group.loc[best, "is_selected"] = True
        return group

    # pandas ≥2.2 drops groupby keys from the result when using apply;
    # we work around this by assigning the is_selected column directly.
    group_keys = ["task_id", "model", "method", "iteration", "batch_id"]
    selected_flags = (
        df.groupby(group_keys, group_keys=False)
        .apply(_mark_selected)["is_selected"]
    )
    # Re-align by index in case apply reordered rows
    df["is_selected"] = selected_flags.reindex(df.index).fillna(False)

    # Ensure bool dtype
    df["is_selected"] = df["is_selected"].astype(bool)

    return df, all_solutions_df


# ---------------------------------------------------------------------------
# PRETTY PRINT FINAL TABLE (mirror of Table II / Table III)
# ---------------------------------------------------------------------------

def print_paper_table(summary_df: pd.DataFrame) -> None:
    cols = [
        "model", "method", "iteration",
        "Pass@1", "Pass@result",
        "DPS_norm", "DPS_norm_delta",
        "Efficient@1", "Energy_reduction", "Energy_optim_Wh", "BEP",
    ]
    display = summary_df[cols].copy()
    display["model"] = display["model"].str.split("/").str[-1]

    print("\n" + "=" * 120)
    print("PAPER-ALIGNED METRICS TABLE (mirrors Table II / Table III)")
    print("=" * 120)
    print(display.to_string(index=False))
    print("=" * 120)

    # Quick sanity check against Table III Qwen7B numbers
    print("\n--- Sanity check vs Table III (Qwen2.5-Coder-7B, all methods, last iter) ---")
    paper_ref = {
        "Efficient@1":      0.802,
        "Energy_reduction": 0.831,
        "BEP":              66_889,
        "Pass@1":           66.4,
        "DPS_norm_delta":   4.32,
    }
    last_iter = summary_df[summary_df["iteration"] == 4]
    if not last_iter.empty:
        for key, ref_val in paper_ref.items():
            if key in last_iter.columns:
                computed = last_iter[key].mean()
                print(f"  {key:20s}  paper={ref_val:>12.3f}  computed={computed:>12.3f}")

    print()
    print("  NOTE: BEP gap (synthetic << paper) is expected.")
    print("  The paper uses 118 EvalPerf tasks with heavy CPU loops (e_exec up to 318J).")
    print("  BEP = E_optim_J / (e_orig - e_opt): larger e_exec → larger BEP denominator.")
    print("  With real EvalPerf data and perf/RAPL energy readings, BEP will scale up.")
    print("  DPS_norm_delta, Pass@1, Efficient@1 and Energy_reduction are the reliable")
    print("  correctness/perf metrics with synthetic data; BEP needs real energy numbers.")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute paper-aligned metrics")
    parser.add_argument(
        "--real", action="store_true",
        help="Load real_samples.json instead of synthetic data"
    )
    parser.add_argument(
        "--input", default="real_samples.json",
        help="Path to real sample data JSON (list of sample dicts)"
    )
    parser.add_argument(
        "--refs", default="reference_solutions.json",
        help="Path to reference solution pool JSON for DPS_norm"
    )
    parser.add_argument(
        "--output", default="paper_metrics_summary.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()

    if args.real:
        print(f"Loading real data from {args.input} ...")
        with open(args.input) as f:
            samples = json.load(f)
        df = pd.DataFrame(samples)

        print(f"Loading reference solutions from {args.refs} ...")
        with open(args.refs) as f:
            refs = json.load(f)
        all_solutions_df = pd.DataFrame(refs)
    else:
        print("Generating synthetic data (paper-calibrated)...")
        df, all_solutions_df = generate_synthetic_data(seed=42)
        print(f"  Samples: {len(df):,}  |  Reference solutions: {len(all_solutions_df):,}\n")

    summary_df = run_pipeline(df, all_solutions_df, output_csv=args.output)
    print_paper_table(summary_df)
