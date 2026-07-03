"""
parse_evalplus_outputs.py
=========================
Converts real EvalPlus output files (jsonl + eval_results.json) into the
flat sample schema consumed by compute_metrics.py.

EvalPlus output structure (after running evalplus.evaluate):
  evalplus_results/
    humaneval/
      <model>_vllm_temp_0.0.jsonl            <- raw generated code per task
      <model>_vllm_temp_0.0.raw.jsonl        <- same + metadata
      <model>_vllm_temp_0.0.eval_results.json <- pass/fail per task per sample

Usage
-----
  python parse_evalplus_outputs.py \
      --results_dir evalplus_results \
      --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
      --method simple \
      --iteration 0 \
      --e_optim_wh 0.79 \
      --output samples_iter0.json

Then run:
  python compute_metrics.py --real --input samples_iter0.json

For multiple iterations, run once per iteration and concatenate:
  python parse_evalplus_outputs.py ... --iteration 1 --e_optim_wh 0.94 >> samples.json
  python compute_metrics.py --real --input samples.json
"""

import json
import argparse
import glob
from pathlib import Path
from typing import Optional


def load_eval_results(eval_json_path: str) -> dict:
    """
    Load the eval_results.json produced by evalplus.
    Returns: {task_id: {"passed": bool, "result": str}, ...}
    """
    with open(eval_json_path) as f:
        data = json.load(f)

    # evalplus eval_results.json structure:
    # { "eval": { "HumanEval/0": [ {"solution": "...", "status": "pass"}, ... ] } }
    results = {}
    eval_block = data.get("eval", data)  # handle both formats
    for task_id, samples in eval_block.items():
        if isinstance(samples, list):
            results[task_id] = samples
        elif isinstance(samples, dict):
            # older format: {"base": [...], "plus": [...]}
            results[task_id] = samples.get("base", [])
    return results


def load_raw_jsonl(jsonl_path: str) -> list[dict]:
    """Load the raw .jsonl file (one JSON object per line)."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_evalplus_files(
    results_dir: str,
    model: str,
    dataset: str = "humaneval",
) -> tuple[Optional[str], Optional[str]]:
    """
    Auto-locate the jsonl and eval_results.json for a given model+dataset.
    Returns (raw_jsonl_path, eval_results_json_path) or (None, None).
    """
    model_slug = model.replace("/", "--")
    base = Path(results_dir) / dataset

    raw_pattern = str(base / f"{model_slug}*.raw.jsonl")
    eval_pattern = str(base / f"{model_slug}*.eval_results.json")

    raw_files = glob.glob(raw_pattern)
    eval_files = glob.glob(eval_pattern)

    if not raw_files:
        # try without --
        raw_pattern2 = str(base / f"*.raw.jsonl")
        raw_files = glob.glob(raw_pattern2)

    if not eval_files:
        eval_pattern2 = str(base / f"*.eval_results.json")
        eval_files = glob.glob(eval_pattern2)

    raw_path = raw_files[0] if raw_files else None
    eval_path = eval_files[0] if eval_files else None

    return raw_path, eval_path


def build_samples(
    raw_jsonl_path: str,
    eval_results_path: str,
    model: str,
    method: str,
    iteration: int,
    e_optim_wh: float,
    dataset: str = "humaneval",
    batch_size: int = 1,
) -> list[dict]:
    """
    Merge raw generations with pass/fail results into flat sample dicts.

    Each dict matches the SAMPLE_SCHEMA in compute_metrics.py.

    Note on energy fields:
      - e_exec_j (execution energy in Joules) is NOT available from evalplus.
        You must fill this in from your energy_logger.sh measurements or
        from perf/RAPL readings. Set to None here as placeholder.
      - cpu_instructions: same — fill from perf counters or EvalPerf measurements.
        Set to None as placeholder.
    """
    raw_rows = load_raw_jsonl(raw_jsonl_path)
    eval_results = load_eval_results(eval_results_path)

    samples = []
    for row in raw_rows:
        task_id = row.get("task_id", row.get("problem", "unknown"))

        # evalplus raw.jsonl may have "completion" or "solution"
        code = row.get("solution", row.get("completion", ""))

        # sample index within the task (evalplus uses list ordering)
        sample_idx = row.get("_index", 0)

        # look up pass/fail
        task_evals = eval_results.get(task_id, [])
        passed = False
        if isinstance(task_evals, list) and sample_idx < len(task_evals):
            entry = task_evals[sample_idx]
            if isinstance(entry, dict):
                passed = entry.get("status", "") == "pass"
            elif isinstance(entry, bool):
                passed = entry
        elif isinstance(task_evals, bool):
            passed = task_evals

        # batch logic: for single-code methods, every sample is "selected"
        # for batching methods you'd pick the best — left as True for now
        is_selected = True  # override for batching in post-processing

        samples.append({
            "task_id":          f"{dataset.upper()}/{task_id}" if "/" not in task_id else task_id,
            "model":            model,
            "method":           method,
            "iteration":        iteration,
            "sample_idx":       sample_idx,
            "batch_size":       batch_size,
            "batch_id":         sample_idx // batch_size,
            "is_selected":      is_selected,
            "passed_tests":     passed,
            # --- FILL THESE IN from energy_logger / perf measurements ---
            "cpu_instructions": None,   # float or None
            "e_exec_j":         None,   # float (Joules) or None
            # --- FROM optimization phase energy logger ---
            "e_optim_wh":       e_optim_wh,
            # extra context
            "_code":            code,
        })

    return samples


def fill_energy_from_csv(
    samples: list[dict],
    energy_csv_path: str,
    task_energy_map: Optional[dict] = None,
) -> list[dict]:
    """
    Optionally fill e_exec_j from a CSV mapping task_id → energy_j.
    The CSV should have columns: task_id, sample_idx, e_exec_j [, cpu_instructions]

    If you have energy_logger.sh output (GPU power CSV), use it to fill
    e_optim_wh before calling build_samples instead.
    """
    import csv

    if task_energy_map is None:
        task_energy_map = {}
        with open(energy_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["task_id"], int(row.get("sample_idx", 0)))
                task_energy_map[key] = {
                    "e_exec_j": float(row["e_exec_j"]) if row.get("e_exec_j") else None,
                    "cpu_instructions": float(row["cpu_instructions"]) if row.get("cpu_instructions") else None,
                }

    for s in samples:
        key = (s["task_id"], s["sample_idx"])
        if key in task_energy_map:
            s["e_exec_j"] = task_energy_map[key]["e_exec_j"]
            s["cpu_instructions"] = task_energy_map[key]["cpu_instructions"]

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Parse EvalPlus outputs into compute_metrics.py schema"
    )
    parser.add_argument("--results_dir", default="evalplus_results")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--method", default="simple")
    parser.add_argument("--dataset", default="humaneval")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--e_optim_wh", type=float, default=0.79,
                        help="Cumulative optimization energy (Wh) for this iteration")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output", default="samples.json")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing output file instead of overwriting")
    parser.add_argument("--energy_csv", default=None,
                        help="Optional CSV with per-sample execution energy")
    args = parser.parse_args()

    raw_path, eval_path = find_evalplus_files(
        args.results_dir, args.model, args.dataset
    )

    if not raw_path or not eval_path:
        print(f"[ERROR] Could not find evalplus output files in {args.results_dir}/{args.dataset}/")
        print(f"  raw.jsonl pattern:       {args.model.replace('/', '--')}*.raw.jsonl")
        print(f"  eval_results.json pattern: {args.model.replace('/', '--')}*.eval_results.json")
        raise SystemExit(1)

    print(f"[INFO] raw jsonl:    {raw_path}")
    print(f"[INFO] eval results: {eval_path}")

    samples = build_samples(
        raw_jsonl_path=raw_path,
        eval_results_path=eval_path,
        model=args.model,
        method=args.method,
        iteration=args.iteration,
        e_optim_wh=args.e_optim_wh,
        dataset=args.dataset,
        batch_size=args.batch_size,
    )

    if args.energy_csv:
        samples = fill_energy_from_csv(samples, args.energy_csv)

    # Load existing if appending
    existing = []
    if args.append and Path(args.output).exists():
        with open(args.output) as f:
            existing = json.load(f)

    all_samples = existing + samples

    with open(args.output, "w") as f:
        json.dump(all_samples, f, indent=2)

    n_passed = sum(1 for s in samples if s["passed_tests"])
    print(f"[INFO] Written {len(samples)} samples ({n_passed} passed) → {args.output}")
    print(f"[INFO] Total in file: {len(all_samples)}")
    print(f"\nNext step:")
    print(f"  python compute_metrics.py --real --input {args.output}")


if __name__ == "__main__":
    main()
