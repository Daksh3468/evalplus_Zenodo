"""
gpu_pipeline.py
===============
LIGHTNING AI SIDE — GPU-only pipeline.
Restricted to: simple, cot, cod, self-refine-nl-feedback.

Lightning blocks perf_event_paranoid, so Cirron instruction counting
(evalplus.perf.profile.num_instruction_profiler) can never succeed here.
That's the backend evalopti.py's perf_worker() needs both for DPS scoring
and for setting CodeEvalResult.profiled — which gates chaining into the
next iteration for any optimizer where can_accept_errored_samples() is
False (true for all four optimizers here). Rather than patch around
that, this bypasses evalopti.py's CLI entirely and drives the optimizer
classes directly:

  - GENERATION/OPTIMIZATION (GPU-bound): runs here. GPU energy is
    measured directly around each call via nvidia-smi polling.
  - CORRECTNESS CHECKING (CPU-bound, perf-INDEPENDENT): also runs here,
    via evalplus's own untrusted_check. Needed to know which samples
    "passed" for chaining.
  - PROFILING (Cirron) and ENERGY (RAPL) are SKIPPED ENTIRELY here.
    Every sample is written with profiled=False, dps=None, cpu_time=None.
    cpu_rapl_pipeline.py fills these in later, locally.

Chaining rule (replaces evalopti.py's `passed AND profiled` filter):
    forward only samples where r.passed is True
  (all four optimizers here have can_accept_errored_samples() == False,
   so this is the only branch that applies — no errored-sample pass-through)

Why these four are a clean fit for this split:
  - Simple / CoT / CoD: prompts only use code_content + task_description,
    never touch cpu_time/dps.
  - Self-Refine-NL: feedback comes from a FRESH LLM call (FEEDBACK_PROMPT)
    critiquing the code, not from execution/profiling data.
  None of the four reference profiling data during optimization, so there
  is no placeholder/None-feedback degradation like Self-Refine-Exec or
  LLM4EFFI would have under this split, and no fitness-ranking dependency
  like EoH has.

Run order
---------
  Lightning, per optimizer (repeat for simple, cot, cod,
  self-refine-nl-feedback):

    python gpu_pipeline.py \
        --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
        --optimizer simple --iterations 4

  Then copy evalplus_results/gpu_only/*_gpu_results.json to your local
  RAPL machine and run cpu_rapl_pipeline.py (see that file's docstring).
"""

import argparse
import json
import os
import time
import subprocess
import threading
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Optional

import rich
from rich.rule import Rule

from evalplus.codegen import run_codegen, get_result_file_identifier
import evalplus.codegen as codegen_mod
from evalplus.data import get_evalperf_data
from evalplus.data.utils import stream_jsonl
from evalplus.evalperf import (
    get_evalplus_data, get_max_workers, TaskEvalResult, check_correctness,
)
from evalplus.perf.optimizer.optimizer import (
    OptimizerSimple, OptimizerBase, OptimizerCoT, OptimizerCoD,
    OptimizerSelfRefineNlFeedback,
)
from evalplus.provider import ModelConfig

OPTIMIZER_MAP = {
    "simple": OptimizerSimple,
    "cot": OptimizerCoT,
    "cod": OptimizerCoD,
    "self-refine-nl-feedback": OptimizerSelfRefineNlFeedback,
}


# ─────────────────────────────────────────────────────────────────────────
# GPU energy — measured directly around each generate/optimize call
# ─────────────────────────────────────────────────────────────────────────

class NvidiaSmiEnergyMonitor:
    """Poll nvidia-smi only while active (context-managed); trapezoidal Wh."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self._rows = []
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._rows = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        return False

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,power.draw",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                )
                ts = time.time()
                for line in out.strip().splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) >= 2:
                        self._rows.append((ts, parts[0], parts[1]))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    @property
    def energy_wh(self) -> float:
        by_gpu = defaultdict(list)
        for ts, gpu, p in self._rows:
            by_gpu[gpu].append((ts, p))
        ws = 0.0
        for rows in by_gpu.values():
            rows.sort(key=lambda x: x[0])
            for i in range(1, len(rows)):
                try:
                    p1, p2 = float(rows[i - 1][1]), float(rows[i][1])
                    dt = rows[i][0] - rows[i - 1][0]
                    ws += 0.5 * (p1 + p2) * dt
                except ValueError:
                    pass
        return ws / 3600.0


# ─────────────────────────────────────────────────────────────────────────
# Result I/O — profiled/dps/cpu_time always left unset here
# ─────────────────────────────────────────────────────────────────────────

def init_eval_results(ptasks):
    results = {}
    for task_id, ptask in ptasks.items():
        results[task_id] = TaskEvalResult(
            task_id=task_id,
            ref=[{"solution": s, "score": r, "_num_cpu_instructions": None}
                 for s, r in zip(ptask["reference"], ptask["scores"])],
        )
    return results


def build_result_path(root, model, optimizer, iteration, backend, temperature):
    identifier = get_result_file_identifier(backend, None, model, temperature)
    if iteration > 0:
        identifier += f"_{optimizer}_{iteration}"
    return os.path.join(root, "gpu_only", f"{identifier}_gpu_results.json")


def save_iteration_results(path, task_eval_results, config, gpu_energy_wh, duration_s):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "config": config,
            "gpu_energy_wh": gpu_energy_wh,   # THIS iteration's segment only
            "gpu_phase_duration_s": duration_s,
            "eval": task_eval_results,
        }, f, default=vars)
    rich.print(f"[green]Saved -> {path}[/]  (GPU: {gpu_energy_wh:.4f} Wh, {duration_s:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────

def run_pipeline(
    model: str, optimizer: str, iterations: int, n_samples: int,
    root: str, backend: str, base_url: Optional[str], temperature: float,
    bs: int, nb_workers_generation: int, nb_workers_wanted: Optional[int],
    min_correct: int = 1,
):
    assert optimizer in OPTIMIZER_MAP, (
        f"Unknown or unsupported optimizer {optimizer!r}. "
        f"This pipeline only supports: {list(OPTIMIZER_MAP.keys())}"
    )
    model_config = ModelConfig(model=model, backend=backend, temperature=temperature,
                              batch_size=bs, dataset="evalperf", base_url=base_url)
    optimizer_model: OptimizerBase = OPTIMIZER_MAP[optimizer](model_config, nb_workers=nb_workers_generation)
    max_workers = get_max_workers(nb_workers_wanted)

    problems, expected_output = get_evalplus_data()
    ptasks_to_do = get_evalperf_data()
    dataset_dict = codegen_mod.get_dataset_dict("evalperf", None, None)

    config = {
        "optimizer": optimizer, "model": model, "backend": backend,
        "temperature": temperature, "n_samples": n_samples, "bs": bs,
        "min_correct": min_correct,
        "accept_errored_samples": optimizer_model.can_accept_errored_samples(),
        "profiling_deferred_to_local_rapl_machine": True,
    }
    rich.print(config)

    prev_results = None
    prev_cumulative_wh = 0.0
    all_result_paths = []

    for it in range(iterations + 1):
        rich.print(Rule(f"Iteration {it}/{iterations}  —  GPU-only (no perf/Cirron on Lightning)"))
        ptasks = copy(ptasks_to_do)

        with NvidiaSmiEnergyMonitor() as mon:
            t0 = time.time()
            if it == 0:
                # None of these four optimizers generate from scratch;
                # iteration 0 is always plain codegen.
                codegen_res = run_codegen(
                    dataset="evalperf", model=model, n_samples=n_samples,
                    temperature=temperature, backend=backend, base_url=base_url,
                    bs=bs, root=root,
                )
                first, _ = codegen_res
                samples_path = first[0] if isinstance(first, tuple) else first
            else:
                # Chain on `passed` only — profiling isn't available on Lightning.
                # (can_accept_errored_samples() is False for all four optimizers
                # here, so this is strictly "forward passed samples only".)
                samples_to_optimize = defaultdict(list)
                for task_id, task in prev_results.items():
                    if task_id not in ptasks:
                        continue
                    for r in task.results:
                        if r.passed:
                            samples_to_optimize[task_id].append(r)
                samples_path = optimizer_model.optimize(
                    samples_to_optimize=samples_to_optimize, dataset_dict=dataset_dict,
                    iter_number=it, root_path=root,
                )
            t1 = time.time()
        seg_wh = mon.energy_wh
        cumulative_wh = prev_cumulative_wh + seg_wh
        prev_cumulative_wh = cumulative_wh

        rich.print(f"  GPU phase: {t1 - t0:.1f}s   segment: {seg_wh:.4f} Wh   "
                  f"cumulative: {cumulative_wh:.4f} Wh")

        # --- Correctness only (perf-independent: plain untrusted_check) ---
        task_eval_results = init_eval_results(ptasks)
        samples_to_evaluate = defaultdict(list)
        for task in stream_jsonl(samples_path):
            samples_to_evaluate[task["task_id"].replace("_", "/")].append(
                (task["solution"], task.get("sample_id")))

        task_eval_results, _correctness_energy = check_correctness(
            task_eval_results, expected_output, max_workers, n_samples,
            problems, ptasks, samples_to_evaluate,
        )
        # profiled/dps/cpu_time/num_cpu_instructions stay at dataclass
        # defaults (False/None) — filled in later by cpu_rapl_pipeline.py.

        n_pass = sum(len([r for r in t.results if r.passed]) for t in task_eval_results.values())
        n_total = sum(len(t.results) for t in task_eval_results.values())
        rich.print(f"  Correctness: {n_pass}/{n_total} passed  ({len(task_eval_results)} tasks)")

        config_it = dict(config, iteration=it)
        result_path = build_result_path(root, model, optimizer, it, backend, temperature)
        save_iteration_results(result_path, task_eval_results, config_it, cumulative_wh, t1 - t0)
        all_result_paths.append(result_path)

        prev_results = task_eval_results

    rich.print(Rule("Done"))
    for p in all_result_paths:
        rich.print(f"  {p}")
    rich.print("\nNext: copy these *_gpu_results.json files to your local RAPL "
              "machine and run cpu_rapl_pipeline.py profile")
    return all_result_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--optimizer", required=True, choices=list(OPTIMIZER_MAP.keys()))
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=40)
    ap.add_argument("--root", default="evalplus_results")
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--nb-workers-generation", type=int, default=50)
    ap.add_argument("--nb-workers-wanted", type=int, default=None)
    ap.add_argument("--min-correct", type=int, default=1)
    args = ap.parse_args()
    run_pipeline(
        model=args.model, optimizer=args.optimizer, iterations=args.iterations,
        n_samples=args.n_samples, root=args.root, backend=args.backend,
        base_url=args.base_url, temperature=args.temperature, bs=args.bs,
        nb_workers_generation=args.nb_workers_generation,
        nb_workers_wanted=args.nb_workers_wanted, min_correct=args.min_correct,
    )


if __name__ == "__main__":
    main()