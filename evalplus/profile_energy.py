"""Evaluate the energy consumptions of optimized and unoptimized programs."""

import glob
import json
import os
import time
from datetime import datetime

import click
import rich

from evalplus import evalperf
from evalplus.data import get_evalperf_data
from evalplus.evalopti import DEFAULT_ROOT, load_task_eval_results_from_json_dict
from evalplus.evalperf import get_max_workers, rule, not_none, table_print, TaskEvalResultEnergy, TaskEvalResult
from evalplus.perf.energy import EnergyMeasurement, calibrate


@click.command()
@click.option('--root', type=str, default=DEFAULT_ROOT, help='The root directory where results will be saved.')
@click.option("--test-single-task", is_flag=True, help="Evaluate a single task. For debugging purposes.")
@click.option("--calibration-time", type=float, default=30,
              help="The time in second to calibrate energy measures between iterations.")
@click.option("--nb-workers", type=int, default=1)
@click.option("--max-profile", type=int, default=20,
              help="The maximum number of samples to profile for energy in a task")
@click.option("--min-duration", type=float, default=0.25,
              help="The minimum duration of a sample profiling")
@click.option("--strategy", type=click.Choice(["evalperf-classic"]), default="evalperf-classic")
@click.option("--result-file", type=str, default=None, help="If not set, all result files will be evaluated")
@click.option("--split-start", type=int, default=None)
@click.option("--split-end", type=int, default=None)
@click.option("--pattern", type=str, default="evalperf/*evalopti_results.json",
              help="glob pattern to filter the results files to process")
def profile_energy(root: str, test_single_task: bool, calibration_time: float, nb_workers: int, max_profile: int,
                   min_duration: float, strategy: str, result_file: str, split_start: int, split_end: int,
                   pattern: str):
    rule(f"Evaluating energy measures !")
    # Data loading
    # problems, expected_output = get_evalplus_data()
    ptasks_to_do = get_evalperf_data()
    print(f"{len(ptasks_to_do)} tasks found")

    if test_single_task:
        task_id_to_keep = "HumanEval/16"
        ptasks_to_do = {task_id_to_keep: ptasks_to_do[task_id_to_keep]}

    max_workers = get_max_workers(nb_workers)
    min_correct = 1
    # method = "task-average"
    method = "sample-level"

    if result_file:
        files = [result_file]
    else:
        files = get_all_result_files(root, pattern)
    files_to_process = get_files_to_process(files, method, strategy)

    files_to_process = get_split(files_to_process, split_end, split_start)

    for original_result_file in files_to_process:
        energy_result_path = build_result_path(original_result_file, method=method, strategy=strategy)
        if os.path.exists(energy_result_path):
            rich.print(f"Skipping {energy_result_path}")
            continue
        rich.print(f"\n")
        rule(f"Evaluating {original_result_file}")
        energy_profiling_config = {
            "root": root,
            "test_single_task": test_single_task,
            "calibration_time": calibration_time,
            "max_workers": max_workers,
            "max_profile": max_profile,
            "min_correct": min_correct,
            "original_result_file": original_result_file,
            "min_duration": min_duration,
            "profiling_method": method,
            "strategy": strategy
        }

        task_eval_results, config = load_results_from_file(result_path=original_result_file)

        # Removing unprofiled codes
        for task_id, task in task_eval_results.items():
            task.results = list(filter(lambda code_res: code_res.profiled, task.results))

        energy_measures = {}
        calibrate_phase(calibration_time, energy_measures)
        task_eval_results, profiling_energy = evalperf.evaluate_energy_of_samples(task_eval_results,
                                                                                  max_workers=max_workers,
                                                                                  min_correct=min_correct,
                                                                                  ptasks=ptasks_to_do,
                                                                                  max_profile=max_profile,
                                                                                  min_duration=min_duration)
        energy_measures["energy_profiling"] = profiling_energy
        print_and_save_results(energy_measures, min_correct, energy_result_path,
                               task_eval_results, config, energy_profiling_config)


def get_split(files_to_process, split_end, split_start):
    if split_start is not None and split_end is not None:
        print(f"Splitting {split_start} to {split_end}")
        files_to_process = files_to_process[split_start:split_end]
    print(f"{len(files_to_process)} split")
    print(f"Split is : {files_to_process}")
    return files_to_process


def get_files_to_process(files, method, strategy):
    files_to_process = []
    for file in files:
        energy_result_path = build_result_path(file, method=method, strategy=strategy)
        if os.path.exists(energy_result_path):
            # rich.print(f"Skipping {energy_result_path}")
            continue
        else:
            files_to_process.append(file)
    print(f"{len(files_to_process)} files to process")
    return files_to_process


def calibrate_phase(calibration_time, energy_measures):
    rich.print("Doing nothing before calibration...")
    time.sleep(calibration_time / 10)
    rich.print(f"Calibrating for {calibration_time} seconds...")
    energy_measures["idle"] = calibrate(calibration_time, gpu=False)


def build_result_path(original_result_path, method: str, strategy: str) -> str:
    assert original_result_path.endswith("evalopti_results.json")
    result_path = original_result_path.replace("evalopti_results.json", f"{method}_{strategy}_energy_results.json")
    return result_path


def get_all_result_files(root, pattern) -> list[str]:
    files = glob.glob(os.path.join(root, pattern))
    files = sorted(list(filter(lambda f: "summary" not in f, files)))
    rich.print(f"Found {len(files)} files in {root}")
    return files


def load_results_from_file(result_path) -> tuple[dict[str, TaskEvalResultEnergy], dict]:
    with open(result_path) as f:
        resumed_results = json.load(f)

    task_eval_results: dict[str, TaskEvalResult] = load_task_eval_results_from_json_dict(resumed_results)
    config = resumed_results["config"]

    task_eval_results: dict[str, TaskEvalResultEnergy] = {key: TaskEvalResultEnergy.from_task_eval_result(res) for
                                                          key, res in task_eval_results.items()}
    rich.print(f"Loaded {len(task_eval_results)} results from {result_path}")
    return task_eval_results, config


def print_and_save_results(energy_measures: dict[str, EnergyMeasurement], min_correct,
                           result_path,
                           task_eval_results: dict[str, TaskEvalResultEnergy], config, energy_profiling_config):
    rule("Evaluation Summary")
    n_evalperfed = len(not_none([res.energy_measures for res in task_eval_results.values()]))

    table_print(
        "Energy Profiling summary",
        {
            "#EvalPerf-ed tasks": f"{n_evalperfed} / {len(task_eval_results)}",
            "min_correct": min_correct,
            "energy_energy_profiling": energy_measures["energy_profiling"].total_energy
        },
    )
    with open(result_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "config": config,
                    "energy_profiling_config": energy_profiling_config,
                    "summary": {
                        "energy_measures": energy_measures,
                    },
                    "eval": task_eval_results,
                }
                , default=vars)
        )
    rich.print(f"Full results have been saved to {result_path}")


if __name__ == "__main__":
    profile_energy()
