"""
Evaluate various optimizers as well as their energy consumption
"""
import json
import os
import time
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Optional

import click
import rich
from rich.rule import Rule
from rich.table import Table

import evalplus.codegen
from evalplus.codegen import run_codegen, get_result_file_identifier, get_target_path
from evalplus.data import get_evalperf_data
from evalplus.data.utils import stream_jsonl
from evalplus.evalperf import get_evalplus_data, get_max_workers, TaskEvalResult, evaluate_performance_of_samples, \
    table_print, rule, check_correctness, not_none, CodeEvalResult
from evalplus.perf.energy import monitor_energy_for_function, EnergyMeasurement, calibrate
from evalplus.perf.optimizer.eoh import OptimizerEoH
from evalplus.perf.optimizer.llm4effi import OptimizerLLM4EFFI
from evalplus.perf.optimizer.optimizer import OptimizerSimple, OptimizerBase, OptimizerCoT, OptimizerCoD, \
    OptimizerSelfRefineNlFeedback, OptimizerSelfRefineExecFeedback
from evalplus.perf.optimizer.simple10 import OptimizerSimple10
from evalplus.perf.profile import simple_test_profiler
from evalplus.provider import ModelConfig

DEFAULT_ROOT = 'evalplus_results'

optimizer_name_map = {
    "simple": OptimizerSimple,
    "cot": OptimizerCoT,
    "cod": OptimizerCoD,
    "self-refine-nl-feedback": OptimizerSelfRefineNlFeedback,
    "self-refine-exec-feedback": OptimizerSelfRefineExecFeedback,
    "llm4effi": OptimizerLLM4EFFI,
    "eoh": OptimizerEoH,
    "simple10": OptimizerSimple10,
}


@dataclass
class EvalOptiResults:
    """Only for one iteration"""
    config: dict
    energy_measures: dict[str, EnergyMeasurement]
    eval_results: dict[str, TaskEvalResult]


@click.command()
@click.option("--optimizer", type=click.Choice(list(optimizer_name_map.keys())), required=True, help="Optimizer to use")
@click.option("--iterations", type=int, default=4, help="Maximum number of iterations")
@click.option('--n-samples', type=int, default=40, help='The number of samples to generate per task.')
@click.option('--min-correct', type=int, default=1,
              help='Minimum number of correct solutions for a given problem to even consider profiling it.')
@click.option('--max-profile', type=int, default=None, help='The number of profiles to get for each task.')
@click.option('--root', type=str, default=DEFAULT_ROOT, help='The root directory where results will be saved.')
@click.option('--nb-workers-wanted', type=int, default=None, help='Number of parallel workers for code generation.')
@click.option("--i-just-wanna-run", is_flag=True,
              help="Re-run the correctness and evaluation no matter if the result file already exists")
@click.option("--test-single-task", is_flag=True, help="Evaluate a single task. For debugging purposes.")
@click.option("--calibration-time", type=float, default=30,
              help="The time in second to calibrate energy measures between iterations.")
@click.option('--model', type=str, required=True, help='The model used for code generation.')
@click.option('--bs', type=int, default=100, help='Batch size for code generation.')
@click.option('--temperature', type=float, default=0.2, help='Temperature for sampling.')
@click.option('--backend', type=click.Choice(["openai"]), default="openai",
              help='The backend used for code generation.')
@click.option('--base-url', type=str, default=None, help='The base URL for the model.')
@click.option("--enforce-n-samples", type=bool, default=False,
              help="Enforce having _at most_ n_samples when evaluating correctness and performance.")
@click.option("--nb-workers-generation", type=int, default=50, help="The number of workers for the generation")
@click.option("--skip-iter", type=int, default=None, help="Iteration to skip")
@click.option("--skip-prev-config-check", is_flag=True)
@click.option("--force-n-samples-with-prev-results", is_flag=True,
              help="Forces the number of samples passed to the optimizers to be lesser or equal to n_samples")
def evaluate_optimizers(
        optimizer: str,
        iterations: Optional[int],
        n_samples: int,
        min_correct: int,
        max_profile: Optional[int],
        root: str,
        nb_workers_wanted: Optional[int],
        i_just_wanna_run: bool,
        test_single_task: bool,
        calibration_time: float,
        model: str,
        bs: int,
        temperature: float,
        backend: str,
        base_url: Optional[str],
        enforce_n_samples: bool,
        nb_workers_generation: int,
        skip_iter: Optional[int],
        skip_prev_config_check: bool,
        force_n_samples_with_prev_results
):
    """Evaluate optimizers and their energy consumptions."""
    max_profile = max_profile or n_samples  # Just to be sure
    if min_correct > 1:
        rich.print(
            f"[yellow]Caution, having a minimum of passing samples for a task to sample can lead to some not being passed down into the next optimization step[/]")
    if max_profile < n_samples:
        rich.print(
            f"[yellow]Caution, profiling less samples than initially generating. Pass@1_global will be unreliable[/]")
    simple_test_profiler()  # test linux perf setup

    if test_single_task and root == DEFAULT_ROOT:
        rich.print("[yellow]Caution, the result path is default when the config looks like a testing config[/]")

    model_config = ModelConfig(
        model=model,
        backend=backend,
        temperature=temperature,
        batch_size=bs,
        dataset="evalperf",  # Fixed parameter,
        base_url=base_url,
    )
    optimizer_model: OptimizerBase = optimizer_name_map[optimizer](model_config, nb_workers=nb_workers_generation)
    print(f"Optimizer: {optimizer_model}")

    max_workers = get_max_workers(nb_workers_wanted)

    config = {
        "optimizer": optimizer,
        "iterations": iterations,
        "n_samples": n_samples,
        "min_correct": min_correct,
        "max_profile": max_profile,
        "root": root,
        "nb_workers_wanted": nb_workers_wanted,
        "nb_workers": max_workers,
        "i_just_wanna_run": i_just_wanna_run,
        "test_single_task": test_single_task,
        "model": model,
        "bs": bs,
        "temperature": temperature,
        "backend": backend,
        "base_url": base_url,
        "samples_path": None,
        "accept_errored_samples_in_optimizer": optimizer_model.can_accept_errored_samples(),
        "enforce_n_samples": enforce_n_samples,
        "nb_workers_generation": nb_workers_generation,
        "force_n_samples_with_prev_results": force_n_samples_with_prev_results,
    }
    rich.print(config)

    # Data loading
    problems, expected_output = get_evalplus_data()
    ptasks_to_do = get_evalperf_data()

    dataset_dict = evalplus.codegen.get_dataset_dict("evalperf", None, None)
    if test_single_task:
        task_id_to_keep = "HumanEval/16"
        ptasks_to_do = {task_id_to_keep: ptasks_to_do[task_id_to_keep]}
        dataset_dict = {task_id_to_keep: dataset_dict[task_id_to_keep]}

    rule("EvalPerf Configurations")
    table_print(
        "Configurations",
        {
            "Max CPU": max_workers,
            "#Tasks": len(ptasks_to_do),
            "#Samples per task": n_samples,
            "Min correct": min_correct,
            "Max profile": max_profile,
        },
    )

    iter_current = 0
    prev_results_path = None
    all_results_paths = []
    while iter_current <= iterations:
        if skip_iter is not None and iter_current == skip_iter:
            print(f"Skipping iteration {iter_current}")
            identifier = get_result_file_identifier(backend, None, model, temperature)
            prev_results_path = build_result_paths(get_target_path("evalperf", identifier, True, root))[1]
            all_results_paths.append(prev_results_path)
            iter_current += 1
            continue
        ptasks = copy(ptasks_to_do)
        energy_measures = defaultdict(
            EnergyMeasurement)  # Init with empty measures so it's only addition after that. Also makes it easier to add previous partial runs if resuming.

        rich.print("Doing nothing before calibration...")
        time.sleep(calibration_time / 10)
        rich.print(f"Calibrating for {calibration_time} seconds...")
        energy_measures["idle"] = calibrate(
            calibration_time)  # We completely replace so as to not inflate the idle energy consumption
        rich.print(Rule(""))
        rich.print(Rule(f"Iteration n°{iter_current}/{iterations}"))
        t_gpu_phase_start = time.time()
        if iter_current <= 0:
            rich.print(f"Generating samples...")
            if optimizer_model.can_generate_samples_from_scratch():
                samples_path, generation_energy = monitor_energy_for_function(gpu=True,
                                                                              func=optimizer_model.generate_samples,
                                                                              dataset_dict=dataset_dict,
                                                                              n_samples=n_samples,
                                                                              root_path=root, )
            else:
                codegen_res: tuple[tuple[str, bool], EnergyMeasurement] = run_codegen(dataset="evalperf", model=model,
                                                                                      n_samples=n_samples,
                                                                                      temperature=temperature,
                                                                                      backend=backend,
                                                                                      base_url=base_url, bs=bs,
                                                                                      root=root, )
                (samples_path, _), generation_energy = codegen_res
            if generation_energy.duration > 30:
                energy_measures["generation"] += generation_energy
        else:
            rich.print(f"Optimizing previous samples...")
            assert prev_results_path is not None

            # Get results from previous iteration
            with open(prev_results_path, "r") as f:
                prev_results_raw = json.load(f)
            assert skip_prev_config_check or result_file_has_same_config(prev_results_raw, max_profile,
                                                                         min_correct, n_samples, temperature)
            prev_results: dict[str, TaskEvalResult] = prev_results_raw[
                "eval"]  # Redundant with the TaskEvalResult initialization
            for etask in prev_results:
                prev_results[etask] = TaskEvalResult(**prev_results_raw["eval"][etask])

            profiled_samples: dict[str, list[CodeEvalResult]] = defaultdict(list)
            for task in prev_results:
                if task not in ptasks.keys():
                    continue
                for code_eval_result in prev_results[task].results:
                    if not (
                            code_eval_result.passed and code_eval_result.profiled) and not optimizer_model.can_accept_errored_samples():
                        continue
                    if iter_current == 1 and force_n_samples_with_prev_results and len(
                            profiled_samples[
                                task]) >= n_samples:  # Only do it at the first optimization iteration. Is useful to allow less samples than the model originally generated
                        continue
                    else:
                        profiled_samples[task].append(code_eval_result)

            # Use the optimizers on them
            samples_path, generation_energy = monitor_energy_for_function(gpu=True,
                                                                          func=optimizer_model.optimize,
                                                                          samples_to_optimize=profiled_samples,
                                                                          dataset_dict=dataset_dict,
                                                                          iter_number=iter_current,
                                                                          root_path=root)
            if generation_energy.duration > 30:
                energy_measures["generation"] += generation_energy

        t_gpu_phase_end = time.time()

        rule("Correctness Checking...")
        config["samples_path"] = samples_path
        brief_result_path, result_path = build_result_paths(samples_path)
        prev_results_path = result_path
        all_results_paths.append(result_path)

        # Write sidecar as soon as result_path is known, before correctness/profiling run.
        gpu_phase_path = result_path.replace("_evalopti_results.json", "_gpu_phase.json")
        with open(gpu_phase_path, "w") as f:
            json.dump({
                "iteration": iter_current,
                "gpu_phase_start": t_gpu_phase_start,
                "gpu_phase_end": t_gpu_phase_end,
                "gpu_phase_duration_s": t_gpu_phase_end - t_gpu_phase_start,
            }, f)
        rich.print(f"GPU phase timing written to {gpu_phase_path} "
                   f"({t_gpu_phase_end - t_gpu_phase_start:.1f}s)")

        task_eval_results: dict[str, TaskEvalResult] = init_eval_results(ptasks)
        if not i_just_wanna_run and os.path.exists(result_path):
            task_eval_results, previous_energy_results = resume_results_if_possible(max_profile, min_correct, n_samples,
                                                                                    ptasks, result_path,
                                                                                    temperature, task_eval_results)
            for key in previous_energy_results:
                energy_measures[key] += previous_energy_results[key]
            rich.print(f"Energy: {energy_measures}")
        samples_to_evaluate = load_samples(n_samples, samples_path, enforce_n_samples)

        task_eval_results, correctness_energy = check_correctness(task_eval_results, expected_output,
                                                                  max_workers,
                                                                  n_samples, problems, ptasks,
                                                                  samples_to_evaluate)
        if correctness_energy.duration >= 30:
            energy_measures["correctness"] += correctness_energy
        rich.print(f"Energy: {energy_measures}")

        rule("Evaluation Start")
        rich.print(f"IDs of tasks to evaluate: {list(ptasks.keys())}")
        task_eval_results, profiling_energy = evaluate_performance_of_samples(task_eval_results, True,
                                                                              max_profile, max_workers,
                                                                              min_correct,
                                                                              ptasks)
        if profiling_energy.duration >= 30:
            energy_measures["profiling"] += profiling_energy

        print_and_save_results(brief_result_path, energy_measures, max_profile, min_correct, n_samples, result_path,
                               task_eval_results, temperature, config)

        iter_current += 1

    print_evalopti_summary(all_results_paths, iterations)


def print_evalopti_summary(all_results_paths, iterations):
    # Print summary of all iterations
    all_results = []
    for i in range(iterations + 1):
        with open(all_results_paths[i], "r") as f:
            results = json.load(f)
        results["eval"] = load_task_eval_results_from_json_dict(results)
        all_results.append(results)
    table = Table(
        title="EvalOpti Summary - End of all iterations",
        show_header=True,
        header_style="bold",
    )
    columns = ["Iteration", "DPS", "DPS_norm", "Pass@1", "Pass@1_global", "Energy_gen", "Energy_correct",
               "Energy_profiling"]
    for col_name in columns:
        table.add_column(col_name)
    for i in range(iterations):
        row_values = []
        row_values.append(f"{i}")
        row_values.append(f"{all_results[i]["summary"]["dps"]}")
        row_values.append(f"{all_results[i]["summary"]["dps_norm"]}")
        row_values.append(f"{all_results[i]["summary"]["pass@1"]}")
        row_values.append(f"{all_results[i]["summary"]["pass@1_global"]}")
        row_values.append(f"{all_results[i]["summary"]["energy_measures"]["generation"]["total_energy"]}")
        row_values.append(f"{all_results[i]["summary"]["energy_measures"]["correctness"]["total_energy"]}")
        row_values.append(f"{all_results[i]["summary"]["energy_measures"]["profiling"]["total_energy"]}")
        table.add_row(*row_values)
    rich.print(table)

    summary_path = all_results_paths[-1]  # We assume all_results is ordered as it is built one per iteration.
    iteration_identifier = f"_{iterations}_"
    if iteration_identifier in summary_path:
        summary_path = summary_path.replace(iteration_identifier, "_summary_")
    else:
        rich.print(f"[red]Could not build summary path from {summary_path} and {iteration_identifier}[/]")
        summary_path += "summary.json"
    # Save summary results
    final_result = all_results[-1]
    with open(summary_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "config": final_result["config"],
                    "summary": [{
                        "iteration": i,
                        "file": all_results_paths[i],
                        "dps": res["summary"]["dps"],
                        "dps_norm": res["summary"]["dps_norm"],
                        "pass@1": res["summary"]["pass@1"],
                        "pass@1_global": res["summary"]["pass@1_global"],
                        "energy_measures": res["summary"]["energy_measures"],
                    } for i, res in enumerate(all_results)],
                    "eval": {
                        task_id: [{
                            "dps": res.dps,
                            "dps_norm": res.dps_norm,
                            "pass@1": res.pass_1,
                            "pass@1_global": res.pass_1_global,
                            "profiled": [
                                {
                                    "solution": r.solution,
                                    "matching_cluster_idx": r.matching_cluster_idx,
                                }
                                for r in res.results
                                if r.profiled
                            ],
                        }]
                        for task_id, res in final_result["eval"].items()
                    },
                }
                , default=vars)
        )
    rich.print(f"Brief results have been saved to {summary_path}")


def result_file_has_same_config(prev_results, max_profile, min_correct, n_samples, temperature):
    return (prev_results["config"]["n_samples"] == n_samples
            and prev_results["config"]["temperature"] == temperature
            and prev_results["config"]["min_correct"] == min_correct
            and prev_results["config"]["max_profile"] == max_profile)


def print_and_save_results(brief_result_path, energy_measures: dict[str, EnergyMeasurement], max_profile, min_correct,
                           n_samples, result_path,
                           task_eval_results: dict[str, TaskEvalResult], temperature, config):
    rule("Evaluation Summary")
    dps = mean(not_none([res.dps for res in task_eval_results.values()], default=[0]))
    dps_norm = mean(not_none([res.dps_norm for res in task_eval_results.values()], default=[0]))
    pass_1 = mean(not_none([res.pass_1 for res in task_eval_results.values()], default=[0]))
    pass_1_global = mean(not_none([res.pass_1_global for res in task_eval_results.values()], default=[0]))
    n_evalperfed = len(not_none([res.dps for res in task_eval_results.values()]))
    table_print(
        "EvalPerf Summary",
        {
            "DPS": f"{dps:.1f}",
            "DPS_norm": f"{dps_norm:.1f}",
            "Pass@1": f"{pass_1:.1f}%",
            "Pass@1_global": f"{pass_1_global:.1f}%",
            "#EvalPerf-ed tasks": f"{n_evalperfed} / {len(task_eval_results)}",
            "min_correct": min_correct,
            "n_samples": n_samples,
            "temperature": temperature,
            "Energy_gen": energy_measures["generation"].total_energy,
            "Energy_correct": energy_measures["correctness"].total_energy,
            "Energy_profiling": energy_measures["profiling"].total_energy
        },
    )
    # Save full results
    with open(result_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "config": config,
                    "summary": {
                        "dps": dps,
                        "dps_norm": dps_norm,
                        "pass@1": pass_1,
                        "pass@1_global": pass_1_global,
                        "energy_measures": energy_measures,
                    },
                    "eval": task_eval_results,
                }
                , default=vars)
        )
    rich.print(f"Full results have been saved to {result_path}")
    # Save brief results
    with open(brief_result_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "config": {
                        "n_samples": n_samples,
                        "temperature": temperature,
                        "min_correct": min_correct,
                        "max_profile": max_profile,
                    },
                    "summary": {
                        "dps": dps,
                        "dps_norm": dps_norm,
                        "pass@1": pass_1,
                        "pass@1_global": pass_1_global,
                        "energy_measures": energy_measures,
                    },
                    "eval": {
                        task_id: {
                            "dps": res.dps,
                            "dps_norm": res.dps_norm,
                            "pass@1": res.pass_1,
                            "pass@1_global": res.pass_1_global,
                            "profiled": [
                                {
                                    "solution": r.solution,
                                    "matching_cluster_idx": r.matching_cluster_idx,
                                }
                                for r in res.results
                                if r.profiled
                            ],
                        }
                        for task_id, res in task_eval_results.items()
                    },
                }
                , default=vars)
        )
    rich.print(f"Brief results have been saved to {brief_result_path}")


def init_eval_results(ptasks):
    eval_results: dict[str, TaskEvalResult] = {}
    for task_id, ptask in ptasks.items():
        eval_results[task_id] = TaskEvalResult(
            task_id=task_id,
            ref=[
                {"solution": s, "score": r, "_num_cpu_instructions": None}
                for s, r in zip(ptask["reference"], ptask["scores"])
            ]
        )
    return eval_results


def load_samples(n_samples, samples_path: str, enforce_n_samples):
    sample_iter = stream_jsonl(samples_path)
    samples_to_evaluate = defaultdict(list)
    for task in sample_iter:
        samples_to_evaluate[task["task_id"].replace("_", "/")].append(
            (task["solution"], (task["sample_id"] if "sample_id" in task else None)))
    if enforce_n_samples:
        samples_to_evaluate = {k: v[:n_samples] for k, v in samples_to_evaluate.items()}
    nb_samples = sum([len(samples) for samples in samples_to_evaluate.values()])
    rich.print(f"Loaded {nb_samples} samples from {len(samples_to_evaluate)} tasks.")
    return samples_to_evaluate


def load_task_eval_results_from_json_dict(results) -> dict[str, TaskEvalResult]:
    results_dict = {}
    for task_id in results["eval"].keys():
        results_dict[task_id] = TaskEvalResult(**results["eval"][task_id])
    return results_dict


def resume_results_if_possible(max_profile, min_correct, n_samples, ptasks, result_path, temperature,
                               task_eval_results) -> tuple[dict[str, TaskEvalResult], dict[str, EnergyMeasurement]]:
    with open(result_path, "r") as f:
        resumed_results = json.load(f)
    if (
            resumed_results["config"]["n_samples"] == n_samples
            and resumed_results["config"]["temperature"] == temperature
            and resumed_results["config"]["min_correct"] == min_correct
            and resumed_results["config"]["max_profile"] == max_profile
    ):
        task_eval_results = load_task_eval_results_from_json_dict(resumed_results)
        for etask in task_eval_results:
            ptasks.pop(etask, None)

        rich.print(f"Resumed {len(task_eval_results)} results from {result_path}")
        energy_measures = {}
        for phase, measures in resumed_results["summary"]["energy_measures"].items():
            energy_measures[phase] = EnergyMeasurement(cpu_energy=measures["cpu_energy"],
                                                       gpu_energy=measures["gpu_energy"],
                                                       duration=measures["duration"],
                                                       perf_duration=measures["perf_duration"], )
        rich.print(f"Also resumed energy measures: {energy_measures}")
    else:
        rich.print(f"Resumed results from {result_path} have different configurations, not loading the results.")
        energy_measures = defaultdict(EnergyMeasurement)
    return task_eval_results, energy_measures


def build_result_paths(samples):
    assert samples.endswith(".jsonl")
    result_path = samples.replace(".jsonl", f"_evalopti_results.json")
    brief_result_path = result_path.replace(
        "evalopti_results.json", "evalopti_results.brief.json"
    )
    return brief_result_path, result_path


if __name__ == '__main__':
    evaluate_optimizers()