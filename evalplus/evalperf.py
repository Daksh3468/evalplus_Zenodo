"""Compute the Differential Performance Scores (DPS) and DPS_{norm} of given samples from a model.

Check our COLM paper for more details: https://www.arxiv.org/abs/2408.06450

^Updates from the COLM paper:
* Condition to activate efficiency evaluation for a task:
  * Paper: as long as you have at least one correct solution, and we select up to 10 correct solutions for efficiency sampling
  * Here: you need to have at least `min_correct` correct solutions, and we evaluate the efficiency of all correct solutions
  * Updating rationale: to make the evaluation more statistically robust

@inproceedings{liu2024evaluating,
  title = {Evaluating Language Models for Efficient Code Generation},
  author = {Liu, Jiawei and Xie, Songrun and Wang, Junhao and Wei, Yuxiang and Ding, Yifeng and Zhang, Lingming},
  booktitle = {First Conference on Language Modeling},
  year = {2024},
  url = {https://openreview.net/forum?id=IBCBMeAhmC},
}
"""

import json
import multiprocessing
import os
import socket
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean
from typing import Dict, List, Optional, Tuple

import rich
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from evalplus.codegen import run_codegen
from evalplus.config import *
from evalplus.config import PERF_EVAL_TIMEOUT_SECOND
from evalplus.data import (
    get_evalperf_data,
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
)
from evalplus.data.mbpp import mbpp_deserialize_inputs
from evalplus.data.utils import stream_jsonl
from evalplus.eval import PASS, untrusted_check
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.evaluate import get_groundtruth
from evalplus.perf.energy import measure_energy, EnergyMeasurement
from evalplus.perf.profile import (
    are_profiles_broken,
    default_parallelism,
    simple_test_profiler,
    profile_num_inst_and_time, profile_time
)
from evalplus.utils import progress


def rule(msg: str):
    rich.print(Rule(msg))


def not_none(l: list, default=None) -> list:
    if default is None:
        default = list()
    not_none_list = [x for x in l if x is not None]
    if len(not_none_list) == 0:
        return default
    return not_none_list


def get_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@dataclass
class ErrorDetails:
    whole: Optional[str]
    case: Optional[str]

    def has_error(self) -> bool:
        return self.whole is not None or self.case is not None

    def has_test_case_errors(self) -> bool:
        return self.case is not None


@dataclass
class CodeEvalResult:
    solution: str
    passed: bool
    sample_id: Optional[str] = None
    error_details: Optional[ErrorDetails] = None
    profiled: bool = False
    matching_cluster_idx: Optional[int] = None
    dps: Optional[float] = None
    dps_norm: Optional[float] = None
    num_cpu_instructions: Optional[float] = None
    cpu_time: Optional[float] = None

    def __post_init__(self):
        if self.error_details is not None and isinstance(self.error_details, dict):
            self.error_details = ErrorDetails(**self.error_details)
        assert self.error_details is None or isinstance(self.error_details,
                                                        ErrorDetails), f"ErrorDetails was incorrect: str(self.error_details)"

    def has_errors(self) -> bool:
        return self.error_details is not None and self.error_details.has_error()


@dataclass
class CodeEvalResultEnergy(CodeEvalResult):
    energy_measures: Optional[EnergyMeasurement] = None
    nb_energy_profiled: Optional[int] = None

    @staticmethod
    def from_code_eval_result(eval_result: CodeEvalResult):
        if isinstance(eval_result, CodeEvalResultEnergy):
            return eval_result
        return CodeEvalResultEnergy(**eval_result.__dict__)


@dataclass
class TaskEvalResult:
    task_id: str
    results: List[CodeEvalResult] = field(default_factory=list)
    ref: List[Dict] = field(default_factory=list)
    dps: Optional[float] = None
    dps_norm: Optional[float] = None
    pass_1: Optional[float] = None
    pass_1_global: Optional[float] = None
    n_profiled: Optional[int] = None
    num_cpu_instructions: Optional[float] = None
    cpu_time: Optional[float] = None

    def __post_init__(self):
        # If results is a list of dict, convert results to a list of CodeEvalResults
        if isinstance(self.results, list) and all(
                isinstance(r, dict) for r in self.results
        ):
            self.results = [
                CodeEvalResult(**r) if isinstance(r, dict) else r for r in self.results
            ]


@dataclass
class TaskEvalResultEnergy(TaskEvalResult):
    energy_measures: Optional[EnergyMeasurement] = None
    nb_tests: Optional[int] = None
    total_profiling_time: Optional[float] = None
    nb_energy_profiled: int = 0
    nb_repeats: int = 0
    nb_concurrent_workers: int = 0

    @staticmethod
    def from_task_eval_result(eval_result: TaskEvalResult):
        if isinstance(eval_result, TaskEvalResultEnergy):
            return eval_result
        return TaskEvalResultEnergy(**eval_result.__dict__)


def correctness_check(
        solution: str, dataset: str, task: Dict, expected_output: List
) -> Tuple:
    assert isinstance(solution, str)
    result = untrusted_check(
        dataset,
        solution,
        task["base_input"] + list(task["plus_input"]),
        task["entry_point"],
        expected_output["base"] + expected_output["plus"],
        task["atol"],
        expected_output["base_time"] + expected_output["plus_time"],
        fast_check=True,
        min_time_limit=DEFAULT_MIN_TIME_LIMIT,
        gt_time_limit_factor=DEFAULT_GT_TIME_LIMIT_FACTOR,
    )
    return result, solution


def get_evalplus_data():
    problems_he = get_human_eval_plus(noextreme=True)
    dataset_hash = get_human_eval_plus_hash(noextreme=True)
    expected_output_human = get_groundtruth(problems_he, dataset_hash, [])
    problems_mbpp = get_mbpp_plus(noextreme=True)
    dataset_hash = get_mbpp_plus_hash(noextreme=True)
    expected_output_mbpp = get_groundtruth(
        problems_mbpp,
        dataset_hash,
        MBPP_OUTPUT_NOT_NONE_TASKS,
    )
    problems = {**problems_he, **problems_mbpp}
    expected_output = {**expected_output_human, **expected_output_mbpp}
    return problems, expected_output


def table_print(table_name: str, kv: Dict):
    table = Table(
        title=table_name,
        show_header=True,
        header_style="bold",
    )
    for col_name in kv:
        table.add_column(col_name)

    table.add_row(*[str(v) for v in kv.values()])
    rich.print(table)


def correctness_worker(task_id: str, samples: list[tuple[str, str]], ctask: Dict, expected_output: Dict):
    assert isinstance(
        samples, list
    ), f"{task_id}: samples is not a list but {type(samples)}"

    results: list[CodeEvalResult] = []

    for solution, sample_id in samples:
        result, solution = correctness_check(
            solution, task_id.split("/")[0].lower(), ctask, expected_output
        )
        results.append(CodeEvalResult(
            solution=solution,
            sample_id=sample_id,
            passed=result[0] == PASS,
        ))
        if not results[-1].passed:
            error_details = ErrorDetails(case=result[2]["case"], whole=result[2]["whole"])
            # rich.print(f"{task_id}/{solution}: {result[0]} ({result[1]})\nException : {error_details}")
            results[-1].error_details = error_details

    return task_id, results


def perf_worker(
        task_id: str,
        ptask: Dict,  # EvalPerf data
        ret_dict: TaskEvalResult,
        lazy_evaluation: bool,
        max_profile: int,
):
    rich.print(f"{task_id}: Started")
    start_time = time.time()

    ######################### Profiling Setup #########################
    n_reference = len(ptask["reference"])
    entry_point = ptask["entry_point"]
    pe_input = (
        mbpp_deserialize_inputs(task_id, ptask["pe_input"])[0]
        if task_id.startswith("Mbpp/")
        else ptask["pe_input"][0]
    )
    ####################################################################

    cache_ref_num_inst = [None] * n_reference  # The number of instructions executed by each reference solution

    def get_avg_ref_profile(idx, check_order=True) -> Optional[Tuple]:
        nonlocal cache_ref_num_inst

        # Check the function is not called out of order
        if check_order:
            assert (
                    idx < n_reference - 1
                    and cache_ref_num_inst[idx + 1] is not None
                    or idx == n_reference - 1
            ), f"Calling get_avg_ref_profile({idx}) before get_avg_ref_profile({idx + 1}) is called, is not allowed! {n_reference = }"

        if cache_ref_num_inst[idx] is not None:
            return cache_ref_num_inst[idx], ptask["scores"][idx]

        evaluation_time = PERF_EVAL_TIMEOUT_SECOND
        ref_solution = ptask["reference"][idx]
        for _ in range(2):  # at most retry twice
            profiles = profile_num_inst_and_time(
                ref_solution,
                entry_point,
                [pe_input],
                timeout_second_per_test=evaluation_time,
            )

            # Bad thing#1: timeout / failure happens
            if are_profiles_broken(profiles[0]):
                print(f"{task_id}: [WARNING] Error in ref: {profiles}")
                rich.print(Syntax(ref_solution, "python"))
                print(f"{task_id}: Retrying w/ +10s timeout...")
                evaluation_time += 10
            else:
                break

        if are_profiles_broken(profiles[0]):
            rich.print(f"[bold red]{task_id}: [ERROR] Failed to profile ref #{idx}: {profiles}[/]")
            rich.print(Syntax(ref_solution, "python"))
            return None

        avg_profile = mean(profiles[0])
        try:
            avg_profile_time = mean(profiles[1])
        except TypeError as e:
            rich.print(f"[bold red]{task_id}: [ERROR] Failed to profile time, returned time was {profiles[1]}: {e}[/]")
            avg_profile_time = 0
        # Bad thing#2: if the current #instruction is faster than that of i+1
        if idx < n_reference - 1 and cache_ref_num_inst[idx + 1] is not None and avg_profile < cache_ref_num_inst[
            idx + 1]:
            print(f"{task_id}: [WARNING] #{idx} ref faster than #{idx + 1}")
            print(f"ref {idx}: #inst {avg_profile}\tscore {ptask['scores'][idx]:.1f}")
            print(
                f"ref {idx + 1}: #inst {cache_ref_num_inst[idx + 1]}\tscore {ptask['scores'][idx + 1]:.1f}"
            )
            # rich.print(Syntax(ref_solution, "python"))
            if check_order:  # I think it is used to verify the order of the solutions? You can keep check_order on False for normal evaluation
                return None

        cache_ref_num_inst[idx] = avg_profile
        ret_dict.ref[idx]["_num_cpu_instructions"] = avg_profile
        ret_dict.ref[idx]["cpu_time"] = avg_profile_time
        return cache_ref_num_inst[idx], ptask["scores"][idx]

    if not lazy_evaluation:  # compute everything ahead of time
        rich.print(f"{task_id}: Eagerly profiling all references")
        for i in range(n_reference - 1, -1, -1):
            if get_avg_ref_profile(i, check_order=False) is None:
                break

        if None in cache_ref_num_inst:
            rich.print(f"[bold red][ERROR]{task_id}: Failed to profile certain reference: {cache_ref_num_inst = }[/]")

    profile_cache = {}

    cur_profiled = 0
    for result in ret_dict.results:
        if cur_profiled >= max_profile:
            rich.print(f"{task_id}: Reached max_profile limit {max_profile}, stopped")
            break
        # Skip results that did not pass
        if not result.passed:
            continue

        solution = result.solution

        if solution in profile_cache:  # reuse cache
            sample_profiles = profile_cache[solution]
        else:
            sample_profiles: Tuple[List, List] = profile_num_inst_and_time(
                solution,
                entry_point,
                [pe_input],
                timeout_second_per_test=PERF_EVAL_TIMEOUT_SECOND,
            )
            profile_cache[solution] = sample_profiles  # store cache

        score = 0
        norm_score = 0
        result.matching_cluster_idx = -1  # -1 means even slower than the slowest ref
        # if the solution results in a timeout, score is 0
        if are_profiles_broken(sample_profiles[0]) or type(sample_profiles[1]) == str:
            print(
                f"{task_id}: Tested solution error'ed out: {sample_profiles} ... regarded as 0 score"
            )
            rich.print(Syntax(solution, "python"))
        else:
            avg_sample_profile = result.num_cpu_instructions = mean(sample_profiles[0])
            result.cpu_time = mean(sample_profiles[1])
            # Get profiles from fast to slow (back to front):
            for j in range(n_reference - 1, -1, -1):
                res = get_avg_ref_profile(j, check_order=False)
                if res is None:
                    rich.print(f"[yellow]{task_id}: [WARNING] as ref #{j} could not be profiled, skipped[/]")
                    continue
                avg_ref_profile, ref_score = res
                if avg_sample_profile <= avg_ref_profile:
                    result.matching_cluster_idx = j
                    score = ref_score
                    norm_score = 100 * (j + 1) / n_reference
                    break

        result.dps = score
        result.dps_norm = norm_score
        result.profiled = True
        cur_profiled += 1

    ret_dict.dps = mean(not_none([r.dps for r in ret_dict.results], default=[0]))
    ret_dict.dps_norm = mean(not_none([r.dps_norm for r in ret_dict.results], default=[0]))
    ret_dict.n_profiled = cur_profiled
    ret_dict.num_cpu_instructions = mean(
        not_none([r.num_cpu_instructions for r in ret_dict.results], default=[0])
    )
    ret_dict.cpu_time = mean(
        not_none([r.cpu_time for r in ret_dict.results], default=[0])
    )

    # table_print(
    #     f"[bold green]{task_id} Completed[/]",
    #     {
    #         "Duration": f"{time.time() - start_time:.1f}s",
    #         "DPS": f"[green]{ret_dict.dps:.1f}[/]",
    #         "DPS_norm": f"[green]{ret_dict.dps_norm:.1f}[/]",
    #         "# Profiled": f"{cur_profiled} / {len(ret_dict.results)}",
    #         "Pass@1": f"{ret_dict.pass_1:.1f}%",
    #         "Num CPU instructions": f"{ret_dict.num_cpu_instructions:.1f}",
    #         "Avg CPU time": f"{ret_dict.cpu_time:.1f}s",
    #     },
    # )

    return ret_dict


def energy_worker(
        task_id: str,
        ptask: Dict,  # EvalPerf data
        ret_dict: TaskEvalResultEnergy,
        max_profile: int,
        min_duration: float,
) -> TaskEvalResultEnergy:
    rich.print(f"{task_id}: Started")
    ######################### Profiling Setup #########################
    entry_point = ptask["entry_point"]
    pe_input = (
        mbpp_deserialize_inputs(task_id, ptask["pe_input"])[0]
        if task_id.startswith("Mbpp/")
        else ptask["pe_input"][0]
    )
    ####################################################################
    min_duration = max(min_duration, 0.1)

    cur_profiled = 0
    profiling_start_time = time.time()
    energy = EnergyMeasurement()
    errs = 0
    for i in range(len(ret_dict.results)):
        result = ret_dict.results[i]
        new_result = CodeEvalResultEnergy.from_code_eval_result(result)
        duration = 0
        sample_energy = EnergyMeasurement()
        nb_energy_profiled = 0
        while duration < min_duration:
            sample_start_time = time.time()
            if cur_profiled >= max_profile:
                print(f"{task_id}: Reached max_profile limit {max_profile}, stopped")
                break
            # Skip results that were not already profiled
            if not new_result.profiled:
                continue

            res = profile_time(
                new_result.solution,
                entry_point,
                [pe_input],
                timeout_second_per_test=PERF_EVAL_TIMEOUT_SECOND,
                min_duration=min_duration,  # TODO: Remove
            )
            if res is None or len(res) != 2:
                print(f"{task_id}: Reached unexpected result {res}")
                errs += 1  # Might be a timeout
                if errs > 3:
                    rich.print(f"[red]Too many errors on task {task_id}, skipping sample[/]")
                    break
                continue
            sample_profile, extra_data = res

            new_result.matching_cluster_idx = -1  # -1 means even slower than the slowest ref
            # if the solution results in a timeout, score is 0
            if are_profiles_broken([sample_profile]) or extra_data == {}:
                print(
                    f"{task_id}: Tested solution error'ed out: {sample_profile} ... regarded as 0 score"
                )
            else:
                nb_energy_profiled += extra_data["nb_executions"]
                ret_dict.nb_energy_profiled += extra_data["nb_executions"]
                sample_energy += extra_data["energy"]
            duration += time.time() - sample_start_time
        new_result.energy_measures = sample_energy
        new_result.nb_energy_profiled = nb_energy_profiled
        ret_dict.results[i] = new_result
        cur_profiled += 1
        energy += sample_energy

    profiling_time = time.time() - profiling_start_time

    ret_dict.nb_tests = len(pe_input)
    ret_dict.energy_measures = energy
    ret_dict.cpu_time = ret_dict.cpu_time = mean(not_none([r.cpu_time for r in ret_dict.results], default=[-1]))
    ret_dict.total_profiling_time = profiling_time

    # table_print(
    #     f"[bold green]{task_id} Completed[/]",
    #     {
    #         "Duration (measure vs total)": f"{ret_dict.energy_measures.duration:.2f}s vs {profiling_time}s",
    #         "Duration - Perf": f"{ret_dict.energy_measures.perf_duration:.2f}",
    #         "# Profiled": f"{cur_profiled} / {len(ret_dict.results)}",
    #         "# Executions": f"{ret_dict.nb_energy_profiled}",
    #         "# Tests": f"{len(pe_input)}",
    #         "Avg CPU time": f"{ret_dict.cpu_time:.3f}s",
    #         "Energy required": f"{ret_dict.energy_measures}"
    #     },
    # )

    return ret_dict


# TODO(@ganler): OPTIMIZATION: reuse the samples from the generations of other datasets

def script(
        samples: Optional[str] = None,  # The path to the samples
        min_correct: int = 10,  # Minimum number of correct solutions for a given problem to even consider profiling it
        max_profile: Optional[int] = None,  # The number of profiles to get for each task
        n_samples: int = 100,  # Number of generated solutions for a given problem
        temperature: float = 1.0,
        parallel: Optional[int] = None,  # Number of workers
        lazy_evaluation: bool = True,  # See `perf_worker`
        i_just_wanna_run: bool = False,
        # Re-run the correctness and evaluation no matter if the result file already exists
        **model_kwargs,
):
    max_profile = max_profile or min(min_correct * 2, n_samples)
    assert min_correct <= max_profile <= n_samples
    try:
        simple_test_profiler()
    except Exception as e:
        print(f"[WARNING] Skipping profiler check: {e}")  # test linux perf setup

    energy_measures = {}

    # Generate code if model kwargs exist
    if model_kwargs:
        rule("Generating samples...")
        # To suppress the warning of tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false"
        )
        # overwrite parameters
        (samples, _), energy_measures["generation"] = run_codegen(
            dataset="evalperf",
            n_samples=n_samples,
            temperature=temperature,
            **model_kwargs,
        )

    assert samples is not None, "Please provide the path to the samples"

    # Data loading
    problems, expected_output = get_evalplus_data()
    ptasks = get_evalperf_data()

    # Filter ptasks to only keep one, useful for debugging
    # task_id_to_keep = "HumanEval/16"
    # ptasks = {task_id_to_keep: ptasks[task_id_to_keep]}

    max_workers = get_max_workers(parallel)
    brief_result_path, result_path = build_result_path(samples)

    # Resume results if some part of them already exist
    eval_results: dict[str, TaskEvalResult] = {}
    if not i_just_wanna_run and os.path.exists(result_path):
        resumed_result = json.load(open(result_path, "r"))
        if (
                resumed_result["config"]["n_samples"] == n_samples
                and resumed_result["config"]["temperature"] == temperature
                and resumed_result["config"]["min_correct"] == min_correct
                and resumed_result["config"]["max_profile"] == max_profile
        ):
            eval_results = resumed_result["eval"]  # Redundant with the TaskEvalResult initialization
            for etask in eval_results:
                ptasks.pop(etask, None)
                eval_results[etask] = TaskEvalResult(**resumed_result["eval"][etask])

            rich.print(f"Resumed {len(eval_results)} results from {result_path}")
        else:
            rich.print(f"Resumed results from {result_path} have different configurations, not loading the results.")

    # Load model's samples into a dict: task_id -> a list of samples
    sample_iter = stream_jsonl(samples)
    samples = defaultdict(list)
    for task in sample_iter:
        samples[task["task_id"].replace("_", "/")].append(
            (task["solution"], (task["sample_id"] if "sample_id" in task else None)))
    samples = {k: v[:n_samples] for k, v in samples.items()}

    # assert each task has n_samples
    for task_id, s in samples.items():
        assert len(s) == n_samples, f"{task_id} has {len(s)} samples != {n_samples}"

    # Initialize eval_results for each task
    for task_id, ptask in ptasks.items():
        eval_results[task_id] = TaskEvalResult(
            task_id=task_id,
            ref=[
                {"solution": s, "score": r, "_num_cpu_instructions": None}
                for s, r in zip(ptask["reference"], ptask["scores"])
            ]
        )

    rule("Correctness Checking...")
    eval_results, energy_measures["correctness"] = check_correctness(eval_results, expected_output, max_workers,
                                                                     n_samples, problems, ptasks, samples)

    rule("EvalPerf Configurations")
    if lazy_evaluation:
        rich.print(
            "[bold yellow]Lazy evaluation is enabled[/]: "
            "Fast evaluation without enumeratively checking reference order consistency."
        )

    table_print(
        "Configurations",
        {
            "Max CPU": max_workers,
            "#Tasks": len(ptasks),
            "#Samples per task": n_samples,
            "Min correct": min_correct,
            "Max profile": max_profile,
            "Result path": result_path,
        },
    )

    rich.print(f"IDs of tasks to evaluate: {list(ptasks.keys())}")
    rule("Evaluation Start")
    eval_results, energy_measures["profiling"] = evaluate_performance_of_samples(eval_results, lazy_evaluation,
                                                                                 max_profile, max_workers, min_correct,
                                                                                 ptasks)

    rule("Evaluation Summary")
    dps = mean(not_none([res.dps for res in eval_results.values()]))
    dps_norm = mean(not_none([res.dps_norm for res in eval_results.values()]))
    pass_1 = mean(not_none([res.pass_1 for res in eval_results.values()]))
    n_evalperfed = len(not_none([res.dps for res in eval_results.values()]))

    table_print(
        "EvalPerf Summary",
        {
            "DPS": f"{dps:.1f}",
            "DPS_norm": f"{dps_norm:.1f}",
            "Pass@1": f"{pass_1:.1f}%",
            "#EvalPerf-ed tasks": f"{n_evalperfed} / {len(eval_results)}",
            "min_correct": min_correct,
            "n_samples": n_samples,
            "temperature": temperature,
        },
    )

    # Save full results
    with open(result_path, "w") as f:
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
                        "energy_measures": energy_measures,
                    },
                    "eval": eval_results,
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
                        "energy_measures": energy_measures,
                    },
                    "eval": {
                        task_id: {
                            "dps": res.dps,
                            "dps_norm": res.dps_norm,
                            "pass@1": res.pass_1,
                            "profiled": [
                                {
                                    "solution": r.solution,
                                    "matching_cluster_idx": r.matching_cluster_idx,
                                }
                                for r in res.results
                                if r.profiled
                            ],
                        }
                        for task_id, res in eval_results.items()
                    },
                }
                , default=vars)
        )

    rich.print(f"Brief results have been saved to {brief_result_path}")

    rule("To visualize win-rates and pair-wise DPS, run:")
    rich.print(
        Syntax(
            f"""\
git clone git@github.com:evalplus/evalplus.github.io.git
git --git-dir=evalplus.github.io/.git pull
cp {brief_result_path} evalplus.github.io/results/evalperf
python evalplus.github.io/results/evalperf/stats.py
python -m http.server -d evalplus.github.io {get_free_port()}""",
            "bash",
        )
    )


def build_result_path(samples):
    # Build result_path
    if os.path.isdir(samples):
        result_path = os.path.join(samples, "evalperf_results.json")
    else:
        assert samples.endswith(".jsonl")
        result_path = samples.replace(".jsonl", "_evalperf_results.json")
    brief_result_path = result_path.replace(
        "evalperf_results.json", "evalperf_results.brief.json"
    )
    return brief_result_path, result_path


def get_max_workers(parallel):
    # Parallelism
    max_workers = parallel or max(1, default_parallelism(divisor=4))
    assert 0 < max_workers < multiprocessing.cpu_count(), "Invalid max CPU workers"
    print(f"Running with {max_workers} workers")
    return max_workers


@measure_energy(gpu=False)
def evaluate_performance_of_samples(eval_results: dict[str, TaskEvalResult], lazy_evaluation, max_profile, max_workers,
                                    min_correct, ptasks):
    undone = []
    with progress("Profiling") as p:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task_id, ptask in ptasks.items():
                n_pass = len([r for r in eval_results[task_id].results if r.passed])
                if n_pass < min_correct:
                    rich.print(
                        f"{task_id}: [bold yellow]{n_pass} < {min_correct} correct solutions, skipped[/]"
                    )
                    continue
                futures.append(
                    executor.submit(
                        perf_worker,
                        task_id,
                        ptask,
                        eval_results[task_id],
                        lazy_evaluation,
                        max_profile,
                    )
                )
                undone.append(task_id)
                rich.print(f"{task_id}: Queued")

            for future in p.track(as_completed(futures), total=len(futures)):
                if future.exception():
                    e: Exception = future.exception()
                    p.print("[red]Error while optimizing sample: [/red]", e, traceback.format_exc())
                    continue
                result: TaskEvalResult = future.result()
                eval_results[result.task_id] = result
                undone.remove(result.task_id)
                if undone and len(undone) < max_workers:
                    print(f"Still running: {undone}")
    return eval_results


@measure_energy(gpu=False)
def evaluate_energy_of_samples(eval_results: dict[str, TaskEvalResultEnergy], max_profile, max_workers,
                               min_correct, ptasks, min_duration):
    undone = []
    if max_workers > 1:
        rich.print("[yellow]Caution, profiling energy of multiple results at a time. Results may be inaccurate[/]")
    if len(eval_results) == 0:
        rich.print("[yellow]No results to evaluate[/]")
        return eval_results
    if len(ptasks) == 0:
        rich.print("[yellow]No task given to profile[/]")
        return eval_results
    with progress("Profiling") as p:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task_id, ptask in ptasks.items():
                n_pass = len([r for r in eval_results[task_id].results if r.passed])
                if n_pass < min_correct:
                    rich.print(
                        f"{task_id}: [bold yellow]{n_pass} < {min_correct} correct solutions, skipped[/]"
                    )
                    continue
                futures.append(
                    executor.submit(
                        energy_worker,
                        task_id,
                        ptask,
                        eval_results[task_id],
                        max_profile,
                        min_duration,
                    )
                )
                undone.append(task_id)
                rich.print(f"{task_id}: Queued")

            for future in p.track(as_completed(futures), total=len(futures)):
                result: TaskEvalResultEnergy = future.result()
                undone.remove(result.task_id)
                result.nb_concurrent_workers = min(max_workers, len(undone) + 1)
                eval_results[result.task_id] = result
                if undone and len(undone) < max_workers:
                    print(f"Still running: {undone}")
    return eval_results


@measure_energy(gpu=False)
def check_correctness(eval_results: dict[str, TaskEvalResult], expected_output, max_workers, n_samples, problems,
                      ptasks, samples):
    with progress("Correctness") as p:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    correctness_worker,
                    task_id,
                    samples[task_id],
                    problems[task_id],
                    expected_output[task_id],
                )
                for task_id in ptasks if samples.get(task_id)
            ]

            for future in p.track(as_completed(futures), total=len(futures)):
                task_id, results = future.result()
                eval_results[task_id].results = results
                eval_results[task_id].pass_1 = (
                        100 * len([r for r in results if r.passed]) / len(eval_results[task_id].results)
                )
                eval_results[task_id].pass_1_global = (
                        100 * len([r for r in results if r.passed]) / n_samples
                )
    return eval_results


def main():
    from fire import Fire

    Fire(script)


if __name__ == "__main__":
    main()
