import concurrent.futures
import dataclasses
import random
import traceback
import uuid
from collections import defaultdict
from typing import Optional

import rich
from datasets.utils.py_utils import asdict
from pydantic import BaseModel
from pydantic_core import ValidationError
from typing_extensions import override

from evalplus.codegen import save_generation_jsonl
from evalplus.data.utils import stream_jsonl
from evalplus.evalperf import CodeEvalResult
from evalplus.perf.optimizer import optimizer
from evalplus.perf.optimizer.optimizer import OptimizerBase, are_all_tasks_complete, GenerationResult, \
    get_execution_feedback
from evalplus.provider import ModelConfig
from evalplus.sanitize import sanitize
from evalplus.utils import progress


class HeuristicGeneration(BaseModel):
    reflexion: Optional[str]
    description: str
    code: str


@dataclasses.dataclass
class Heuristic:
    description: str
    code: str
    task_id: str
    sample_id: str
    batch_id: str
    fitness: float = 0

    def __repr__(self):
        return f"{self.description}\n```\n{self.code}\n```"

    def __str__(self):
        return self.__repr__()


_task_presentation = """Here is a task:
```
{task_description}
```
"""

_json_only_trigger = """
Your goal is to produce code that solves the task in the most efficient manner, saving time and energy resources.
You will write your answer in JSON in the following format:
{{"reflexion": "...", "description": "...", "code": "..."}}
You can use the `reflexion` field to think before writing the heuristic description and code.
"""

_trigger = """Firstly, describe your new heuristic and main steps in one sentence. Next, implement it in Python as described in the task above.""" + _json_only_trigger

heuristic_initialization_prompt = _task_presentation + """I want you to design a new heuristic to solve this task.
""" + _trigger

heuristic_e1_prompt = _task_presentation + """I have {nb_algo} existing algorithms with their codes as follows: 
{heuristics_str}

Please help me create a new algorithm that has a totally different form from the given ones.
""" + _trigger

heuristic_e2_prompt = _task_presentation + """I have {nb_algo} existing algorithms with their codes as follows: 
{heuristics_str}

Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. Thirdly, implement it in Python as described in the task above.
""" + _json_only_trigger

heuristic_m1_prompt = _task_presentation + """I have one algorithm with its code as follows.
Heuristic: {heuristic_str}
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.
""" + _trigger

heuristic_m2_prompt = _task_presentation + """I have one algorithm with its code as follows.
Heuristic: {heuristic_str}
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.
""" + _trigger

heuristic_m3_prompt = _task_presentation + """First, you need to identify the main components in the function below. Next, analyze whether any of these components can be overfit to the in-distribution instances. Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged.
Heuristic: {heuristic_str}
""" + _trigger

heuristic_fix_prompt = _task_presentation + """The following heuristic did not pass the tests:
Heuristic: {heuristic_str}

Here are some execution feedback on the heuristics:
{exec_feedback}

Fix the heuristic so that it passes the tests.
""" + _trigger


class OptimizerEoH(OptimizerBase):
    def __init__(self, model_config: ModelConfig, nb_workers: int):
        super().__init__(model_config, nb_workers)
        self.strict_result_file_verification = True

    POPULATION_SIZE = 10  # Arbitrary
    NB_PARENTS = 3  # Arbitrary

    def can_accept_errored_samples(self) -> bool:
        """Whether the optimizer should be passed only correct samples."""
        return True

    def can_generate_samples_from_scratch(self):
        return True

    @override
    def generate_samples(self, dataset_dict: dict, n_samples: int, root_path: str):
        """To call in the first iteration"""
        identifier = self.get_result_file_identifier(0)
        target_path = optimizer.get_target_path("evalperf", identifier, True, root=root_path)
        raw_target_path = target_path.replace(".jsonl", ".raw.jsonl")
        if are_all_tasks_complete(dataset_dict.keys(), target_path):
            rich.print(f"Tasks in {target_path} were already complete, skipping generation.")
            return target_path
        self._population_initialization(dataset_dict, num_batches=n_samples,
                                        population_size=OptimizerEoH.POPULATION_SIZE, target_path=target_path,
                                        raw_target_path=raw_target_path)
        return target_path

    @override
    def _do_optimize_codes(self, dataset_dict, raw_target_path, samples_to_optimize: dict[str, list[CodeEvalResult]],
                           target_path, iter_number):
        previous_identifier = self.get_result_file_identifier(iter_number - 1)
        identifier = self.get_result_file_identifier(iter_number)
        previous_target_path = target_path.replace(identifier, previous_identifier)
        all_heuristics: dict[str, dict[str, list[Heuristic]]] = defaultdict(lambda: defaultdict(list))
        sample_id_to_code_eval_result = {}
        for task_id, samples in samples_to_optimize.items():
            for sample in samples:
                sample_id_to_code_eval_result[sample.sample_id] = sample
        for previous_result in stream_jsonl(previous_target_path):
            heuristic = Heuristic(**previous_result["extra_data"]["heuristic"])
            all_heuristics[heuristic.task_id][heuristic.batch_id].append(heuristic)
        tasks = []
        for task_id in all_heuristics.keys():
            for batch_id in all_heuristics[task_id].keys():
                heuristics = all_heuristics[task_id][batch_id]
                associated_samples: dict[str, CodeEvalResult] = {
                    heur.sample_id: sample_id_to_code_eval_result[heur.sample_id] for heur in heuristics}
                tasks.append((task_id, batch_id, heuristics, associated_samples))
        self.load_model()
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(self.nb_workers // 5)) as executor:
            future_list = []
            for task_id, batch_id, heuristics, samples in tasks:
                future_list.append(executor.submit(self._evolve_heuristics, task_id=task_id,
                                                   task_description=dataset_dict[task_id]["prompt"].strip() + "\n",
                                                   population_size=OptimizerEoH.POPULATION_SIZE, batch_id=batch_id,
                                                   heuristics=heuristics, samples_by_id=samples))
            print("\nAdded all tasks")
            with progress(f"Optimizing - {self.get_optimizer_identifier()}") as prog_bar:
                for future in prog_bar.track(concurrent.futures.as_completed(future_list), total=len(tasks)):
                    if future.exception():
                        e: Exception = future.exception()
                        prog_bar.print("[red]Error while optimizing sample: [/red]", e, traceback.format_exc())
                        continue
                    results: list[GenerationResult] = future.result()
                    for result in results:
                        sanitized_solution = sanitize(result.solution,
                                                      entrypoint=dataset_dict[result.task_id]["entry_point"])
                        save_generation_jsonl(target_path, raw_target_path, sanitized_solution, result.solution,
                                              result.task_id,
                                              usage_share=result.usage,
                                              sample_id=result.sample_id, extra_data=result.extra_data_to_save)
        self.unload_model()
        return all_heuristics

    def get_optimizer_identifier(self):
        return "eoh"

    def _evolve_heuristics(self, task_id, batch_id, task_description, population_size, heuristics: list[Heuristic],
                           samples_by_id: dict[str, CodeEvalResult]) -> list[GenerationResult]:
        try:
            # Remove heuristics that did not pass or profile
            heuristics_passed = [heur for heur in heuristics if
                                 samples_by_id[heur.sample_id].profiled and samples_by_id[
                                     heur.sample_id].cpu_time is not None]
            # Associate fitness to heuristics and rank them according to fitness
            for heur in heuristics_passed:
                heur.fitness = samples_by_id[heur.sample_id].cpu_time
            heuristics_passed.sort(key=lambda heur: heur.fitness)  # Lower is better
            heuristics_passed = heuristics_passed[:population_size]

            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                if len(heuristics_passed) > 0:
                    weights = []
                    for i, heur in enumerate(heuristics_passed):
                        weights.append(1 / (i + population_size))
                    future_list = [
                        executor.submit(self.evolve_e_prompt, task_description, heuristics_passed, population_size,
                                        weights,
                                        task_id,
                                        batch_id,
                                        heuristic_e1_prompt, nb_parents=OptimizerEoH.NB_PARENTS),
                        executor.submit(self.evolve_e_prompt, task_description, heuristics_passed, population_size,
                                        weights,
                                        task_id,
                                        batch_id,
                                        heuristic_e2_prompt, nb_parents=OptimizerEoH.NB_PARENTS),
                        executor.submit(self.evolve_m_prompt, task_description, heuristics_passed, population_size,
                                        weights,
                                        task_id,
                                        batch_id,
                                        heuristic_m1_prompt, ),
                        executor.submit(self.evolve_m_prompt, task_description, heuristics_passed, population_size,
                                        weights,
                                        task_id,
                                        batch_id,
                                        heuristic_m2_prompt),
                        executor.submit(self.evolve_m_prompt, task_description, heuristics_passed, population_size,
                                        weights,
                                        task_id,
                                        batch_id,
                                        heuristic_m3_prompt)]
                else:
                    # print(f"No valid heuristic for task {task_id} and batch {batch_id}, trying a fixing approach")
                    future_list = [
                        executor.submit(self.evolve_fix_prompt, task_description, heuristics, population_size,
                                        task_id,
                                        batch_id,
                                        heuristic_fix_prompt, samples_by_id)
                    ]
                for future in concurrent.futures.as_completed(future_list):
                    results += future.result()
            return results
        except Exception as e:
            # Put all exception text into an exception and raise that
            raise Exception(f"{e}\n{traceback.format_exc()}")

    def evolve_e_prompt(self, task_description, heuristics, population_size, weights, task_id, batch_id, e_prompt,
                        nb_parents) -> list[GenerationResult]:
        all_results = []
        extra_body = {"guided_json": HeuristicGeneration.model_json_schema()}
        nb_parents = min(len(heuristics), nb_parents)
        for i in range(population_size):
            # Select heuristics
            chosen_heuristics: list[Heuristic] = random.choices(heuristics, k=nb_parents,
                                                                weights=weights)
            chosen_heuristics_str = "\n".join([str(heur) for heur in chosen_heuristics])
            prompt = e_prompt.format(task_description=task_description, heuristics_str=chosen_heuristics_str,
                                     nb_algo=len(chosen_heuristics))
            heuristic_str, usage = self._generate_with_usage(prompt, extra_body=extra_body)
            generation_result = self.get_generation_result_from_heuristic_str(batch_id, heuristic_str, task_id, usage)
            all_results.append(generation_result)
        return all_results

    def evolve_m_prompt(self, task_description, heuristics, population_size, weights, task_id, batch_id, m_prompt) -> \
            list[GenerationResult]:
        all_results = []
        extra_body = {"guided_json": HeuristicGeneration.model_json_schema()}
        for i in range(population_size):
            # Select heuristics
            chosen_heuristic: Heuristic = random.choices(heuristics, k=1,
                                                         weights=weights)[0]
            chosen_heuristics_str = str(chosen_heuristic)
            prompt = m_prompt.format(task_description=task_description, heuristic_str=chosen_heuristics_str)
            heuristic_str, usage = self._generate_with_usage(prompt, extra_body=extra_body)
            generation_result = self.get_generation_result_from_heuristic_str(batch_id, heuristic_str, task_id, usage)
            all_results.append(generation_result)
        return all_results

    def evolve_fix_prompt(self, task_description, heuristics, population_size, task_id, batch_id,
                          fix_prompt, samples: dict[str, CodeEvalResult]) -> list[GenerationResult]:
        all_results = []
        extra_body = {"guided_json": HeuristicGeneration.model_json_schema()}
        for i in range(population_size):
            # Select heuristics
            chosen_heuristic: Heuristic = random.choice(heuristics)
            chosen_heuristics_str = str(chosen_heuristic)
            prompt = fix_prompt.format(task_description=task_description, heuristic_str=chosen_heuristics_str,
                                       exec_feedback=get_execution_feedback(samples.get(chosen_heuristic.sample_id)))
            heuristic_str, usage = self._generate_with_usage(prompt, extra_body=extra_body)
            generation_result = self.get_generation_result_from_heuristic_str(batch_id, heuristic_str, task_id, usage)
            all_results.append(generation_result)
        return all_results

    def get_generation_result_from_heuristic_str(self, batch_id, heuristic_str, task_id, usage):
        try:
            heuristic_dto = HeuristicGeneration.model_validate_json(heuristic_str)
            heuristic = Heuristic(description=heuristic_dto.description, code=heuristic_dto.code, task_id=task_id,
                                  sample_id=str(uuid.uuid4()), batch_id=batch_id)
        except ValidationError as e:
            rich.print(f"[red]Error while validating json for task {task_id} - batch {batch_id}: {e}[/]")
            heuristic = Heuristic(description="invalid", code="invalid", task_id=task_id, sample_id=str(uuid.uuid4()),
                                  batch_id=batch_id)
        generation_result = GenerationResult(heuristic.code, usage=usage, sample_id=heuristic.sample_id,
                                             extra_data_to_save={"heuristic": asdict(heuristic)}, task_id=task_id)
        return generation_result

    def _get_init_heuristics(self, task_id, task_description, population_size, batch_id) -> list[GenerationResult]:
        prompt = heuristic_initialization_prompt.format(task_description=task_description)
        extra_body = {"guided_json": HeuristicGeneration.model_json_schema()}
        res = self._generate_with_usage_n_samples(prompt, num_samples=population_size, extra_body=extra_body)
        generation_results: list[GenerationResult] = []
        for heuristic_str, usage in res:
            generation_result = self.get_generation_result_from_heuristic_str(batch_id, heuristic_str, task_id, usage)
            generation_results.append(generation_result)
        return generation_results

    def _population_initialization(self, dataset_dict, num_batches, population_size, target_path, raw_target_path):
        self.load_model()
        all_heuristics = defaultdict(dict)
        # For every task initialize N*M heuristics in M batches
        tasks = []
        for task_id in dataset_dict:
            for i in range(num_batches):
                batch_id = str(uuid.uuid4())
                tasks.append((task_id, batch_id))
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.nb_workers) as executor:
            future_list = []
            for task_id, batch_id in tasks:
                future_list.append(executor.submit(self._get_init_heuristics, task_id=task_id,
                                                   task_description=dataset_dict[task_id]["prompt"].strip() + "\n",
                                                   population_size=population_size, batch_id=batch_id))
            print("\nAdded all tasks")
            with progress(f"Generating samples - {self.get_optimizer_identifier()}") as prog_bar:
                for future in prog_bar.track(concurrent.futures.as_completed(future_list), total=len(tasks)):
                    results: list[GenerationResult] = future.result()
                    for result in results:
                        sanitized_solution = sanitize(result.solution,
                                                      entrypoint=dataset_dict[result.task_id]["entry_point"])
                        save_generation_jsonl(target_path, raw_target_path, sanitized_solution, result.solution,
                                              result.task_id,
                                              usage_share=result.usage,
                                              sample_id=result.sample_id, extra_data=result.extra_data_to_save)
        self.unload_model()
        return all_heuristics
