import concurrent.futures
import traceback
import uuid
from typing import Optional

import rich
from pydantic import BaseModel
from typing_extensions import override

from evalplus.codegen import save_generation_jsonl
from evalplus.evalperf import CodeEvalResult
from evalplus.perf.optimizer import optimizer
from evalplus.perf.optimizer.optimizer import are_all_tasks_complete, OptimizerSingleCode, get_execution_feedback, \
    GenerationResult
from evalplus.sanitize import sanitize
from evalplus.utils import progress


class Algorithm(BaseModel):
    algorithm_key_description: str
    pseudo_algorithm: Optional[str]


class AlgorithmGenerationModel(BaseModel):
    algorithms: list[Algorithm]


class TaskDescriptionCheck(BaseModel):
    analysis: str
    matches_original_requirements: bool
    explanation: Optional[str]


task_formalization_prompt: str = """
As a professional algorithm engineer, please analyze this algorithm problem according to the following categories.Do not generate any example implementation:
1. Entry point function name:
2. Input/Output conditions
3. Edge Cases and Parameters type(Int String...)
4. expected behavior

The algorithm problem description is as follows: {task_description}
"""

task_formalization_check_prompt: str = """
As an excellent algorithm engineer, please analyze whether the explanation of the problem matches the original requirements of the problem. If they are consistent, output "Yes"; if they are not consistent, output "No" along with the reason, as shown below:
{{"analysis": "...", "matches_original_requirements": true, "explanation": null}}
{{"analysis": "...", "matches_original_requirements": false, "explanation": "..."}}

Here is the original problem content: {task_description}
Here is the explanation of the problem description: {task_formalization}
"""

algorithm_generation: str = """As a professional algorithm engineer, you can effectively design multiple algorithms to solve the problem with low time complexity and output them in pseudo algorithm format, and pseudo algorithm is a nonlinear, high-level programming language for algorithmic logic. It combines natural language and programming structures to express the steps and sums of algorithms. The main purpose of process algorithms is to clearly display the core ideas and logic of the algorithm without relying on specific programming language syntax. Please design an excellent algorithm solution based on the problem description provided. The time complexity of the algorithm needs to be as small as possible, and try to output 10 algorithms description
PS: DO NOT provide implementation example nor pseudocode!
{{"algorithms: [
{{"algorithm_key_description": "this algorithm using xxx,the key is to make sure xxx", "pseudo_algorithm": null}},
{{"algorithm_key_description": "this algorithm using xxx,the key is to make sure xxx", "pseudo_algorithm": null}},
{{"algorithm_key_description": "this algorithm using xxx,the key is to make sure xxx", "pseudo_algorithm": null}}
...
]}}

Here is the problem description: {task_formalization}
"""

algorithm_pseudo_code_generation: str = """
As a professional algorithm engineer, you can effectively design multiple algorithms to solve the problem with low time complexity and output them in pseudo algorithm format, and pseudo algorithm is a nonlinear, high-level programming language for algorithmic logic. It combines natural language and programming structures to express the steps and sums of algorithms. The main purpose of process algorithms is to clearly display the core ideas and logic of the algorithm without relying on specific programming language syntax. Please design an excellent algorithm solution based on the problem and algorithm descriptions provided. The time complexity of the algorithm needs to be as small as possible
PS: DO NOT provide implementation example!

You will answer in the following format:
{{"algorithm_key_description": "this algorithm using xxx,the key is to make sure xxx", "pseudo_algorithm": "..."}},

Here is the problem description: {task_formalization}

Here is the algorithm description: {algorithm_description}
"""

optimization_suggestions_prompt: str = """
As a professional Python algorithm programming expert, please provide suggestions for improving code efficiency based on the potential inefficiencies mentioned above. For example:
1.	Using xxx instead of xxx can significantly improve code efficiency.
Please provide at least 20 suggestions. 

This is the algorithm: {algorithm}
"""

code_candidate_generation: str = """
As a professional Python algorithm engineer, please convert the selected algorithm into corresponding Python code. Ensure the code is complete and well-formatted. When converting to a standardized format, be sure to follow the guidelines specified in the “original question format”:
1.	If “from typing import List” appears in the original question, please retain it.
2.	Use the same function name as given in the original question format; do not rename it.
3.	You may incorporate practical optimization details drawn from the knowledge base.
The final output format should be as follows:
```python
[code]
```

Original question format: {task_description}
Description of the question: {task_formalization}
Selected plan and algorithm: {algorithm}
Efficiency and optimization suggestions: {optimization_suggestions}
"""

improve_code_correctness: str = """As a professional code programming algorithm expert, your task is to correct the code and ensure that the code is fixed without impacting its time complexity or practical efficiency. Then I will provide you with specific code and test cases. Based on the algorithm requirements:
If the code is correct but the test case is wrong, please output the unmodified code directly.
If the test case is correct but the code is wrong, please modify the code.
Important Notes:
1.	Do not alter the algorithm itself—maintain minimal time complexity.
2.	Do not change the format, such as the function name.
3.	Please output in the specified format.
4.  Ensure there are no syntax errors.
please output in this format:
```python
[code]
```
    
This is the algorithm problem description:{task_description}
This is the wrong code: {code_candidate}
This is the execution feedback: {exec_feedback}
    """


class OptimizerLLM4EFFI(OptimizerSingleCode):
    def _get_prompt(self, solution, task_prompt):
        pass

    @override
    def _optimize_single_code(self, code: CodeEvalResult, task_prompt: str, task_id: str) -> list[GenerationResult]:
        prompt = improve_code_correctness.format(task_description=task_prompt,
                                                 # task_formalization=task_formalization, # Disabled for now, as it's harder to get the task_formalization
                                                 code_candidate=code.solution,
                                                 exec_feedback=get_execution_feedback(code))
        return self._generate_single_code(code, prompt, task_id)

    def get_optimizer_identifier(self):
        return "llm4effi"

    def can_accept_errored_samples(self) -> bool:
        """Whether the optimizer should be passed only correct samples."""
        return True

    def can_generate_samples_from_scratch(self):
        return True

    @override
    def filter_samples_to_optimize(self, samples_to_optimize: dict[str, list[CodeEvalResult]]) -> dict[
        str, list[CodeEvalResult]]:
        samples_to_optimize_filtered = {}
        for task_id in samples_to_optimize.keys():
            new_list = []
            for sample in samples_to_optimize[task_id]:
                if not sample.passed:
                    new_list.append(sample)
            if len(new_list) > 0:
                samples_to_optimize_filtered[task_id] = new_list
        return samples_to_optimize_filtered

    def _formalize_task(self, task_description, task_id) -> tuple[str, list[dict]]:
        result_dicts = []

        task_formalization = ""
        task_check = ""
        task_check_object = None
        max_loops = 5  # As in the original LLM4EFFI code
        for _ in range(max_loops):
            prompt = task_formalization_prompt.format(task_description=task_description)
            prompt += f"\n{task_formalization}\n{task_check}"
            task_formalization, usage_formalization = self._generate_with_usage(prompt)

            prompt = task_formalization_check_prompt.format(task_description=task_description,
                                                            task_formalization=task_formalization)
            extra_body = {"guided_json": TaskDescriptionCheck.model_json_schema()}
            task_check, usage_check = self._generate_with_usage(prompt, extra_body)
            task_check_object = TaskDescriptionCheck.model_validate_json(task_check)
            result_dicts.append(
                {"formalization": {"content": task_formalization, "usage": usage_formalization},
                 "formalization_check": {"content": task_check_object, "usage": usage_check}}
            )
            if task_check_object.matches_original_requirements:
                break
        assert task_check_object is not None
        if not task_check_object.matches_original_requirements or task_formalization == "":
            rich.print(f"[red]No successful formalization could be produced for task {task_id}[/]")
        return task_formalization, result_dicts

    def _explore_algorithms(self, task_formalization) -> tuple[AlgorithmGenerationModel, dict]:
        prompt = algorithm_generation.format(task_formalization=task_formalization)
        extra_body = {"guided_json": AlgorithmGenerationModel.model_json_schema()}
        algorithms_json, usage_algo = self._generate_with_usage(prompt, extra_body)
        algorithms = AlgorithmGenerationModel.model_validate_json(algorithms_json)
        return algorithms, usage_algo

    def _complete_algorithm(self, task_formalization, algorithm_description) -> tuple[Algorithm, dict]:
        prompt = algorithm_pseudo_code_generation.format(task_formalization=task_formalization,
                                                         algorithm_description=algorithm_description)
        extra_body = {"guided_json": Algorithm.model_json_schema()}
        algorithm_json, usage_algo = self._generate_with_usage(prompt, extra_body)
        algorithm = Algorithm.model_validate_json(algorithm_json)
        return algorithm, usage_algo

    def _optimize_implementation(self, algorithm) -> tuple[str, dict]:
        prompt = optimization_suggestions_prompt.format(algorithm=algorithm)
        optimization_suggestions, usage_opti = self._generate_with_usage(prompt)
        return optimization_suggestions, usage_opti

    def _generate_code_candidate(self, task_description, task_formalization, algorithm, optimization_suggestions):
        prompt = code_candidate_generation.format(task_description=task_description,
                                                  task_formalization=task_formalization,
                                                  algorithm=algorithm,
                                                  optimization_suggestions=optimization_suggestions)
        code_candidate, usage = self._generate_with_usage(prompt)
        return code_candidate, usage

    def _improve_code_correctness(self, task_description, task_formalization, code_candidate, exec_feedback):
        prompt = improve_code_correctness.format(task_description=task_description,
                                                 task_formalization=task_formalization,
                                                 code_candidate=code_candidate,
                                                 exec_feedback=exec_feedback)
        improved_code, usage = self._generate_with_usage(prompt)
        return improved_code, usage

    def _generate_one_batch_of_code_candidates(self, task_id, task_description):
        batch_id = str(uuid.uuid4())
        extra_data = {}
        task_formalization, results_dicts = self._formalize_task(task_description, task_id)
        extra_data["formalization"] = results_dicts
        algorithms, usage_algo = self._explore_algorithms(
            task_formalization)  # We need to split algorithm creation and pseudo code generation in order to keep the number of tokens being created low under the limit
        usage_algo_completion = [None for i in range(len(algorithms.algorithms))]
        for i in range(len(algorithms.algorithms)):
            algorithms.algorithms[i], usage_algo_completion[i] = self._complete_algorithm(task_formalization,
                                                                                          algorithm_description=
                                                                                          algorithms.algorithms[
                                                                                              i].algorithm_key_description)
        extra_data["algorithm"] = {"content": algorithms, "usage": usage_algo,
                                   "usage_completion": usage_algo_completion}
        code_candidates = []
        for algo in algorithms.algorithms:
            sample_id = str(uuid.uuid4())
            algo_str = algo.algorithm_key_description + "\n" + algo.pseudo_algorithm
            optimization_suggestions, usage_opti = self._optimize_implementation(algo_str)
            code_candidate, usage_code_candidate = self._generate_code_candidate(task_description, task_formalization,
                                                                                 algo_str, optimization_suggestions)
            if not "optimizations_suggestion" in extra_data:
                extra_data["optimizations_suggestion"] = list()
            extra_data["optimizations_suggestion"].append(
                {"content": optimization_suggestions, "usage": usage_opti, "sample_id": sample_id})
            if not "code_candidate" in extra_data:
                extra_data["code_candidate"] = list()
            extra_data["code_candidate"].append(
                {"content": code_candidate, "usage": usage_code_candidate, "sample_id": sample_id})
            code_candidates.append((code_candidate, batch_id, sample_id))
        return code_candidates, extra_data, task_id

    @override
    def generate_samples(self, dataset_dict: dict, n_samples: int, root_path: str):
        """To call in the first iteration"""
        identifier = self.get_result_file_identifier(0)
        target_path = optimizer.get_target_path("evalperf", identifier, True, root=root_path)
        raw_target_path = target_path.replace(".jsonl", ".raw.jsonl")
        if are_all_tasks_complete(dataset_dict.keys(), target_path):
            rich.print(f"Tasks in {target_path} were already complete, skipping generation.")
            return target_path

        self.load_model()
        tasks = []
        for task_id in dataset_dict.keys():
            tasks += [task_id] * n_samples
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.nb_workers) as executor:
            future_list = []
            for task_id in tasks:
                future_list.append(executor.submit(self._generate_one_batch_of_code_candidates, task_id=task_id,
                                                   task_description=dataset_dict[task_id]["prompt"].strip() + "\n"))
            print("\nAdded all tasks")
            with progress(f"Generating samples - {self.get_optimizer_identifier()}") as prog_bar:
                for future in prog_bar.track(concurrent.futures.as_completed(future_list), total=len(tasks)):
                    try:
                        if future.exception(timeout=60 * 10):
                            e: Exception = future.exception()
                            prog_bar.print("[red]Error while optimizing sample: [/red]", e, traceback.format_exc())
                            continue
                    except TimeoutError as e:
                        prog_bar.print("[red]Timeout while optimizing sample: [/red]", e, traceback.format_exc())
                    code_candidates, extra_data, task_id = future.result()
                    for i, (solution, batch_id, sample_id) in enumerate(code_candidates):
                        if i == 0:
                            extra_data_to_save = extra_data
                        else:
                            extra_data_to_save = {}
                        extra_data_to_save["batch_id"] = batch_id
                        sanitized_solution = sanitize(solution, entrypoint=dataset_dict[task_id]["entry_point"])
                        save_generation_jsonl(target_path, raw_target_path, sanitized_solution, solution, task_id,
                                              usage_share=None,
                                              sample_id=sample_id, extra_data=extra_data_to_save)
        self.unload_model()
        return target_path
