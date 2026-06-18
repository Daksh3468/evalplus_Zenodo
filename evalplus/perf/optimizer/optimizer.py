import concurrent.futures
import dataclasses
import json
import os
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Iterable, override

import rich

from evalplus.codegen import save_generation_jsonl, get_target_path, get_result_file_identifier, get_usage_metrics
from evalplus.evalperf import CodeEvalResult
from evalplus.provider import DecoderBase, ModelConfig, make_model_from_config
from evalplus.provider.base import DecoderExtra
from evalplus.sanitize import sanitize
from evalplus.utils import progress


class OptimizerBase(ABC):
    """Optimizer base class"""

    def __init__(self, model_config: ModelConfig, nb_workers: int):
        self.model_config = model_config
        self.model: DecoderBase = None
        self.nb_workers = nb_workers
        self.strict_result_file_verification = True

    def optimize(self, samples_to_optimize: dict[str, list[CodeEvalResult]], dataset_dict: dict, iter_number,
                 root_path: str) -> str:
        """
        Args:
            samples_to_optimize (list): list of codes
        """
        samples_to_optimize = self.filter_samples_to_optimize(samples_to_optimize)

        identifier = self.get_result_file_identifier(iter_number)
        target_path = get_target_path("evalperf", identifier, True, root=root_path)
        raw_target_path = target_path.replace(".jsonl", ".raw.jsonl")
        if are_all_tasks_complete(samples_to_optimize.keys(), target_path, strict=self.strict_result_file_verification):
            rich.print(f"Tasks in {target_path} were already complete, skipping optimization.")
            return target_path
        self._do_optimize_codes(dataset_dict=dataset_dict, raw_target_path=raw_target_path,
                                samples_to_optimize=samples_to_optimize, target_path=target_path,
                                iter_number=iter_number)
        return target_path

    @abstractmethod
    def _do_optimize_codes(self, dataset_dict, raw_target_path, samples_to_optimize: dict[str, list[CodeEvalResult]],
                           target_path, iter_number):
        pass

    def _generate_with_usage(self, prompt, extra_body=None) -> tuple[str, dict]:
        res = self.model.codegen(prompt, do_sample=True, num_samples=1, extra_body=extra_body)[0]
        if isinstance(self.model, DecoderExtra):
            usage = self.model.get_usage_of_last_request()
        else:
            usage = None
        return res, usage

    def _generate_with_usage_n_samples(self, prompt, num_samples, extra_body=None) -> list[tuple[str, dict]]:
        outputs: list[str] = self.model.codegen(prompt, do_sample=True, num_samples=num_samples, extra_body=extra_body)
        if isinstance(self.model, DecoderExtra):
            usage = self.model.get_usage_of_last_request()
        else:
            usage = None
        res: list[tuple[str, dict]] = []
        for solution in outputs:
            res.append((solution, get_usage_metrics(outputs, usage)))
        return res

    def load_model(self):
        self.model = make_model_from_config(self.model_config)

    def unload_model(self):
        del self.model
        import gc
        gc.collect()

    @abstractmethod
    def get_optimizer_identifier(self):
        pass

    def get_result_file_identifier(self, iteration_number) -> str:
        base_identifier = get_result_file_identifier(self.model_config.backend, None,
                                                     self.model_config.model, self.model_config.temperature)
        return base_identifier + f"_{self.get_optimizer_identifier()}_{iteration_number}"

    def can_accept_errored_samples(self) -> bool:
        """Whether the optimizer should be passed only correct samples."""
        return False  # TODO: Refactor into filter_samples_to_optimize

    def can_generate_samples_from_scratch(self):
        return False

    def generate_samples(self, dataset_dict: dict, n_samples: int, root_path: str):
        raise NotImplementedError("Operation not supported")

    def filter_samples_to_optimize(self, samples_to_optimize: dict[str, list[CodeEvalResult]]) -> dict[
        str, list[CodeEvalResult]]:
        """Some optimizers might want to filter samples that already passed"""
        return samples_to_optimize


@dataclasses.dataclass
class GenerationResult:
    solution: str
    usage: dict
    sample_id: str
    extra_data_to_save: dict
    task_id: str


class OptimizerSingleCode(OptimizerBase):
    """Singe-code optimizer base class"""

    def _do_optimize_codes(self, dataset_dict, raw_target_path, samples_to_optimize, target_path, iter_number):
        self.load_model()
        tasks: list[tuple[str, CodeEvalResult]] = []
        for task_id in samples_to_optimize:
            for sample in samples_to_optimize[task_id]:
                tasks.append((task_id, sample))

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.nb_workers) as executor:
            future_list = []
            for task_id, sample_to_optimize in tasks:
                future_list.append(executor.submit(self._optimize_single_code,
                                                   code=sample_to_optimize,
                                                   task_prompt=dataset_dict[task_id]["prompt"],
                                                   task_id=task_id
                                                   ))
            print("\nAdded all tasks")
            with progress(f"Generating samples - {self.get_optimizer_identifier()}") as prog_bar:
                for future in prog_bar.track(concurrent.futures.as_completed(future_list), total=len(tasks)):
                    try:
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
                    except Exception as e:
                        print("Exception when optimizing sample " + str(e), traceback.format_exc())
        self.unload_model()

    def _optimize_single_code(self, code: CodeEvalResult, task_prompt: str, task_id: str) -> list[GenerationResult]:
        prompt = self._get_prompt(code.solution, task_prompt)
        return self._generate_single_code(code, prompt, task_id)

    def _generate_single_code(self, code: CodeEvalResult, prompt: str, task_id: str,
                              extra_data_to_save: Optional[dict] = None) -> list[GenerationResult]:
        res, usage = self._generate_with_usage(prompt)
        return [GenerationResult(res, usage, code.sample_id, extra_data_to_save,
                                 task_id)]  # Note: the id of the previous sample is propagated

    @abstractmethod
    def _get_prompt(self, solution, task_prompt):
        pass


class OptimizerSimple(OptimizerSingleCode):
    """Simple Prompt based optimizer"""

    def __init__(self, model_config: ModelConfig, nb_workers):
        super().__init__(model_config, nb_workers)

    def _get_prompt(self, code_content, task_description):
        return SIMPLE_PROMPT.format(code_content=code_content, task_description=task_description, trigger=TRIGGER,
                                    regular_requirement=REGULAR_REQUIREMENT)

    def get_optimizer_identifier(self):
        return "simple"


class OptimizerCoT(OptimizerSingleCode):
    """Simple Prompt based optimizer"""

    def _get_prompt(self, code_content, task_description):
        return COT_PROMPT.format(code_content=code_content, task_description=task_description, trigger=TRIGGER,
                                 regular_requirement=REGULAR_REQUIREMENT)

    def get_optimizer_identifier(self):
        return "cot"


class OptimizerCoD(OptimizerSingleCode):
    """Simple Prompt based optimizer"""

    def _get_prompt(self, code_content, task_description):
        return COD_PROMPT.format(code_content=code_content, task_description=task_description, trigger=TRIGGER,
                                 regular_requirement=REGULAR_REQUIREMENT)

    def get_optimizer_identifier(self):
        return "cod"


class OptimizerSelfRefineNlFeedback(OptimizerSingleCode):

    @override
    def _optimize_single_code(self, code: CodeEvalResult, task_prompt: str, task_id: str) -> list[GenerationResult]:
        feedback, usage = self._generate_with_usage(self._get_feedback_prompt(code))
        prompt = self._get_prompt(code.solution, feedback)
        return self._generate_single_code(code, prompt, task_id, {"feedback": {"content": feedback, "usage": usage}})

    def _get_feedback_prompt(self, code_content):
        return FEEDBACK_PROMPT.format(code_content=code_content)

    def _get_prompt(self, code_content, feedback):
        return REFINE_FEEDBACK_PROMPT.format(code_content=code_content, feedback=feedback, trigger=TRIGGER,
                                             regular_requirement=REGULAR_REQUIREMENT)

    def get_optimizer_identifier(self):
        return "self-refine-nl-feedback"

    def can_accept_errored_samples(self):
        return True


class OptimizerSelfRefineExecFeedback(OptimizerSingleCode):

    @override
    def _optimize_single_code(self, code: CodeEvalResult, task_prompt: str, task_id: str) -> list[GenerationResult]:
        feedback = get_execution_feedback(code)
        prompt = self._get_prompt(code.solution, feedback)
        return self._generate_single_code(code, prompt, task_id)

    def _get_prompt(self, code_content, feedback):
        return REFINE_FEEDBACK_PROMPT.format(code_content=code_content, feedback=feedback, trigger=TRIGGER,
                                             regular_requirement=REGULAR_REQUIREMENT)

    def get_optimizer_identifier(self):
        return "self-refine-exec-feedback"

    def can_accept_errored_samples(self):
        return True


def are_all_tasks_complete(task_ids: Iterable[str], target_path, strict=True, delete_if_not_complete=True):
    if not os.path.isfile(target_path):
        return False
    if not strict:
        print("Result file already exists. skipping generation")
        return True  # Return early if the file already exists
    task_counts = {}
    with open(target_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            task_id = data["task_id"]
            task_counts[task_id] = task_counts.get(task_id, 0) + 1

        all_tasks_complete = all(
            task_counts.get(task_id, 0) > 0
            for task_id in task_ids
        )
        print("All tasks complete :" + str(all_tasks_complete))
        print("Missing tasks: ", [task_id for task_id in task_ids if task_counts.get(task_id, 0) == 0])
    if not all_tasks_complete:
        rich.print(f"[yellow]Deleting {target_path}[/]")
        os.remove(target_path)
    return all_tasks_complete


def get_execution_feedback(code: CodeEvalResult):
    if not code.has_errors() and not code.passed:
        rich.print(
            f"[red]ERROR: A code had no errors but did not pass previously. Falling back to default feedback.[/] : {code}")
        return "Your solution was INCORRECT and passed 0 test cases. This could either be a flaw in logic or a syntax error. Fix it.\n"
    if not code.has_errors():  # Passed all test cases
        if code.cpu_time is None or code.dps is None:
            rich.print(f"[red]code had no error but has no profiling data: {code}[/]")
            dps_to_show = 50
        else:
            dps_to_show = code.dps
        feedback = f'Your solution was functionally CORRECT across ALL test cases!\n'
        feedback += f'Here is the run time that your code utilized for all test cases: Run time: {code.cpu_time}s\n'
        feedback += f'Your code was faster than {dps_to_show}% of the other codes, but can still be improved.\n'
    else:  # Passed no test cases
        # We expect the correctness worker to stop at the first error. The size of the error dict must not be greater than two (one for the test case and one for the whole profiling)
        # assert len(code.error_details.case) <= 1, "Too many errors, this optimizer was built only for fast checking"
        nb_test_passed = 0  # if len(code.error_details.case) == 0 else int(list(code.error_details.case.keys())[0]) - 1
        feedback = f'Your solution was INCORRECT and passed {nb_test_passed} test cases. This could either be a flaw in logic or a syntax error. Please see error logs.\n'

    if code.has_errors() and not code.error_details.has_test_case_errors():  # Code-level error, like a syntax error
        feedback += 'Here are the error logs for the failed test cases\n'
        feedback += f'-- Error log for failed test case 0 --\n'  # Using first test case
        feedback += code.error_details.whole + '\n'
    elif code.has_errors() and code.error_details.has_test_case_errors():  # Test-case level errors
        test_case_errors: dict = {0: code.error_details.case}
        feedback += 'Here are the error logs for the failed test cases\n'
        for test_id in test_case_errors.keys():
            # Test Input of failed test case NOT added to feedback
            feedback += f'-- Error log for failed test case {test_id} --\n'
            if test_case_errors[test_id]:
                feedback += test_case_errors[test_id] + '\n'  # Wrong Answer: {} Expected Answer: {}
            else:
                feedback += '\n'

    return feedback


TRIGGER = "Give your solution as follows. Wrap it with ```python```.\n"
REGULAR_REQUIREMENT = "Please make sure your refined solution is functionally equivalent with the original solution and do not change the input-output format and the name of the major components.\n"

SIMPLE_PROMPT = "Optimize the python program below to run faster and use less memory. {regular_requirement} {trigger}\n\n{task_description}\n\n```python\n{code_content}\n```\n"
COT_PROMPT = "Optimize the python program below to run faster and use less memory. Think step by step {regular_requirement} {trigger}\n\n{task_description}\n\n```python\n{code_content}\n```\n"
COD_PROMPT = "Optimize the python program below to run faster and use less memory. Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. {regular_requirement} {trigger}\n\n{task_description}\n\n```python\n{code_content}\n```\n"

FEEDBACK_PROMPT = "Give feedback in English for why the code solution below is incorrect or inefficient and how the program can be fixed.\n\n## Candidate solution:\n```python\n{code_content}\n```\n\n## Feedback for incorrectness/inefficiency and how it can be improved:\n"
REFINE_FEEDBACK_PROMPT = "Refine the given incorrect or sub-optimal code solution based on the feedback specified below. {regular_requirement} {trigger}\n\n## Candidate solution:\n```python\n{code_content}\n```\n\n## Feedback to improve the code:\n{feedback}\n\n## Refined code that includes optimizations specified in feedback:\n"
