import concurrent.futures
import traceback
import uuid
from typing import override

from evalplus.codegen import save_generation_jsonl
from evalplus.evalperf import CodeEvalResult
from evalplus.perf.optimizer.optimizer import OptimizerSimple, GenerationResult
from evalplus.sanitize import sanitize
from evalplus.utils import progress


class OptimizerSimple10(OptimizerSimple):
    """Simple optimizer with batches of 10 samples instead of batches of one sample."""

    def _optimize_single_code_in_batch_of_ten(self, code: CodeEvalResult, task_description: str, task_id: str) -> list[
        GenerationResult]:
        prompt = self._get_prompt(code, task_description)
        all_results = [self._generate_single_code(code, prompt, task_id)[0] for _ in range(10)]
        batch_id = str(uuid.uuid4())
        extra_data = {"batch_id": batch_id}
        for res in all_results:
            res.extra_data_to_save = extra_data
        return all_results

    @override
    def _do_optimize_codes(self, dataset_dict, raw_target_path, samples_to_optimize: dict[str, list[CodeEvalResult]],
                           target_path, iter_number):
        if iter_number > 1:
            return super()._do_optimize_codes(dataset_dict, raw_target_path, samples_to_optimize, target_path,
                                              iter_number)
        assert iter_number == 1, "Iteration number should be one!"

        tasks = []
        for task_id in samples_to_optimize:
            for sample in samples_to_optimize[task_id]:
                tasks.append((task_id, sample))
        self.load_model()
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(self.nb_workers)) as executor:
            future_list = []
            for task_id, code in tasks:
                future_list.append(executor.submit(self._optimize_single_code_in_batch_of_ten, code=code,
                                                   task_description=dataset_dict[task_id]["prompt"], task_id=task_id))
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

    def get_optimizer_identifier(self):
        return "simple10"
