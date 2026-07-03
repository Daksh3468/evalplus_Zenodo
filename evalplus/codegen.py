import json
import os
import uuid
from typing import Dict, List, Optional

from evalplus.data import get_evalperf_data, get_human_eval_plus, get_mbpp_plus
from evalplus.perf.energy import measure_energy
from evalplus.provider import DecoderBase, make_model
from evalplus.provider.base import DecoderExtra
from evalplus.sanitize import sanitize
from evalplus.utils import progress


def codegen(
        target_path: str,
        model: DecoderBase,
        dataset: Dict,
        greedy=False,
        n_samples=1,
        id_range=None,
        resume=True,
):
    """
    Generates code solutions for a given dataset using a specified model and saves the outputs.

    Args:
        target_path (str): The path where the generated code outputs will be saved.
        model (DecoderBase): The model used for code generation.
        dataset (Dict): The dataset containing tasks for which code needs to be generated.
        greedy (bool, optional): If True, uses greedy decoding. Defaults to False.
        n_samples (int, optional): The number of samples to generate per task. Defaults to 1.
        id_range (tuple, optional): A tuple specifying the range of task IDs to process. Defaults to None.
        resume (bool, optional): If True, resumes from previously saved outputs. Defaults to True.
    """
    already_processed_tasks = {}

    should_load_previously_saved_outputs_with_jsonl = resume and target_path.endswith(".jsonl") and os.path.isfile(
        target_path)
    if should_load_previously_saved_outputs_with_jsonl:
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                task_id = json.loads(line)["task_id"]
                already_processed_tasks[task_id] = already_processed_tasks.get(task_id, 0) + 1

    if target_path.endswith(".jsonl"):
        raw_target_path = target_path.replace(".jsonl", ".raw.jsonl")
    else:
        raw_target_path = target_path + ".raw"
        os.makedirs(target_path, exist_ok=True)

    print(f"Sanitized code outputs will be saved to {target_path}")
    print(f"Raw outputs will be saved to {raw_target_path}")

    backend_type: str = type(model).__name__
    with progress(backend_type) as prog_bar:  # TODO: make use of threads ?
        for task_id, task in prog_bar.track(dataset.items()):
            if _should_skip_task(id_range, prog_bar, task_id):
                continue

            if not target_path.endswith(".jsonl"):
                task_path = task_id.replace("/", "_")
                os.makedirs(os.path.join(target_path, task_path), exist_ok=True)
                number_of_generations = len(
                    [f for f in os.listdir(os.path.join(target_path, task_path)) if f.endswith(".py")])
                already_processed_tasks[task_id] = number_of_generations

            n_more_samples = n_samples
            log = f"Codegen: {task_id} @ {model}"
            if resume and already_processed_tasks.get(task_id, 0) > 0:
                log += f" (resuming from {already_processed_tasks[task_id]})"
                n_more_samples -= already_processed_tasks[task_id]

            prog_bar.console.print(log)

            sidx = n_samples - n_more_samples
            while sidx < n_samples:
                prompt = task["prompt"].strip() + "\n"
                outputs = model.codegen(prompt, do_sample=not greedy, num_samples=n_samples - sidx)
                assert outputs, "No outputs from model!"

                # Not all models support usage metrics.
                if isinstance(model, DecoderExtra):
                    request_usage_info = model.get_usage_of_last_request()
                else:
                    request_usage_info = None

                for impl in outputs:
                    solution = prompt + impl if model.is_direct_completion() else impl
                    sanitized_solution = sanitize(solution, entrypoint=task["entry_point"])

                    if target_path.endswith(".jsonl"):
                        usage_share = get_usage_metrics(outputs, request_usage_info)
                        id = str(uuid.uuid4())
                        save_generation_jsonl(target_path, raw_target_path, sanitized_solution, solution, task_id,
                                              usage_share, id)
                    else:
                        write_generation_python(raw_target_path, sanitized_solution, sidx, solution, target_path,
                                                task_path)
                    sidx += 1


def write_generation_python(raw_target_path, sanitized_solution, sidx, solution, target_path, task_path):
    # Writing the sanitized version
    with open(
            os.path.join(target_path, task_path, f"{sidx}.py"),
            "w",
            encoding="utf-8",
    ) as f:
        f.write(sanitized_solution)
    # Writing the raw version
    with open(
            os.path.join(raw_target_path, task_path, f"{sidx}.py"),
            "w",
            encoding="utf-8",
    ) as f:
        f.write(solution)


def save_generation_jsonl(target_path, raw_target_path, sanitized_solution, solution, task_id, usage_share, sample_id,
                          extra_data=None):
    # Writing the sanitized version
    with open(target_path, "a") as f:
        json_to_write_sanitized = {"task_id": task_id, "sample_id": sample_id, "solution": sanitized_solution,
                                   "usage_share": usage_share, "extra_data": extra_data}
        f.write(
            json.dumps(
                json_to_write_sanitized,
                default=vars
            )
            + "\n"
        )
    # Writing the raw version
    with open(raw_target_path, "a") as f:
        json_to_write_raw = {"task_id": task_id, "sample_id": sample_id, "solution": solution,
                             "usage_share": usage_share, "extra_data": extra_data}
        f.write(
            json.dumps(json_to_write_raw, default=vars)
            + "\n"
        )


def get_usage_metrics(outputs, usage):
    usage_share = {}
    if usage:
        usage_share = {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": round(usage["completion_tokens"] / len(outputs), 2)
        }
        usage_share["total_tokens"] = round(usage_share["prompt_tokens"] + usage_share["completion_tokens"], 2)
    else:
        for key in ["completion_tokens", "prompt_tokens", "total_tokens"]:
            usage_share[key] = None
    return usage_share


def _should_skip_task(id_range, p, task_id):
    should_skip = False
    if id_range is not None:
        id_num = int(task_id.split("/")[1])
        low, high = id_range
        if id_num < low or id_num >= high:
            p.console.print(f"Skipping {task_id} as it is not in {id_range}")
            should_skip = True
    return should_skip


def run_codegen(
        model: str,
        dataset: str,
        root: str = "evalplus_results",
        bs: Optional[int] = None,
        n_samples: int = 1,
        temperature: float = 0.0,
        resume: bool = True,
        greedy: bool = False,
        id_range: List = None,
        version: str = "default",
        backend: str = "openai",
        force_base_prompt: bool = False,
        base_url: str = None,
        tp: int = 1,
        evalperf_type: str = None,  # For EvalPerf
        jsonl_fmt: bool = True,
        attn_implementation: str = "eager",
        device_map: Optional[str] = None,
        trust_remote_code: bool = False,
        enable_prefix_caching: bool = False,
        enable_chunked_prefill: bool = False,
        dtype: str = "bfloat16",
        gptqmodel_backend: str = "auto",  # For GPTQModel
        gguf_file: Optional[str] = None
) -> tuple[str, bool]:
    """
    Generate code solutions for each task in the dataset. The code solutions will be saved in `target_path`.

    Second return value is a flag indicating whether the codegen was really done or not
    """

    generated_everything_this_execution = True

    assert dataset in ["humaneval", "mbpp", "evalperf"], f"Invalid dataset {dataset}"
    assert evalperf_type is None or evalperf_type in [
        "instruct",
        "perf-instruct",
        "perf-CoT",
    ]

    # Make dir for codes generated by each model
    identifier = get_result_file_identifier(backend, evalperf_type, model, temperature)

    target_path = get_target_path(dataset, identifier, jsonl_fmt, root)

    dataset_dict = get_dataset_dict(dataset, id_range, version)

    all_tasks_complete, generated_everything_this_execution = are_all_tasks_complete(dataset_dict,
                                                                                     generated_everything_this_execution,
                                                                                     jsonl_fmt, n_samples, target_path)
    if all_tasks_complete:
        print("All samples are already cached. Skipping codegen.")
        return target_path, False

    if greedy and (temperature != 0 or bs != 1 or n_samples != 1):
        temperature = 0.0
        bs = 1
        n_samples = 1
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if id_range is not None:
        assert len(id_range) == 2, "id_range must be a list of length 2"
        assert id_range[0] < id_range[1], "id_range must be increasing"
        id_range = tuple(id_range)

    if bs is None:
        bs = min(n_samples, 32)
        print(f"Setting batch size to {bs}")

    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, dataset), exist_ok=True)

    instruction_prefix, response_prefix = get_model_instructions(evalperf_type)

    # Model creation
    model_runner = make_model(
        model=model,
        backend=backend,
        batch_size=bs,
        temperature=temperature,
        force_base_prompt=force_base_prompt,
        dataset=dataset,
        base_url=base_url,
        tp=tp,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        dtype=dtype,
        gptqmodel_backend=gptqmodel_backend,
        gguf_file=gguf_file,
    )

    codegen(
        target_path=target_path,
        dataset=dataset_dict,
        greedy=greedy,
        model=model_runner,
        n_samples=n_samples,
        resume=resume,
        id_range=id_range,
    )

    # force shutdown the model runner
    del model_runner
    import gc

    gc.collect()

    return (target_path, generated_everything_this_execution)


def get_model_instructions(evalperf_type):
    # Model instructions
    instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
    if evalperf_type == "perf-instruct":
        instruction_prefix = "Please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    elif evalperf_type == "perf-CoT":
        instruction_prefix = "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    elif evalperf_type is not None and evalperf_type != "instruct":
        raise ValueError(f"Invalid evalperf_type: {evalperf_type}")
    return instruction_prefix, response_prefix


def are_all_tasks_complete(dataset_dict, generated_everything_this_execution, jsonl_fmt, n_samples, target_path):
    all_tasks_complete = False
    if jsonl_fmt and os.path.isfile(target_path):
        task_counts = {}
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                task_id = data["task_id"]
                task_counts[task_id] = task_counts.get(task_id, 0) + 1
            generated_everything_this_execution = False

            all_tasks_complete = all(
                task_counts.get(task_id, 0) >= n_samples
                for task_id in dataset_dict.keys()
            )
    return all_tasks_complete, generated_everything_this_execution


def get_dataset_dict(dataset, id_range, version):
    if dataset == "humaneval":
        dataset_dict = get_human_eval_plus(version=version)
    elif dataset == "mbpp":
        dataset_dict = get_mbpp_plus(version=version)
    elif dataset == "evalperf":
        original_dataset = {**get_human_eval_plus(), **get_mbpp_plus()}
        dataset_dict = {k: original_dataset[k] for k in get_evalperf_data()}
        assert id_range is None, "id_range not supported for evalperf"
    else:
        raise ValueError(f"Invalid dataset {dataset}")
    return dataset_dict


def get_result_file_identifier(backend, evalperf_type, model, temperature):
    """
    bigcode--starcoder2-15b-instruct-v0.1_openai_temp_0.5.jsonl
    """
    identifier = model.strip("./").replace("/", "--") + f"_{backend}_temp_{temperature}"
    if evalperf_type:
        identifier += f"-{evalperf_type}"
    return identifier


def get_target_path(dataset: str, identifier: str, jsonl_fmt: bool, root: str):
    target_path = os.path.join(root, dataset, identifier)
    if jsonl_fmt:
        target_path += ".jsonl"
    else:
        os.makedirs(target_path, exist_ok=True)
    return target_path


def main():
    from fire import Fire

    Fire(run_codegen)


if __name__ == "__main__":
    main()
