import glob
import os

import polars as pl

from utils import load_library


def include_energy_data(task_summaries, root, min_energy_profiled):
    energy_profiling_by_task = get_energy_profiling_by_task(_get_all_energy_profiling_files(root), min_energy_profiled,
                                                            task_summaries)
    task_summaries = task_summaries.join(
        energy_profiling_by_task.select("task_id", "original_result_file", "average_energy_per_execution",
                                        "nb_energy_profiled_recalculated", "idle_power"),
        left_on=["task_id", "file"],
        right_on=["task_id", "original_result_file"],
        how="left",
    )
    # task_summaries = handle_optimizers_with_non_recurring_tasks(task_summaries)
    return task_summaries


def _get_all_energy_profiling_files(root):
    pattern = "*sample-level*energy_results.json"
    files = glob.glob(os.path.join(root, pattern))
    return files


def _load_energy_profiling_file(file):
    raw_df = pl.read_json(open(file, "r"))
    df = (
        raw_df.lazy()
        .with_columns(pl.col("eval").crinkles.struct_to_list(), energy_profiling_file=pl.lit(file))
        .drop("date")
        .explode("eval")
        .unnest("eval")
        .collect()
    )
    return df


def _get_idle_power():
    idle_col = pl.col("summary").struct.field("energy_measures").struct.field("idle")
    return idle_col.struct.field("total_energy").truediv(idle_col.struct.field("perf_duration"))


def get_best_samples_for_batches_with_duplicate_sample_ids(energy_profiling_by_task_duplicate_batches):
    # For these optimizers, tasks have duplicate sample ids. The best result, the one we want to select is the best out of a sample id instead of a batch.
    df_by_sample = energy_profiling_by_task_duplicate_batches.select("original_result_file", "task_id",
                                                                     "results").explode("results")
    df_by_sample_best_results = df_by_sample.with_columns(
        best_num_cpu=pl.col("results").struct.field("num_cpu_instructions").min().over(
            pl.col("results").struct.field("sample_id"))
    ).filter(pl.col("results").struct.field("num_cpu_instructions") == pl.col("best_num_cpu"))
    return df_by_sample_best_results


def get_best_samples_for_normal_batches(energy_profiling_by_task_normal_batches, task_summaries,
                                        normal_batch_optimizers):
    task_summaries_only_batched = task_summaries.filter(
        pl.col("optimizer").is_in(normal_batch_optimizers)).select("best_sample", "file", "task_id")
    energy_profiling_by_task_normal_batches = energy_profiling_by_task_normal_batches.join(
        task_summaries_only_batched,
        left_on=["original_result_file", "task_id"],
        right_on=["file", "task_id"])
    df_by_sample_best_results = energy_profiling_by_task_normal_batches.select("best_sample", "original_result_file",
                                                                               "task_id",
                                                                               "results").with_columns(
        results=pl.col("results")).explode("results").filter(
        pl.col("results").struct.field("sample_id").is_in(pl.col("best_sample")))
    return df_by_sample_best_results


def handle_batching_optimizers(energy_profiling_by_task, task_summaries):
    # For EOH, LLM4EFFI and Simple10, the average energy consumption and duration is made among the best results of the batches.
    normal_batch_optimizers = ["eoh", "llm4effi"]
    duplicate_sample_ids_optimizers = ["simple10"]  # Has duplicate sample ids.
    normal_batch_predicate = pl.col("config").struct.field("optimizer").is_in(normal_batch_optimizers)
    duplicate_sample_ids_optimizers_predicate = pl.col("config").struct.field("optimizer").is_in(
        duplicate_sample_ids_optimizers)

    normal_batch_best_samples = get_best_samples_for_normal_batches(
        energy_profiling_by_task.filter(normal_batch_predicate), task_summaries, normal_batch_optimizers).select(
        "original_result_file", "task_id", "results")
    duplicate_sample_ids_best_samples = get_best_samples_for_batches_with_duplicate_sample_ids(
        energy_profiling_by_task.filter(duplicate_sample_ids_optimizers_predicate)).select("original_result_file",
                                                                                           "task_id", "results")
    best_samples = pl.concat([normal_batch_best_samples, duplicate_sample_ids_best_samples], how="vertical_relaxed")
    df_by_task_only_batched = energy_profiling_by_task.filter(
        normal_batch_predicate | duplicate_sample_ids_optimizers_predicate)

    df_by_task_only_batched_from_best_samples = (
        best_samples.group_by("original_result_file", "task_id")
        .agg(pl.exclude("results").first(),
             pl.col("results").struct.field("energy_measures").alias("energy_to_average"),
             pl.col("results").struct.field("nb_energy_profiled").alias("nb_energy_profiled_selected_values")
             )
        .join(df_by_task_only_batched, on=["original_result_file", "task_id"])
        .with_columns(
            perf_duration=pl.col("energy_to_average").list.eval(
                pl.element().struct.field("perf_duration")).list.sum(),
            total_energy=pl.col("energy_to_average").list.eval(
                pl.element().struct.field("total_energy")).list.sum(),
            nb_energy_profiled_recalculated=pl.col("nb_energy_profiled_selected_values").list.sum(),
            nb_energy_profiled=pl.col("results").list.eval(
                pl.element().struct.field("nb_energy_profiled")).list.sum()
        )
        .drop("energy_to_average", "^.*right$")
    )

    # Join the results

    energy_profiling_by_task_no_batches = energy_profiling_by_task.filter(
        ~normal_batch_predicate & ~duplicate_sample_ids_optimizers_predicate).with_columns(
        perf_duration=pl.col("energy_measures").struct.field("perf_duration"),
        total_energy=pl.col("energy_measures").struct.field("total_energy"),
        nb_energy_profiled_recalculated=pl.col("nb_energy_profiled"),
    )
    energy_profiling_by_task = pl.concat(
        [df_by_task_only_batched_from_best_samples, energy_profiling_by_task_no_batches],
        how="diagonal_relaxed")
    return energy_profiling_by_task


# @mo.cache
def get_energy_profiling_by_task(energy_profiling_files, min_energy_profiled, task_summaries):
    energy_profiling_by_task = (
        pl.concat(
            [_load_energy_profiling_file(file) for file in energy_profiling_files],
            how="diagonal_relaxed",
        )  # Task level
        # .filter(pl.col("energy_profiling_config").struct.field("min_correct") >= _THRESH)
        .with_columns(original_result_file=pl.col("energy_profiling_config").struct.field("original_result_file").cast(
            pl.Categorical))
    )
    energy_profiling_by_task = handle_optimizers_with_non_recurring_tasks_before_calculations(energy_profiling_by_task)
    energy_profiling_by_task = handle_batching_optimizers(energy_profiling_by_task, task_summaries)
    idle_col = pl.col("summary").struct.field("energy_measures").struct.field("idle")
    idle_power = idle_col.struct.field("total_energy").truediv(idle_col.struct.field("perf_duration"))
    # perf_duration = pl.col("energy_measures").struct.field("perf_duration")
    perf_duration = pl.col("perf_duration")
    # total_power = pl.col("energy_measures").struct.field("total_energy").truediv(perf_duration)
    total_power = pl.col("total_energy").truediv(perf_duration)
    marginal_power = total_power - idle_power
    power_share = marginal_power.truediv(pl.col("nb_concurrent_workers"))
    energy_usage_share = power_share * perf_duration
    average_energy_per_execution = energy_usage_share.truediv(pl.col("nb_energy_profiled_recalculated"))

    # breakpoint()
    energy_profiling_by_task = (
        energy_profiling_by_task
        .filter(pl.col("nb_energy_profiled") >= min_energy_profiled)
        # .drop_nulls(["nb_tests", "perf_duration"])
        .with_columns(
            average_energy_per_execution=average_energy_per_execution,
            energy_usage_share=energy_usage_share,
            power_share=power_share,
            marginal_power=marginal_power,
            total_power=total_power,
            idle_power=idle_power,
        )
    )
    return energy_profiling_by_task


def post_process_energy_df(energy_df):
    energy_df = energy_df.with_columns(
        cpu_energy=pl.col("cpu_energy").cast(pl.Float64),
        gpu_energy=pl.col("gpu_energy").cast(pl.Float64),
        start=pl.col("start"),
        end=pl.col("end"),
        duration=pl.col("duration"),
    )

    # Invalidate incorrect durations.
    thresh = 20
    energy_df = energy_df.with_columns(
        duration=pl.when((pl.col("duration") < thresh) & (pl.col("phase") != "idle"))
        .then(None)
        .otherwise(pl.col("duration")),
        perf_duration=pl.when((pl.col("duration") < thresh) & (pl.col("phase") != "idle"))
        .then(None)
        .otherwise(pl.col("perf_duration")),
        start=pl.when(pl.col("duration") < thresh).then(None).otherwise(pl.col("start")),
        end=pl.when(pl.col("duration") < thresh).then(None).otherwise(pl.col("end")),
    )
    return energy_df


def get_energy_per_processed_samples(df: pl.DataFrame):
    return df.with_columns(
        energy_per_processed_sample=pl.when(pl.col("phase") == "idle")
        .then(None)
        .when(pl.col("phase") == "generation")
        .then(pl.col("total_energy").truediv(pl.col("nb_samples")))
        .when(pl.col("phase") == "correctness")
        .then(pl.col("total_energy").truediv(pl.col("nb_samples")))
        .when(pl.col("phase") == "profiling")
        .then(pl.col("total_energy").truediv(pl.col("nb_profiled")))
    )


def get_energy_per_result(df: pl.DataFrame):
    return df.with_columns(
        energy_per_result=pl.when(pl.col("phase") == "idle")
        .then(None)
        .when(pl.col("phase") == "generation")
        .then(pl.col("total_energy").truediv(pl.col("nb_results")))
        .when(pl.col("phase") == "correctness")
        .then(pl.col("total_energy").truediv(pl.col("nb_results")))
        .when(pl.col("phase") == "profiling")
        .then(pl.col("total_energy").truediv(pl.col("nb_results")))
    )


def get_energy_by_phase(summaries, task_eval_results_raw):
    _energy = (
        summaries.with_columns(energy_measures=pl.col("energy_measures").crinkles.explode_struct_into_list("phase"))
        .explode("energy_measures")
        .unnest("energy_measures")
    )

    _energy = post_process_energy_df(_energy)

    energy_by_phase = (
        task_eval_results_raw.group_by(["file", "iteration"])
        .agg(
            pl.col("nb_samples").sum(),
            pl.col("nb_results").sum(),
            pl.col("nb_passed").sum(),
            pl.col("nb_profiled").sum(),
            pl.first("config_id"),
        )
        .join(_energy, on=["iteration", "file"])
    )

    energy_by_phase = get_energy_per_processed_samples(energy_by_phase).drop(pl.col("^*right$"))
    energy_by_phase = get_energy_per_result(energy_by_phase).drop(pl.col("^*right$"))
    return energy_by_phase


def handle_optimizers_with_non_recurring_tasks_before_calculations(energy_profiling_by_task):
    _key_cols = ["task_id", "optimizer", "model"]
    _optimizers = ["llm4effi"]
    _llm4effi_tasks = energy_profiling_by_task.filter(
        pl.col("config").struct.field("optimizer").is_in(_optimizers)).with_columns(
        iteration=pl.col("original_result_file").cast(pl.String).str.strip_suffix("_evalopti_results.json").str.tail(
            1).cast(pl.Int64),
        optimizer=pl.col("config").struct.field("optimizer"),
        model=pl.col("config").struct.field("model"),
    )
    if len(_llm4effi_tasks) == 0:
        return energy_profiling_by_task
    _llm4effi_tasks_pivot: pl.DataFrame = (
        _llm4effi_tasks.select("iteration", "results", pl.col(_key_cols))
        .sort("iteration")
        .pivot("iteration", index=_key_cols, values="results")
    )

    for i in range(0, 4):
        _llm4effi_tasks_pivot = _llm4effi_tasks_pivot.with_columns(
            pl.when(pl.col(str(i + 1)).is_null()).then(pl.col(str(i))).otherwise(
                pl.concat_list(str(i), str(i + 1))).alias(str(i + 1)))

    def propagate_stats(series: pl.Series):
        for i in range(1, len(series)):
            if series[i] == None or series[i] == 0:
                series[i] = series[i - 1]
        return series

    def sum_stats(series: pl.Series):
        for i in range(1, len(series)):
            series[i] += series[i - 1]
        return series

    _llm4effi_tasks_reconstituted = _llm4effi_tasks_pivot.unpivot(
        ["0", "1", "2", "3", "4"],
        index=_key_cols,
        variable_name="iteration",
        value_name="results",
    ).with_columns(iteration=pl.col("iteration").cast(pl.Int32)).join(
        _llm4effi_tasks.drop("results"), on=_key_cols + ["iteration"]).with_columns(
        pl.col("nb_concurrent_workers").map_batches(propagate_stats).over(
            _key_cols,
            order_by="iteration"),
        pl.col("nb_energy_profiled").map_batches(sum_stats).over(
            _key_cols,
            order_by="iteration")
    )

    energy_profiling_by_task = pl.concat(
        [energy_profiling_by_task.filter(pl.col("config").struct.field("optimizer").is_null() | (
            ~pl.col("config").struct.field("optimizer").is_in(_optimizers))),
         _llm4effi_tasks_reconstituted], how="diagonal_relaxed")
    return energy_profiling_by_task.drop("optimizer", "model", "iteration")


load_library()
