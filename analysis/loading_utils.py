"""
Utility functions for loading and processing task summary data in the analysis pipeline.

This module provides functions to load summary files, process result files, and compute
various statistics for task summaries. It handles special optimizers and post-processing
of data to ensure consistent and accurate analysis results.
"""
import json
import os
import sys

import polars as pl

from energy_post_processing import include_energy_data, get_energy_by_phase
from eoh_loading import get_correct_eoh_task_summary
from llm4effi_loading import get_correct_llm4effi_task_summary
from simple10_loading import get_correct_simple10_task_summary
from stats_post_processing import compute_speedup, compute_accumulated_energy_use, \
    compute_energy_data_for_task_summaries

# Columns that should be excluded from processing as they're not relevant for analysis
unused_columns = [
    "iterations",
    # "n_samples",
    "min_correct",
    "max_profile",
    "root",
    "nb_workers_wanted",
    "i_just_wanna_run",
    "test_single_task",
    "bs",
    "backend",
    "base_url",
]


def load_summary_file(path, root, default_root):
    with open(path, "r") as f:
        file_df_raw = pl.read_json(f)
    config_id = json.dumps(file_df_raw.select("config").to_dicts()[0]["config"])
    file_df = (
        file_df_raw.select("summary", "config")
        .unnest("config")
        .drop(unused_columns)
        .explode("summary")
        .unnest("summary")
        .with_columns(config_id=pl.lit(config_id), summary_file=pl.lit(path))
        .with_columns(pl.col("dps_norm").round(4))
    )
    if "iteration" not in file_df.columns:
        print("Iteration column not present. Adding iterations manually.")
        file_df = file_df.with_columns(iteration=pl.arange(0, file_df.height))
    return file_df


#
# def post_process_config(df: pl.DataFrame):
#     if "accept_errored_samples_in_optimizer" not in df.columns:
#         print("Missing config value : 'accept_errored_samples_in_optimizer', setting default: False")
#         df = df.with_columns(accept_errored_samples_in_optimizer=pl.lit(False))
#     return df
#
#
# def _fix_broken_file_paths(df, col, root, default_root):
#     if root != default_root and col in df.columns:
#         print("Fixing incorrect path")
#         df = df.with_columns(
#             file=pl.when(pl.col(col).str.contains(default_root))
#             .then(pl.col(col).str.replace(default_root, root, literal=True))
#             .otherwise(pl.col(col))
#         )
#     return df


def get_all_results_files_from_summary_df(summary_df):
    return summary_df["file"].unique()


# @mo.cache
def _load_result_file(result_file):
    if not os.path.exists(result_file):
        print(f"Result file {result_file} does not exist! Skiping.", file=sys.stderr)
        return pl.LazyFrame()
    with open(result_file, "r") as f:
        file_df_raw = pl.read_json(f)
    config_id = json.dumps(
        file_df_raw.select("config").to_dicts()[0]["config"])  # Basically, convert to a string

    result_df = (
        file_df_raw.select(pl.col("eval").crinkles.struct_to_list(), pl.col("config"))
        .unnest("config")
        .explode("eval")
        .unnest("eval")
        .with_columns(
            pl.lit(str(result_file)).alias("file"),
            pl.lit(config_id).alias("config_id"),
            # pl.col("config").cast(pl.Categorical)
        )
        .drop("ref")
        .drop(unused_columns)
    )
    return result_df


def result_file_extra_processing(res_df, summaries):
    res_df = res_df.with_columns(
        nb_samples=pl.col("results").list.len(),
        nb_results=pl.col("results").list.len(),  # For the basic methods, each sample is one final result
        nb_passed=pl.col("results").list.eval(pl.element().struct.field("passed")).list.sum(),
        nb_profiled=pl.col("results").list.eval(pl.element().struct.field("profiled")).list.sum(),
        batch_id=None,
    ).join(summaries.select("file", "iteration"), on=["file"])
    return res_df


def remove_optimizer_for_first_iteration(res_df):
    excluded_optimizers = ["eoh", "llm4effi"]
    return res_df.with_columns(
        optimizer=pl.when((pl.col("iteration") == 0) & (~pl.col("optimizer").is_in(excluded_optimizers)))
        .then(None)
        .otherwise(pl.col("optimizer"))
    )


def deduplicate_task_df(task_df):
    """Useful when loading the same file multiple times on accident."""
    return task_df.unique(["task_id", "file", "batch_id", "optimizer"])
    # return task_df


def handle_special_optimizers(task_summaries):
    task_summaries = task_summaries.with_columns(
        pass_result_global=pl.col("pass_1_global")
    )
    llm4effi_tasks = task_summaries.filter(optimizer="llm4effi")
    if len(llm4effi_tasks) > 0:
        llm4effi_tasks_correct = get_correct_llm4effi_task_summary(llm4effi_tasks)
        task_summaries = pl.concat(
            [
                task_summaries.filter(pl.col("optimizer").is_null() | (pl.col("optimizer") != "llm4effi")),
                llm4effi_tasks_correct,
            ],
            how="diagonal_relaxed",
        )

    eoh_tasks = task_summaries.filter(optimizer="eoh")
    if len(eoh_tasks) > 0:
        eoh_tasks_correct = get_correct_eoh_task_summary(eoh_tasks)
        task_summaries = pl.concat(
            [
                task_summaries.filter(pl.col("optimizer").is_null() | (pl.col("optimizer") != "eoh")),
                eoh_tasks_correct,
            ],
            how="diagonal_relaxed",
        )

    simple10_predicate = (pl.col("iteration") > 0) & (pl.col("optimizer") == "simple10")
    simple10_tasks = task_summaries.filter(simple10_predicate)
    if len(simple10_tasks) > 0:
        all_task_ids = set(task_summaries["task_id"].unique())
        simple10_tasks_correct = get_correct_simple10_task_summary(simple10_tasks, all_task_ids)
        task_summaries = pl.concat(
            [
                task_summaries.filter(pl.col("optimizer").is_null() | ~simple10_predicate),
                simple10_tasks_correct,
            ],
            how="diagonal_relaxed",
        )

    return task_summaries


def load_all_result_files(summaries):
    result_files_list = set(get_all_results_files_from_summary_df(summaries))
    task_summaries = pl.DataFrame()
    for result_file in result_files_list:
        result_file_df = _load_result_file(result_file)
        task_summaries = pl.concat([task_summaries, result_file_df], how="diagonal_relaxed")
    task_summaries = task_summaries.with_columns(file=pl.col("file").cast(pl.Categorical),
                                                 samples_path=pl.col("samples_path").cast(pl.Categorical))
    # task_summaries = task_summaries.collect()
    task_summaries = result_file_extra_processing(task_summaries, summaries)
    task_summaries = handle_special_optimizers(task_summaries)
    task_summaries = remove_optimizer_for_first_iteration(task_summaries)
    task_summaries = deduplicate_task_df(task_summaries)
    return task_summaries


def filter_summaries(summaries: pl.DataFrame, analysis_filter: dict):
    if analysis_filter["only_one_temperature"]:
        summaries = summaries.filter(pl.col("temperature") == analysis_filter["temperature"])
    return summaries


def exclude_tasks_when_nb_profiled_low(task_summaries, summaries, min_profiled):
    # Exclude tasks based on the nb_profiled samples
    task_summaries = task_summaries.with_columns(
        include_in_stats=pl.when(pl.col("nb_profiled") >= min_profiled).then(True).otherwise(False)
    )
    _stats_col_task_level = ["dps", "dps_norm", "cpu_time", "num_cpu_instructions"]
    task_summaries_partitioned = task_summaries.partition_by("include_in_stats", as_dict=True)
    task_summaries_included, task_summaries_excluded = task_summaries_partitioned[(True,)], task_summaries_partitioned[
        (False,)]
    # Wipe the stats from the excluded tasks
    task_summaries_excluded = task_summaries_excluded.with_columns(
        pl.col(_stats_col_task_level).map_elements(lambda col: None, return_dtype=pl.Float64)
    )
    task_summaries = pl.concat([task_summaries_included, task_summaries_excluded], how="vertical")

    # Update summaries based on the previously excluded stats
    _stats_col = ["dps", "dps_norm"]
    _new_summaries_stats = (
        task_summaries.filter("include_in_stats")
        .group_by("file")
        .agg(pl.col(_stats_col).mean(), pl.len().alias("nb_tasks_profiled"))
    )
    summaries = summaries.drop(_stats_col).join(_new_summaries_stats, on="file", how="left")
    return task_summaries, summaries


def correct_pass_1_on_task_and_summaries(task_summaries, summaries):
    task_summaries = task_summaries.with_columns(pl.col("^pass.*$", "nb_profiled").fill_null(0))
    summaries = summaries.drop("pass@1", "pass@1_global").join(
        task_summaries.group_by("file").agg(
            pl.col("pass_1").mean().alias("pass@1"), pl.col("pass_1_global").mean().alias("pass@1_global"),
            pl.col("pass_result_global").mean().alias("pass@result_global"),
        ),
        on="file",
        how="left",
    )
    return task_summaries, summaries


def _remove_outliers_expr(col_name, percentage):
    return pl.col(col_name).filter((pl.col(col_name) < pl.col(col_name).quantile(1 - percentage / 2)) & (
            pl.col(col_name) > pl.col(col_name).quantile(percentage / 2)))


def propagate_stats_from_task_to_whole_summary(task_summaries, summaries, outliers_percent):
    summaries = summaries.join(
        task_summaries.group_by("file").agg(
            pl.col(
                "average_energy_per_execution_speedup",
                "average_energy_per_execution_reduction",
                "cpu_time_speedup",
                "dps_norm_baseline",
                # "dps_norm_improvement",
                # "runs_necessary_to_justify_optimization",
                "energy_expenditure_bottom_up_cumul",
                "energy_gain_over_baseline_per_exec",
                "idle_power"
            ).mean().round(4),
            _remove_outliers_expr("average_energy_per_execution_speedup", outliers_percent).mean().round(4).alias(
                "average_energy_per_execution_speedup_no_outliers"),
            _remove_outliers_expr("average_energy_per_execution_reduction", outliers_percent).mean().round(4).alias(
                "average_energy_per_execution_reduction_no_outliers"),
            pl.col("runs_necessary_to_justify_optimization").median().alias(
                "runs_necessary_to_justify_optimization_median").round(1),
            pl.col("runs_necessary_to_justify_optimization").mean().alias(
                "runs_necessary_to_justify_optimization_average").round(1),
            _remove_outliers_expr("runs_necessary_to_justify_optimization", outliers_percent).mean().round(4).alias(
                "runs_necessary_to_justify_optimization_no_outliers"),
            (~pl.col("is_worst_than_baseline")).sum().alias(
                "nb_task_where_break_even_point_is_possible"),
            (pl.col("is_worst_than_baseline")).sum().alias(
                "nb_task_where_break_even_point_is_not_possible")
        ).with_columns(
            total=pl.col("nb_task_where_break_even_point_is_possible")
                  + pl.col("nb_task_where_break_even_point_is_not_possible")
        ).with_columns(prop_possible_profitability=pl.col("nb_task_where_break_even_point_is_possible").truediv(
            pl.col("total")).round(3),
                       prop_impossible_profitability=pl.col("nb_task_where_break_even_point_is_not_possible").truediv(
                           pl.col("total")).round(3)),
        on="file",
        how="left",
    ).with_columns(
        dps_norm_improvement=(pl.col("dps_norm") / pl.col("dps_norm_baseline")).round(4)
    )
    return summaries


def null_results_with_pass_1_too_low(summaries, threshold):
    def null_results_for_col(col, thresh):
        return pl.when(pl.col("pass@1_global") > thresh).then(pl.col(col)).alias(col)

    cols = [
        "average_energy_per_execution_speedup",
        "average_energy_per_execution_reduction",
        "runs_necessary_to_justify_optimization_median",
        "dps_norm_improvement",
        "cpu_time_speedup",
    ]
    exprs = [null_results_for_col(col, threshold) for col in cols]

    summaries = summaries.with_columns(exprs)
    return summaries


def list_available_summary_files(root, keyword="summary") -> list[str]:
    all_files = os.listdir(root)
    return [file for file in all_files if keyword in file]


def load_all_summaries(root, default_root):
    # Loading all files
    summary_files = list_available_summary_files(root)
    summaries_raw = pl.concat(
        [load_summary_file(os.path.join(root, file), root=root, default_root=default_root) for file in summary_files],
        how="diagonal_relaxed",
    )
    summaries_raw = summaries_raw.with_columns(file=pl.col("file").cast(pl.Categorical))
    return summaries_raw


def post_process_task_summaries_and_global_summaries(task_summaries_raw, summaries, min_profiled, root,
                                                     outliers_percent):
    task_summaries, summaries = correct_pass_1_on_task_and_summaries(
        *exclude_tasks_when_nb_profiled_low(
            task_summaries_raw, summaries, min_profiled=min_profiled
        ))
    task_summaries = include_energy_data(task_summaries, root, min_energy_profiled=min_profiled)
    task_summaries = compute_speedup(task_summaries)
    energy_by_phase = get_energy_by_phase(summaries, task_summaries_raw)

    summaries = summaries.join(
        energy_by_phase.group_by("file").agg(
            pl.col("total_energy").sum(), pl.col("energy_per_processed_sample", "energy_per_result").sum()
        ),
        on="file",
        how="left",
    )

    summaries = compute_accumulated_energy_use(summaries, "total_energy")
    summaries = compute_accumulated_energy_use(summaries, "energy_per_processed_sample")
    summaries = compute_accumulated_energy_use(summaries, "energy_per_result")

    task_summaries = compute_energy_data_for_task_summaries(task_summaries, summaries, clip_outliers=False)

    summaries = propagate_stats_from_task_to_whole_summary(task_summaries, summaries, outliers_percent)
    summaries = null_results_with_pass_1_too_low(summaries, 30)

    task_summaries = deduplicate_task_df(task_summaries)
    return task_summaries, summaries, energy_by_phase


def explode_task_summaries_results(task_summaries: pl.DataFrame):
    return (
        task_summaries.lazy()
        .explode("results")
        .drop(["dps", "dps_norm", "num_cpu_instructions", "cpu_time"])
        .unnest("results")
        .drop(
            "pass_1",
            "pass_1_global",
            "n_profiled",
            "nb_samples",
            "nb_results",
            "nb_profiled",
            "solution",
            strict=False,
        )
        .collect()
    )


def load_sample_file(file):
    # return pl.DataFrame(stream_jsonl(file)) # 4 * slower
    return pl.scan_ndjson(open(file, "r")).with_columns(samples_path=pl.lit(file).cast(pl.Categorical))


def _get__samples_generation_details_df(summary_df):
    sample_files = (
        summary_df.select(pl.col("file").unique())
        .drop_nulls()
        .with_columns(pl.all().cast(pl.String).str.replace("_evalopti_results.json", ".jsonl").alias("sample_file"))
    )
    sample_df = (
        pl.concat(
            [
                load_sample_file(row["sample_file"])
                .with_columns(file=pl.lit(row["file"]).cast(pl.Categorical))
                .drop("solution")
                for row in sample_files.iter_rows(named=True)
            ],
            how="diagonal_relaxed",
        )
        .unnest("usage_share")
        .collect()
    )
    return sample_df


def get_samples_eval_results(task_summaries, summaries):
    _samples_generation_details = _get__samples_generation_details_df(summaries)

    return explode_task_summaries_results(
        task_summaries.select(
            pl.col(["file", "dps", "dps_norm", "num_cpu_instructions", "cpu_time", "results"]), pl.col("^*tokens$")
        )
    ).join(_samples_generation_details, on=(["file", "sample_id"]))
