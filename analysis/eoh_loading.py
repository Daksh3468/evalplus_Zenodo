import polars as pl


def _load_sample_file(file):
    # return pl.DataFrame(stream_jsonl(file)) # 4 * slower
    with open(file, "r") as f:
        return pl.read_ndjson(f).with_columns(samples_path=pl.lit(file))


def get_full_sample_df(task_summaries):
    """Returns a sample-level df with information from summaries and generation files"""

    task_summaries_by_samples = (
        task_summaries.explode("results").drop("dps", "dps_norm", "num_cpu_instructions", "cpu_time").with_columns(
            results_copy="results").unnest("results")
    )
    sample_file_to_iteration = task_summaries.select("samples_path", "iteration").unique()
    samples = pl.concat(
        [
            _load_sample_file(row["samples_path"]).with_columns(iteration=pl.lit(row["iteration"]))
            for row in sample_file_to_iteration.iter_rows(named=True)
        ],
        how="vertical_relaxed",
    )
    return samples.join(task_summaries_by_samples, on=["sample_id", "iteration"])


def get_correct_eoh_task_summary(eoh_tasks):
    eoh_samples = (
        get_full_sample_df(eoh_tasks)
        .drop("batch_id")
        .unnest("extra_data")
        # .with_columns(generated_this_iteration=True)
        .with_columns(batch_id=pl.col("heuristic").struct.field("batch_id"))
        .drop("heuristic")
    )

    # Rebuild the task_df with these samples
    eoh_samples_full = (eoh_samples.group_by(["file", "task_id", "batch_id", "iteration"]).agg(
        nb_samples=pl.len(),
        nb_results=pl.lit(1),  # One batch is only one result
        nb_passed=pl.col("passed").sum(),
        nb_profiled=pl.col("profiled").sum(),
        best_sample=pl.col("sample_id").filter(
            pl.col("num_cpu_instructions") == pl.col("num_cpu_instructions").min()).first(),
        dps=pl.col("dps").max(),  # Per batch, the best value is the result
        dps_norm=pl.col("dps_norm").max(),
        cpu_time=pl.col("cpu_time").min(),
        num_cpu_instructions=pl.col("num_cpu_instructions").min(),
        samples_path=pl.col("samples_path").first(),
        results=pl.col("results_copy")
    ).with_columns(nb_batch_passed=pl.col("nb_passed") > 0)
                        # .with_columns(
                        #     nb_generated_this_iteration=pl.col("nb_samples"),
                        #     nb_passed_this_iteration=pl.col("nb_passed"),
                        #     nb_profiled_this_iteration=pl.col("nb_profiled")
                        # )
                        )
    eoh_samples_full = eoh_samples_full.group_by(["file", "task_id", "iteration"]).agg(
        pl.col("^nb_.*$").sum(),
        pl.col("dps", "dps_norm", "cpu_time", "num_cpu_instructions").mean(),
        pl.col("batch_id", "best_sample"),
        samples_path=pl.col("samples_path").first().cast(pl.Categorical),
        results=pl.col("results").flatten()
    )
    eoh_samples_full = eoh_samples_full.with_columns(
        pass_1=pl.col("nb_passed").truediv(pl.col("nb_samples")) * 100,
        pass_1_global=pl.col("nb_passed").truediv(pl.col("nb_samples")) * 100,
        pass_result_global=pl.col("nb_batch_passed").truediv(pl.col("batch_id").list.len()) * 100
    )
    eoh_samples_full = eoh_samples_full.join(eoh_tasks, on=["task_id", "iteration", "samples_path"]).drop(
        "^.*right$"
    )
    return eoh_samples_full
