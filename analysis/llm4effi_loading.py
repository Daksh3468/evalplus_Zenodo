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
    ).with_columns(samples_path=pl.col("samples_path").cast(pl.Categorical))
    return samples.join(task_summaries_by_samples, on=["sample_id", "iteration"])


def get_reformed_samples_for_llm4effi(llm4effi_samples):
    """
    Passed samples are not optimized in the next generation.
    So to get the overall stats by iteration, we need to rebuild each iteration samples with the passed samples from the previous iterations
    """
    prev_iter = llm4effi_samples.filter(iteration=0)
    llm4effi_samples_reformed = prev_iter
    for i in range(llm4effi_samples["iteration"].n_unique() - 1):
        samples_for_new_iter = llm4effi_samples.filter(iteration=i + 1)
        assert samples_for_new_iter["samples_path"].n_unique() == 1, samples_for_new_iter["samples_path"]
        df_for_iter = pl.concat(
            [
                prev_iter.filter(passed=True).with_columns(
                    iteration=pl.lit(i + 1),
                    generated_this_iteration=False,
                    file=pl.lit(samples_for_new_iter["file"].first()).cast(pl.Categorical),
                    samples_path=pl.lit(samples_for_new_iter["samples_path"].first()).cast(pl.Categorical),
                ),
                samples_for_new_iter,
            ],
            how="vertical",
        )
        llm4effi_samples_reformed = pl.concat([llm4effi_samples_reformed, df_for_iter], how="vertical")
        prev_iter = df_for_iter

    llm4effi_samples_reformed = llm4effi_samples_reformed.with_columns(
        passed_this_iteration=pl.col("generated_this_iteration") & pl.col("passed"),
        profiled_this_iteration=pl.col("generated_this_iteration") & pl.col("profiled"),
    )  # Will be useful when calculating energy data
    assert len(llm4effi_samples_reformed.unique(["sample_id", "iteration"])) == len(llm4effi_samples_reformed)
    return llm4effi_samples_reformed


def reassign_batch_id_to_samples(llm4effi_samples):
    iter_0_samples = llm4effi_samples.filter(iteration=0)
    sample_id_to_batch_id = {row["sample_id"]: row["batch_id"] for row in iter_0_samples.iter_rows(named=True)}
    return llm4effi_samples.with_columns(
        # batch_id = pl.col("sample_id").map_elements(lambda sample_id: sample_id_to_batch_id[sample_id], return_dtype=pl.String)
        batch_id=pl.col("sample_id").replace_strict(sample_id_to_batch_id)
    )


def get_correct_llm4effi_task_summary(llm4effi_tasks):
    llm4effi_samples = (
        get_full_sample_df(llm4effi_tasks)
        .drop("batch_id")
        .unnest("extra_data")
        .with_columns(generated_this_iteration=True)
    )
    llm4effi_samples = reassign_batch_id_to_samples(llm4effi_samples)
    llm4effi_samples_reformed = pl.concat(
        [get_reformed_samples_for_llm4effi(df) for df in llm4effi_samples.partition_by(["model", "optimizer"])],
        how="vertical")

    # Rebuild the task_df with these samples
    llm4effi_tasks_full = llm4effi_samples_reformed.group_by(["file", "task_id", "batch_id", "iteration"]).agg(
        nb_samples=pl.len(),
        nb_results=pl.lit(1),  # One batch is only one result
        nb_generated_this_iteration=pl.col("generated_this_iteration").sum(),
        nb_passed=pl.col("passed").sum(),
        nb_passed_this_iteration=pl.col("passed_this_iteration").sum(),
        nb_profiled=pl.col("profiled").sum(),
        nb_profiled_this_iteration=pl.col("profiled_this_iteration").sum(),
        best_sample=pl.col("sample_id").filter(
            pl.col("num_cpu_instructions") == pl.col("num_cpu_instructions").min()).first(),
        dps=pl.col("dps").max(),  # Per batch, the best value is the result
        dps_norm=pl.col("dps_norm").max(),
        cpu_time=pl.col("cpu_time").min(),
        num_cpu_instructions=pl.col("num_cpu_instructions").min(),
        samples_path=pl.col("samples_path").first(),
        results=pl.col("results_copy")
    ).with_columns(nb_batch_passed=pl.col("nb_passed") > 0)
    llm4effi_tasks_full = llm4effi_tasks_full.group_by(["file", "task_id", "iteration"]).agg(
        pl.col("^nb_.*$").sum(),
        pl.col("dps", "dps_norm", "cpu_time", "num_cpu_instructions").mean(),
        pl.col("batch_id", "best_sample"),
        samples_path=pl.col("samples_path").first().cast(pl.Categorical),
        results=pl.col("results").flatten()
    )
    llm4effi_tasks_full = llm4effi_tasks_full.with_columns(
        pass_1=pl.col("nb_passed").truediv(pl.col("nb_samples")) * 100,
        pass_1_global=pl.col("nb_passed").truediv(pl.col("nb_samples")) * 100,
        pass_result_global=pl.col("nb_batch_passed").truediv(pl.col("batch_id").list.len()) * 100
    )
    llm4effi_tasks_full = llm4effi_tasks_full.join(llm4effi_tasks, on=["task_id", "iteration", "samples_path"]).drop(
        "^.*right$"
    )
    return llm4effi_tasks_full
