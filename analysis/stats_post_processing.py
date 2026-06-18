import polars as pl

# from analysis.utils import remove_outliers

_stats_metrics = ["dps", "dps_norm", "cpu_time", "num_cpu_instructions", "average_energy_per_execution"]
_key_cols = ["task_id", "temperature", "model"]


def _get_speedup(stats_cols):  # When lower is better
    for col_name in stats_cols:
        old_col = pl.col(col_name + "_baseline")
        new_col = pl.col(col_name)
        yield (old_col.truediv(new_col).alias(col_name + "_speedup"))


def _get_reduction(stats_cols):  # When lower is better
    for col_name in stats_cols:
        old_col = pl.col(col_name + "_baseline")
        new_col = pl.col(col_name)
        yield (new_col.truediv(old_col).alias(col_name + "_reduction"))


def _get_improvement(col_name):  # When higher is better
    old_col = pl.col(col_name + "_baseline")
    new_col = pl.col(col_name)
    return (
        pl.when((new_col == 0) | (old_col == 0))
        .then(None)
        .otherwise(new_col.truediv(old_col).alias(col_name + "_improvement").fill_nan(None))
    )


def _get_baselines_for_every_task_and_model(possible_baselines):
    possible_baselines = possible_baselines.filter(pl.col("iteration") == 0, pl.col("optimizer").is_null()).select(
        pl.col(_key_cols), pl.col(_stats_metrics).name.suffix("_baseline")
    )
    return possible_baselines


def compute_speedup(task_summaries, drop_nulls=False):
    baselines = _get_baselines_for_every_task_and_model(task_summaries)
    task_results_with_baseline = task_summaries.join(baselines, on=["task_id", "model", "temperature"],
                                                     how="left").with_columns(
        is_potential_baseline=pl.when(pl.col("iteration") == 0, pl.col("optimizer").is_null()).then(True).otherwise(
            False)
    ).with_columns(
        is_baseline=pl.when(pl.col("is_potential_baseline"), pl.col("include_in_stats")).then(True).otherwise(False))
    task_results_with_speedup = (
        # task_summaries.filter(pl.col("iteration") == 0)  # FIX - Iteration 0 are excluded from final result !!
        # .select(pl.col(_key_cols), pl.col(_stats_metrics).name.suffix("_baseline"))
        # .join(task_summaries, on=_key_cols)
        task_results_with_baseline.with_columns(
            _get_speedup(["cpu_time", "num_cpu_instructions", "average_energy_per_execution"])
        )
        .with_columns(_get_reduction(["average_energy_per_execution"]))
        .with_columns(
            pl.when((pl.col("dps") == 0) | (pl.col("dps_baseline") == 0))
            .then(None)
            .otherwise(pl.col("dps").truediv(pl.col("dps_baseline")))
            .alias("dps_improvement"),
            pl.when((pl.col("dps_norm") == 0) | (pl.col("dps_norm_baseline") == 0))
            .then(None)
            .otherwise(pl.col("dps_norm").truediv(pl.col("dps_norm_baseline")))
            .alias("dps_norm_improvement"),
        )
        .with_columns(
            energy_gain_over_baseline_per_exec=pl.col("average_energy_per_execution_baseline")
                                               - pl.col("average_energy_per_execution")
        )
    )
    if drop_nulls:
        task_results_with_speedup = task_results_with_speedup.drop_nulls(["cpu_time", "cpu_time_baseline"])
    return task_results_with_speedup


def compute_accumulated_energy_use(summaries, col):
    return summaries.join(
        summaries.select("iteration", "config_id", col)
        .sort("iteration")
        .pivot("iteration", index="config_id", values=col)
        .with_columns(pl.cum_fold(acc=pl.lit(0), function=lambda acc, x: acc + x, exprs=pl.all().exclude("config_id")))
        .select("config_id", "cum_fold")
        .unnest("cum_fold")
        .unpivot(
            ["0", "1", "2", "3", "4"],
            index="config_id",
            variable_name="iteration",
            value_name=col + "_cumulative",
        )
        .with_columns(iteration=pl.col("iteration").cast(pl.Int32)),
        on=["config_id", "iteration"],
        how="left",
    )


def compute_energy_data_for_task_summaries(task_summaries, summaries, clip_outliers=True):
    # impossible_profitability_predicate = (pl.col("runs_necessary_to_justify_optimization") <= 0) | (
    #     pl.col("runs_necessary_to_justify_optimization").is_infinite())
    df = (
        task_summaries.join(
            summaries.select(
                "file",
                "total_energy",
                "energy_per_processed_sample",
                "energy_per_result",
                "total_energy_cumulative",
                "energy_per_processed_sample_cumulative",
                "energy_per_result_cumulative",
            ),
            on="file",
        )
        .with_columns(
            energy_expenditure_bottom_up=pl.col("energy_per_result") * pl.col("nb_results"),
            energy_expenditure_bottom_up_cumul=pl.col("energy_per_result_cumulative") * pl.col("nb_results"),
            energy_expenditure_top_down=pl.col("total_energy").truediv(task_summaries["task_id"].n_unique()),
            energy_expenditure_top_down_cumul=pl.col("total_energy_cumulative").truediv(
                task_summaries["task_id"].n_unique()
            ),
        )
        .with_columns(
            runs_necessary_to_justify_optimization=pl.col("energy_expenditure_bottom_up_cumul").truediv(
                "energy_gain_over_baseline_per_exec"
            ))
        # .with_columns(
        #     runs_necessary_to_justify_optimization_true=pl.when(
        #         pl.col("runs_necessary_to_justify_optimization") < 0).then(
        #         math.inf).otherwise(pl.col("runs_necessary_to_justify_optimization"))
        # )
        .with_columns(
            is_worst_than_baseline=pl.when(pl.col("runs_necessary_to_justify_optimization") < 0).then(True).otherwise(
                False),
            runs_necessary_to_justify_optimization=pl.when(
                pl.col("runs_necessary_to_justify_optimization") < 0).then(None).otherwise(
                pl.col("runs_necessary_to_justify_optimization"))
        )

    )
    if clip_outliers:
        df = df.drop("runs_necessary_to_justify_optimization").join(
            remove_outliers(
                df.select("file", "task_id", "runs_necessary_to_justify_optimization"),
                "runs_necessary_to_justify_optimization",
                0.05,
            ),
            on=["file", "task_id"],
            how="left",
        )
    return df


def remove_outliers(df, col, percentage):
    first_val = df[col].quantile(percentage)
    last_val = df[col].quantile(1 - percentage)
    return df.filter((pl.col(col) > first_val) & (pl.col(col) < last_val))
