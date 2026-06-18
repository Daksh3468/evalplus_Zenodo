import marimo

__generated_with = "0.13.9"
app = marimo.App(width="medium", app_title="Crinkles Analysis")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import polars as pl
    import seaborn as sns
    import altair as alt
    import numpy as np
    import itertools
    import sys
    from utils import remove_outliers, load_library
    from okabe_ito_theme import okabe_ito_theme
    import resource
    from great_tables import GT


    def memory_limit_half():
        """Limit max memory usage to half."""
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        print(f"Soft: {soft / 1024**3} Hard: {hard}")
        # Convert KiB to bytes, and divide in two to half
        free_memory = get_free_memory() * 1024
        allocated_memory = int(free_memory * 1.8)
        print(f"Free {round(free_memory / (1024**3), 3)} Allocated: {round(allocated_memory / (1024**3), 3)}")
        resource.setrlimit(resource.RLIMIT_AS, (allocated_memory, hard))


    def get_free_memory():
        with open("/proc/meminfo", "r") as mem:
            free_memory = 0
            for i in mem:
                sline = i.split()
                if str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                    free_memory += int(sline[1])
        return free_memory  # KiB


    load_library()
    # memory_limit_half()

    pl.enable_string_cache()

    alt.themes.register("okabe-ito", okabe_ito_theme)
    alt.themes.enable("okabe-ito")


@app.cell
def _():
    mo.md(
        r"""
    # When Faster Isn’t Greener: The Hidden Costs of LLM-Based Code Optimization - Analysis Companion Notebook

    Welcome! This is the companion notebook for the paper "When Faster Isn’t Greener: The Hidden Costs of LLM-Based Code Optimization". You can find all of the figures and tables used in the paper in this notebook, and more!

    This notebook requires aproximatively 10 Gio of memory to run.
    """
    )
    return


@app.cell(hide_code=True)
def _(analysis_options, debug_mode_checkbox, display_options, root_selector):
    mo.md(
        f"""
    ## Notebook configuration

    {mo.as_html(debug_mode_checkbox)}

    {mo.as_html(root_selector)}

    #### Analysis options
        {analysis_options}

    #### Display options
        {display_options}
    """
    )
    return


@app.cell
def _():
    # Config step 1
    default_root = "evalplus_results/evalperf"
    available_roots = [default_root, "evalplus_results/_test/evalperf", "evalplus_results/invalid"]

    root_selector = mo.ui.dropdown(
        options=available_roots,
        value=available_roots[0],
        allow_select_none=False,
        label=f"Root directory",
    )

    debug_mode_checkbox = mo.ui.checkbox(label="Debug mode (show intermediary data)", value=True)

    analysis_options = mo.ui.dictionary(
        {
            "only_one_temperature": mo.ui.checkbox(label="Analyze only one temperature", value=True),
            "temperature": mo.ui.slider(
                label="Temperature",
                steps=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                show_value=True,
                value=0.2,
            ),
            "min_profiled": mo.ui.slider(
                label="Minimum profiled sample for each task of an iteration to be considered in the stats",
                steps=range(0, 51),
                show_value=True,
                value=3,
                debounce=True,
            ),
        },
        label="Analysis options",
    )
    return analysis_options, debug_mode_checkbox, default_root, root_selector


@app.cell
def _():
    display_options = mo.ui.dictionary(
        {
            "convert_j_to_wh": mo.ui.checkbox(
                label="Convert Joules to Watt-hours (only in figures)",
                value=True,
            ),
        }
    )
    return (display_options,)


@app.cell
def _(debug_mode_checkbox, display_options):
    def convert_to_correct_unit(df, cols):
        if display_options.value["convert_j_to_wh"]:
            df = df.with_columns(pl.col(cols).truediv(3600))
        return df


    if display_options.value["convert_j_to_wh"]:
        energy_unit = "Wh"
    else:
        energy_unit = "J"


    def show_if_debug_mode(output):
        if debug_mode_checkbox.value:
            return output
        else:
            return None
    return convert_to_correct_unit, energy_unit, show_if_debug_mode


@app.cell
def _(root_selector):
    # Config step 2
    from loading_utils import list_available_summary_files

    root = root_selector.value
    available_summary_files = list_available_summary_files(root_selector.value)
    return (root,)


@app.cell
def _(analysis_options, default_root, root):
    from loading_utils import (
        load_all_summaries,
        filter_summaries,
        load_all_result_files,
        post_process_task_summaries_and_global_summaries,
        get_samples_eval_results,
    )
    import energy_post_processing as en


    def set_iteration_from_filename():
        iteration_pattern = (
            pl.col("original_result_file")
            .cast(pl.String)
            .str.strip_suffix("_evalopti_results.json")
            .str.tail(1)
            .cast(pl.Int64)
        )
        return (
            pl.when(pl.col("original_result_file").cast(pl.String).str.contains(pl.col("config").struct.field("optimizer")))
            .then(iteration_pattern)
            .otherwise(pl.lit(0))
        )


    with mo.status.progress_bar(
        range(6),
        title="Loading the data",
        subtitle="Loading summaries",
        show_rate=False,
        remove_on_exit=True,
        show_eta=False,
    ) as pbar:
        summaries_raw = load_all_summaries(root, default_root)
        pbar.update(subtitle="Filtering summaries")
        summaries_filtered = filter_summaries(summaries_raw, analysis_options.value)
        pbar.update(subtitle="Loading task results")
        task_summaries_raw = load_all_result_files(summaries_filtered)
        pbar.update(subtitle="Post processing the results")

        task_summaries, summaries, energy_by_phase = post_process_task_summaries_and_global_summaries(
            task_summaries_raw,
            summaries_filtered,
            min_profiled=analysis_options["min_profiled"].value,
            root=root,
            outliers_percent=0.10,
        )

        pbar.update(subtitle="Loading sample-level results")
        samples_eval_results = get_samples_eval_results(task_summaries, summaries)

        pbar.update(subtitle="Loading task energy profiling data")
        energy_debug_df = en.get_energy_profiling_by_task(
            en._get_all_energy_profiling_files(root), 3, task_summaries
        ).with_columns(
            iteration=set_iteration_from_filename(),
            model=pl.col("config").struct.field("model"),
            optimizer=pl.col("config").struct.field("optimizer"),
        )
        pbar.update(subtitle="Done!")
    return (
        energy_by_phase,
        energy_debug_df,
        samples_eval_results,
        summaries,
        summaries_raw,
        task_summaries,
    )


@app.cell(hide_code=True)
def _(energy_by_phase, samples_eval_results, summaries, task_summaries):
    mo.md(
        f"""
    ## Raw results

    Here, we present the raw results from our experiment. You can find the various dataframes that we used to obtain our visualizations

    {
            mo.ui.tabs(
                {
                    "Configuration-level results": mo.as_html(
                        mo.ui.table(summaries, show_column_summaries=True, max_columns=None, selection=None)
                    ),
                    "Task-level results": mo.lazy(
                        mo.ui.table(task_summaries, show_column_summaries=True, max_columns=None, selection=None)
                    ),
                    "Sample-level results": mo.lazy(mo.as_html(samples_eval_results)),
                    "Energy results - Optimization": mo.lazy(mo.as_html(energy_by_phase)),
                }
            )
        }
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Research questions""")
    return


@app.cell
def _(best_model, summaries_pretty):
    measured_energy_type_dropdown = mo.ui.dropdown(
        options={
            "energy_per_result": "Energy cost of one optimization in an iteration",
            "energy_per_result_cumulative": "Cumulative energy cost of one optimization",
            "energy_per_processed_sample": "Energy to process one sample once",
        },
        value="energy_per_result_cumulative",
        label="Energy metric for figures",
    )

    which_optimization_iteration_to_consider_dropdown = mo.ui.dropdown(
        options={
            "At last iteration": {"iterations_to_select": [4], "iterations_file_label": "last_iter"},
            "At first optimization iteration (iteration-1)": {"iterations_to_select": [1], "iterations_file_label": "first_opti_iter"},
            "At second optimization iteration (iteration-2)": {"iterations_to_select": [2], "iterations_file_label": "first_opti_iter"},
            "Average of all iteration": {"iterations_to_select": [0, 1, 2, 3, 4], "iterations_file_label": "all_iter"},
        },
        value="At last iteration",
        label="When possible, at which iteration should the analysis be done?",
    )

    consider_only_best_model_checkbox = mo.ui.checkbox(
        label=f"Consider only the best model ({best_model}) when aggregating results. Otherwise, results are averaged across models",
        value=False,
    )

    model_for_energy_proportion_study_dropdown = mo.ui.dropdown(
        options=summaries_pretty["model"].unique(), value="Qwen2.5-Coder-14B"
    )  # Used later

    inverse_energy_speedup_checkbox = mo.ui.checkbox(
        label=f"Inverse energy speedup",
        value=False,
    )

    mo.md(f"""
    ### Results display configuration
    {measured_energy_type_dropdown}

    {which_optimization_iteration_to_consider_dropdown}

    {consider_only_best_model_checkbox}
    """)
    return (
        consider_only_best_model_checkbox,
        measured_energy_type_dropdown,
        model_for_energy_proportion_study_dropdown,
        which_optimization_iteration_to_consider_dropdown,
    )


@app.cell
def _(
    best_model,
    consider_only_best_model_checkbox,
    measured_energy_type_dropdown,
    summaries_pretty,
    which_optimization_iteration_to_consider_dropdown,
):
    measured_energy_type = measured_energy_type_dropdown.selected_key
    measured_energy_label = measured_energy_type_dropdown.value
    iterations_to_select = which_optimization_iteration_to_consider_dropdown.value["iterations_to_select"]
    iteration_label = which_optimization_iteration_to_consider_dropdown.selected_key
    iteration_file_label = which_optimization_iteration_to_consider_dropdown.value["iterations_file_label"]

    if consider_only_best_model_checkbox.value:
        models_to_select = [best_model]
        models_label = best_model
        models_file_label = "best_model"
    else:
        models_to_select = summaries_pretty["model"].unique()
        models_label = "all models"
        models_file_label = "all_models"
    return (
        iteration_file_label,
        iteration_label,
        iterations_to_select,
        measured_energy_label,
        measured_energy_type,
        models_file_label,
        models_label,
        models_to_select,
    )


@app.cell(disabled=True)
def _(summaries_pretty, summary_table):
    _results_details = None
    if len(summary_table.value) > 0:
        _summary_selection = summary_table.value[0]
        _results_details = summaries_pretty.join(_summary_selection.select("model", "optimizer"), on=["model", "optimizer"])

    if _results_details is None:
        _results_details = "No result selected"

    mo.md(f"""
    ### Result synthesis. Comparison of every model and optimizer across iterations on performance, correctness and cost
    {mo.as_html(mo.vstack([summary_table, _results_details]))}
    """)
    return


@app.cell
def _(
    energy_unit,
    iteration_label,
    measured_energy_label,
    measured_energy_type,
    summaries_by_optimizer_and_model,
):
    _rename = {
        "model": "Model",
        "optimizer": "Method",
        "prop_possible_profitability": "Efficient@1",  # > 50% of non-null BEP
        "average_energy_per_execution_reduction_no_outliers": "Energy Reduction",
        "runs_necessary_to_justify_optimization_no_outliers": "BEP",
        "pass@1_global": "Pass@1",
        "dps_norm": "DPS Final",
        "dps_norm_delta": "DPS delta",
        "dps_norm_baseline": "DPS Baseline",
        measured_energy_type: measured_energy_label + f" ({energy_unit})",
    }


    mo.md(
        f"""### Result synthesis
    Summary of all the results into one table ({iteration_label})

    {mo.as_html(summaries_by_optimizer_and_model.select(_rename.keys()).rename(_rename))}
    """
    )
    return


@app.cell
def _(
    energy_unit,
    iteration_label,
    measured_energy_type,
    models_label,
    summaries_by_optimizer_and_iteration,
):
    _rename = {
        # "model": "Model",
        "optimizer": "Method",
        "iteration": "Iteration",
        "prop_possible_profitability": "Efficient@1",  # > 50% of non-null BEP
        "average_energy_per_execution_reduction_no_outliers": "Energy Reduction",
        measured_energy_type: "Energy / result" + f" ({energy_unit})",
        "runs_necessary_to_justify_optimization_no_outliers": "BEP",
        "pass@1_global": "Pass@1",
        "pass@result_global": "Pass@Result",
        "dps_norm": "DPSNorm",
        "dps_norm_delta": "DPSNormDelta",
    }

    bep_optimizer_table = (
        GT(summaries_by_optimizer_and_iteration.filter(pl.col("iteration").is_in([2,4])).sort("iteration").select(_rename.keys()))
        .fmt_number("runs_necessary_to_justify_optimization_no_outliers", decimals=0)
        .fmt_number(measured_energy_type, decimals=2)
        .fmt_number(["pass@1_global", "pass@result_global", "dps_norm"], decimals=1)
        .cols_label(_rename)
    )

    print(
        bep_optimizer_table.as_latex()
        .replace("\\toprule", f"\\toprule % {models_label} {iteration_label}")
        .replace("DPSNormDelta", "\\dpsnormdelta")
        .replace("DPSNorm", "\\dpsnorm")
    )

    bep_optimizer_table = bep_optimizer_table.tab_header(
        title=f"Synthetic results by optimizer (no outliers)",
        subtitle=f"With {models_label} - {iteration_label}",
    )

    # mo.md(f"""
    # {mo.as_html(energy_per_optimizer_over_iterations_table)}""")
    bep_optimizer_table
    return


@app.cell
def _(energy_unit, iteration_label, measured_energy_type, summaries_by_model):
    _rename = {
        "model": "Model",
        # "optimizer": "Optimization Method",
        "prop_possible_profitability": "Efficient@1",  # > 50% of non-null BEP
        # "runs_necessary_to_justify_optimization_median": "Median BEP",
        # "runs_necessary_to_justify_optimization_average": "Mean BEP",
        "average_energy_per_execution_reduction_no_outliers": "Energy Reduction",
        measured_energy_type: "Optimization cost / result" + f" ({energy_unit})",
        "runs_necessary_to_justify_optimization_no_outliers": "BEP",
        # "average_energy_per_execution_speedup": "Energy Speedup",
        # "average_energy_per_execution_reduction": "Energy Reduction",
        # "average_energy_per_execution_speedup_no_outliers": "Energy Speedup w/o outliers",
        "pass@1_global": "Pass@1",
        "pass@result_global": "Pass@result",
        "dps_norm": "DPSNorm",
        "dps_norm_delta": "DPSNormDelta",
    }

    bep_model_table = (
        GT(summaries_by_model.select(_rename.keys()))
        .fmt_number("runs_necessary_to_justify_optimization_no_outliers", decimals=0)
        .fmt_number(measured_energy_type, decimals=2)
        .fmt_number(["pass@1_global", "pass@result_global", "dps_norm"], decimals=1)
        .cols_label(_rename)
    )

    print(
        bep_model_table.as_latex()
        .replace("\\toprule", f"\\toprule % {iteration_label}")
        .replace("DPSNormDelta", "\\dpsnormdelta")
        .replace("DPSNorm", "\\dpsnorm")
    )

    bep_model_table = bep_model_table.tab_header(
        title=f"Synthetic results by model (no outliers)",
        subtitle=f"{iteration_label}",
    )

    bep_model_table
    return


@app.cell
def _(convert_to_correct_unit, summaries):
    model_pretty = {
        "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen2.5-Coder-7B",
        "Qwen/Qwen2.5-Coder-3B-Instruct": "Qwen2.5-Coder-3B",
        "Qwen/Qwen2.5-Coder-14B-Instruct": "Qwen2.5-Coder-14B",
        "bigcode/starcoder2-15b-instruct-v0.1": "StarCoder2",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "DeepSeek-R1-14B",
    }

    optimizer_pretty = {
        "self-refine-exec-feedback": "Self-Refine-Exec",
        "cod": "CoD",
        "llm4effi": "LLM4EFFI",
        "simple": "Simple",
        "simple10": "Simple10",
        "eoh": "EoH",
        "cot": "CoT",
        "self-refine-nl-feedback": "Self-Refine-NL",
    }

    pretty_names = model_pretty | optimizer_pretty


    def pretty_summaries(summaries):
        return (
            convert_to_correct_unit(summaries, ["^energy_per.*$", "^energy_gain.*$"])
            .with_columns(pl.col("model", "optimizer").replace(pretty_names))
            .sort("optimizer", "model")
        )


    summaries_pretty = pretty_summaries(summaries)
    return pretty_names, pretty_summaries, summaries_pretty


@app.cell
def _(
    iterations_to_select,
    measured_energy_type,
    models_to_select,
    pretty_names,
    summaries_pretty,
):
    _agg_stats = [
        pl.col(
            "runs_necessary_to_justify_optimization_median",
            "runs_necessary_to_justify_optimization_average",
            "runs_necessary_to_justify_optimization_no_outliers",
        )
        .mean()
        .round(0),  # Should cast to int, but can't because of the `inf` values
        pl.col("dps_norm", "dps_norm_baseline", "pass@1_global", "pass@result_global").mean().round(2),
        pl.col(
            "prop_possible_profitability",
            measured_energy_type,
            "average_energy_per_execution_speedup",
            "average_energy_per_execution_reduction",
            "average_energy_per_execution_speedup_no_outliers",
            "average_energy_per_execution_reduction_no_outliers",
        )
        .mean()
        .round(3),
        pl.col("cpu_time_speedup").median().round(2),
    ]

    _dps_delta = (pl.col("dps_norm") - pl.col("dps_norm_baseline")).round(3).alias("dps_norm_delta")

    summaries_by_optimizer_and_iteration = (
        summaries_pretty.filter(pl.col("model").is_in(models_to_select))
        .group_by("optimizer", "iteration")
        .agg(
            *_agg_stats,
            pl.col("model").n_unique().alias("nb_models"),
        )
        .with_columns(pl.col("optimizer").replace(pretty_names))
        .sort("optimizer", "iteration")
    ).with_columns(_dps_delta)

    # _baseline_dps = summaries_by_optimizer_and_iteration.filter(iteration=0, optimizer=pretty_names["simple"])["dps_norm"]

    summaries_by_optimizer_and_model = (
        summaries_pretty.filter(pl.col("iteration").is_in(iterations_to_select))
        .group_by("model", "optimizer")
        .agg(
            *_agg_stats,
        )
        .with_columns(_dps_delta)
        .sort("model", "optimizer")
    )

    summaries_by_model = (
        summaries_pretty.filter(pl.col("iteration").is_in(iterations_to_select))
        .group_by("model")
        .agg(
            *_agg_stats,
            pl.col("optimizer").n_unique().alias("nb_optimizers"),
        )
        .with_columns(_dps_delta)
        .sort("model")
    )

    summaries_by_optimizer = (
        summaries_pretty.filter(pl.col("iteration").is_in(iterations_to_select), pl.col("model").is_in(models_to_select))
        .group_by("optimizer")
        .agg(
            *_agg_stats,
            pl.col("model").n_unique().alias("nb_models"),
        )
        .with_columns(_dps_delta)
        .sort("optimizer")
    )
    return (
        summaries_by_model,
        summaries_by_optimizer,
        summaries_by_optimizer_and_iteration,
        summaries_by_optimizer_and_model,
    )


@app.cell
def _():
    mo.md(r"""### RQ1: How  well  do  various  methods  optimize  code  when evaluated under a common benchmark?""")
    return


@app.cell
def _(models_file_label, models_label, summaries_by_optimizer_and_iteration):
    HEIGHT = 200
    WIDTH_RATIO = 2

    _correctness_of_optimizers_by_iteration_base = (
        alt.Chart(summaries_by_optimizer_and_iteration)
        .mark_line()
        .encode(
            alt.X("iteration:N"), alt.StrokeDash("optimizer"), alt.Color("optimizer").legend(orient="bottom", columns=2)
        )
        .properties(width=HEIGHT * WIDTH_RATIO, height=HEIGHT)
        .interactive()
    )

    _correctness_of_optimizers_by_iteration = _correctness_of_optimizers_by_iteration_base.encode(
        alt.Y("dps_norm", title="DPSnorm").scale(domain=(70, 95))
    ) & _correctness_of_optimizers_by_iteration_base.encode(alt.Y("pass@1_global", title="Pass@1").scale(domain=(40, 90)))

    _correctness_of_optimizers_by_iteration.save(
        f"figures/fig_correctness_of_optimizers_by_iteration_{models_file_label}.pdf"
    )
    _correctness_of_optimizers_by_iteration = _correctness_of_optimizers_by_iteration.properties(
        title=[
            "Pass@1 and DPSNorm across iterations depending on",
            "the method.",
            f"Using {models_label}",
        ]
    ).configure_title(fontSize=14)
    mo.ui.altair_chart(_correctness_of_optimizers_by_iteration)
    return HEIGHT, WIDTH_RATIO


@app.cell
def _(
    consider_only_last_iteration,
    iteration_file_label,
    iteration_label,
    iterations_to_select,
    summaries_by_optimizer_and_model,
):
    if len(iterations_to_select) == 1:
        _performance_delta_by_model_chart = (
            alt.Chart(summaries_by_optimizer_and_model)
            .mark_bar()
            .encode(
                alt.X("model").axis(labelAngle=0),
                alt.Y("dps_norm_delta", title="DPSNormΔ"),
                alt.Color("optimizer").legend(orient="top-right", columns=2),
                alt.XOffset("optimizer"),
            )
        )
        _performance_delta_by_model_chart.save(f"figures/fig_performance_delta_by_model_{iteration_file_label}.pdf")
        _performance_delta_by_model_chart = _performance_delta_by_model_chart.properties(
            title=[
                "DPSNormΔ of each method and model combination, at the last iteration. Negative results show decreases in DPSNorm compared to the baseline.",
                iteration_label,
            ]
        )
        res = mo.ui.altair_chart(_performance_delta_by_model_chart)
    else:
        res = mo.md(
            f"""{mo.callout('''Can only display the figure if a single iteration is selected''', kind="warn")}\n\n{consider_only_last_iteration}"""
        )
    res
    return


@app.cell
def _(iteration_file_label, iteration_label, summaries_by_optimizer_and_model):
    _correctness_by_model_chart = (
        alt.Chart(summaries_by_optimizer_and_model)
        .mark_bar()
        .encode(
            alt.X("model").axis(labelAngle=0),
            alt.Y("pass@1_global", title="Pass@1"),
            alt.Color("optimizer").legend(orient="top-right", columns=2),
            alt.XOffset("optimizer"),
        )
    )
    _correctness_by_model_chart.save(f"figures/fig_correctness_by_model_{iteration_file_label}.pdf")
    _correctness_by_model_chart = _correctness_by_model_chart.properties(
        title=["Correctness of the optimization depending on the model and the optimizer", iteration_label]
    )

    mo.ui.altair_chart(_correctness_by_model_chart)
    return


@app.cell
def _(task_summaries):
    _nb_tasks_with_no_noticable_change = len(
        task_summaries.filter((pl.col("dps_norm_improvement") - 1).abs() < 0.01, ~pl.col("is_baseline"))
    )
    _nb_tasks_with_no_noticable_improvement = len(
        task_summaries.filter((pl.col("dps_norm_improvement") - 1) < 0.01, ~pl.col("is_baseline"))
    )

    _nb_tasks_no_baseline = len(task_summaries.filter(~pl.col("is_baseline")))
    mo.md(f"""
    Interestingly enough, we find that ${_nb_tasks_with_no_noticable_change / _nb_tasks_no_baseline * 100:.1f}\\%$ of the optimizations (at the task level, not the sample level) don't change the DPS from the baseline, or change it from less than 1% and that ${_nb_tasks_with_no_noticable_improvement / _nb_tasks_no_baseline * 100:.1f}\\%$ don't improve the DPS at all within a 1\\% margin. This means that most of the optimizations have really little effect on the performance of the code being optimized.""")
    return


@app.cell
def _(task_summaries):
    nb_samples_passed = len(task_summaries.select("results").explode("results").unnest("results").filter("passed"))
    mo.md(f"""There is a total of {nb_samples_passed:,} samples that passed the tests and were profiled.""")
    return


@app.cell
def _():
    mo.md(r"""### RQ2: What is the energy necessary to produce one optimized result?""")
    return


@app.cell
def _(
    energy_unit,
    iteration_file_label,
    iteration_label,
    measured_energy_label,
    measured_energy_type,
    summaries_by_optimizer_and_model,
):
    _energy_by_model_chart = (
        alt.Chart(
            summaries_by_optimizer_and_model,
        )
        .mark_bar()
        .encode(
            alt.X("model").axis(labelAngle=0),
            alt.Y(measured_energy_type, title=f"{measured_energy_label} ({energy_unit})"),
            alt.Color("optimizer").legend(orient="top-right", columns=2),
            alt.XOffset("optimizer"),
        )
    )

    _energy_by_model_chart.save(f"figures/fig_energy_by_model_{iteration_file_label}.pdf")

    mo.ui.altair_chart(
        _energy_by_model_chart.properties(title=[f"{measured_energy_label} depending on the optimizer.", iteration_label])
    )
    return


@app.cell
def _(
    HEIGHT,
    WIDTH_RATIO,
    energy_unit,
    measured_energy_label,
    measured_energy_type,
    models_file_label,
    models_label,
    summaries_by_optimizer_and_iteration,
):
    _energy_per_result_per_optimizer_over_iterations_chart = (
        alt.Chart(summaries_by_optimizer_and_iteration)
        .mark_line()
        .encode(
            alt.X("iteration:N"),
            alt.StrokeDash("optimizer"),
            alt.Color("optimizer").legend(orient="bottom", columns=2),
            alt.Y(measured_energy_type, title=f"{measured_energy_label} ({energy_unit})"),
        )
        .properties(width=HEIGHT * WIDTH_RATIO, height=HEIGHT)
    )

    _energy_per_result_per_optimizer_over_iterations_chart.save(
        f"figures/fig_energy_per_result_over_iterations_{models_file_label}.pdf"
    )

    mo.ui.altair_chart(
        _energy_per_result_per_optimizer_over_iterations_chart.properties(
            title=[f"Energy usage across iterations", f"With {models_label}"]
        )
    )
    return


@app.cell
def _(
    energy_unit,
    measured_energy_label,
    measured_energy_type,
    models_label,
    summaries_by_optimizer_and_iteration,
):
    energy_per_optimizer_over_iterations_pivot = summaries_by_optimizer_and_iteration.pivot(
        on="iteration",
        index=["optimizer"],
        values=[
            measured_energy_type,
        ],
    )

    _rename = {f"{i}": f"Iter-{i}" for i in range(0, 5)}

    energy_per_optimizer_over_iterations_table = (
        GT(energy_per_optimizer_over_iterations_pivot)
        .tab_spanner(label=f"{measured_energy_label} ({energy_unit})", columns=pl.selectors.exclude("optimizer"))
        .fmt_number(pl.selectors.float(), decimals=2)
        .cols_label(_rename)
    )
    print(energy_per_optimizer_over_iterations_table.as_latex().replace("\\toprule", f"\\toprule % {models_label}"))

    energy_per_optimizer_over_iterations_table = energy_per_optimizer_over_iterations_table.tab_header(
        title=f"{measured_energy_label} depending on the optimizer", subtitle=f"With {models_label}"
    )

    mo.md(f"""
    {mo.as_html(energy_per_optimizer_over_iterations_table)}""")
    return


@app.cell
def _(
    energy_by_phase,
    model_for_energy_proportion_study_dropdown,
    pretty_summaries,
):
    _model_to_analyze = model_for_energy_proportion_study_dropdown.value

    _energy_usage_by_phase_df = (
        pretty_summaries(energy_by_phase)
        .drop_nulls("duration")
        .filter(pl.col("phase") != "idle")
        .filter(pl.col("model").str.contains(_model_to_analyze))
        .with_columns((pl.col("total_energy") / pl.col("duration")).alias("powerdraw"))
        .group_by("phase")
        .agg(
            total_energy=pl.col("total_energy").sum().round(3),
            energy_var_coef=pl.col("total_energy").std().truediv(pl.col("total_energy").mean()).round(3),
            duration_var_coef=pl.col("duration").std().truediv(pl.col("duration").mean()).round(3),
            powerdraw_var_coef=pl.col("powerdraw").std().truediv(pl.col("powerdraw").mean()).round(3),
        )
    )
    _energy_usage_by_phase_df = (
        _energy_usage_by_phase_df.with_columns(
            proportion=pl.col("total_energy").truediv(_energy_usage_by_phase_df["total_energy"].sum()).round(4)
        )
        .with_columns(proportion_percent=(pl.col("proportion") * 100).round(2))
        .sort("proportion")
    )

    _energy_usage_by_phase_table = GT(_energy_usage_by_phase_df.select("phase", "proportion_percent"))

    # _base = alt.Chart(_energy_usage_by_phase_df).encode(
    #     alt.Theta("total_energy", type="quantitative"),
    #     alt.Color("phase:N", type="nominal"),
    # )
    # _pie = _base.mark_arc(tooltip=True, innerRadius=80)
    # _text = _base.mark_text(radius=180, size=20).encode(text="phase:N")

    # mo.md(f"""#### Energy usage of the different phases of the optimization process

    # In the figure below, we can observe that a significant proportion of energy is used by the generation phase (${round(_energy_usage_by_phase_df.filter(pl.col("phase") == "generation")["proportion"].first() * 100, 1)}\%$). This can be explained by the very high energy consumption of LLM inference.

    # {mo.as_html(_pie + _text)}
    # """)

    _profiling_energy_prop = _energy_usage_by_phase_df.filter(pl.col("phase") == "profiling")["proportion_percent"].first()
    _generation_energy_prop = _energy_usage_by_phase_df.filter(pl.col("phase") == "generation")[
        "proportion_percent"
    ].first()
    _correctness_energy_prop = _energy_usage_by_phase_df.filter(pl.col("phase") == "correctness")[
        "proportion_percent"
    ].first()

    mo.md(f"""
    #### Energy usage by phase

    Proportion of energy used by phases for model {model_for_energy_proportion_study_dropdown} and all optimizers:

    - Generation: ${_generation_energy_prop}\\%$
    - Correctness verification: ${_correctness_energy_prop}\\%$
    - Performance evaluation: ${_profiling_energy_prop}\\%$

    *Full table*
    {mo.ui.table(_energy_usage_by_phase_df, selection=None)}
    """)
    return


@app.cell
def _(energy_by_phase, pretty_summaries):
    def coefficient_of_variation(col):
        return pl.col(col).std().truediv(pl.col(col).mean())


    _generation_duration_df = (
        pretty_summaries(
            energy_by_phase.drop_nulls("duration").filter(
                pl.col("phase") == "generation",
                pl.col("iteration") == 0,
                ~pl.col("optimizer").is_in(["llm4effi", "eoh", "simple10"]),
            )
        )
        .group_by("model")
        .agg(
            min_duration=pl.col("duration").min().round(2),
            # duration_var_coef=coefficient_of_variation("duration").round(2),
        )
        .sort("min_duration")
    )

    mo.md(f"""
    #### Generation duration

    The table below presents the duration (in seconds) of the generation phase when generating samples for the baseline.

    {mo.ui.table(_generation_duration_df)}
    """)
    return


@app.cell
def _(summaries):
    def find_best_config(summaries, _col):
        best_profitability = summaries.filter(pl.col(_col) > 0)[_col].max()
        return summaries.filter(pl.col(_col) == best_profitability)


    # find_best_config(summaries, on="runs_necessary_to_justify_optimization")
    _col = "dps_norm"
    best_config_profitability = find_best_config(summaries, _col)
    best_model = best_config_profitability["model"].first()
    # best_config_profitability
    # best_model = "Qwen2.5-Coder-14B"
    return (best_model,)


@app.cell
def _():
    mo.md(r"""### RQ3: In terms of energy, at what point is it profitable to optimize a given code?""")
    return


@app.cell
def _(iteration_label, models_label, summaries_by_optimizer):
    _rename = {
        # "model": "Model",
        "optimizer": "Method",
        "prop_possible_profitability": "Efficient@1",  # > 50% of non-null BEP
        "runs_necessary_to_justify_optimization_median": "Median BEP",
        "runs_necessary_to_justify_optimization_average": "Mean BEP",
        "runs_necessary_to_justify_optimization_no_outliers": "Mean BEP w/o outliers",
        "average_energy_per_execution_speedup": "Energy Speedup",
        "average_energy_per_execution_reduction": "Energy Reduction",
        "average_energy_per_execution_speedup_no_outliers": "Energy Speedup w/o outliers",
        "average_energy_per_execution_reduction_no_outliers": "Energy Reduction w/o outliers",
    }

    bep_optimizer_table_full = GT(summaries_by_optimizer.select(_rename.keys())).cols_label(_rename)

    print(bep_optimizer_table_full.as_latex().replace("\\toprule", f"\\toprule % {models_label} {iteration_label}"))

    bep_optimizer_table_full = bep_optimizer_table_full.tab_header(
        title=f"All Energy profitability metrics by optimizer", subtitle=f"With {models_label} - {iteration_label}"
    )

    bep_optimizer_table_full
    return


@app.cell
def _():
    energy_speedup_col = "average_energy_per_execution_reduction_no_outliers"
    energy_speedup_label = "Energy reduction thanks to the optimization"
    return energy_speedup_col, energy_speedup_label


@app.cell
def _(
    energy_speedup_col,
    energy_speedup_label,
    energy_unit,
    iteration_label,
    measured_energy_label,
    summaries_by_optimizer_and_model,
):
    _energy_gain_by_model_chart = (
        alt.Chart(
            summaries_by_optimizer_and_model.with_columns(pl.col(energy_speedup_col)),
        )
        .mark_bar()
        .encode(
            alt.X("model").axis(labelAngle=0),
            alt.Y(energy_speedup_col, title=f"{measured_energy_label} ({energy_unit})"),
            alt.Color("optimizer").legend(orient="top-left"),
            alt.XOffset("optimizer"),
        )
    )

    # _energy_by_model_chart.save(f"figures/fig_energy_by_model_{iteration_file_label}.pdf")

    mo.ui.altair_chart(
        _energy_gain_by_model_chart.properties(
            title=[f"{energy_speedup_label} depending on the optimizer.", iteration_label]
        )
    )
    return


@app.cell
def _(
    energy_speedup_col,
    energy_speedup_label,
    energy_unit,
    models_label,
    summaries_by_optimizer_and_iteration,
):
    energy_speedup_per_optimizer_over_iterations_pivot = summaries_by_optimizer_and_iteration.with_columns(
        pl.col(energy_speedup_col).round(3)
    ).pivot(
        on="iteration",
        index=["optimizer"],
        values=[
            energy_speedup_col,
        ],
    )

    _rename = {f"{i}": f"Iter-{i}" for i in range(0, 5)}

    energy_speedup_per_optimizer_over_iterations_table = (
        GT(energy_speedup_per_optimizer_over_iterations_pivot)
        .tab_spanner(label=f"{energy_speedup_label} ({energy_unit})", columns=pl.selectors.exclude("optimizer"))
        .cols_label(_rename)
    )
    print(energy_speedup_per_optimizer_over_iterations_table.as_latex().replace("\\toprule", f"\\toprule % {models_label}"))

    energy_speedup_per_optimizer_over_iterations_table = energy_speedup_per_optimizer_over_iterations_table.tab_header(
        title=f"{energy_speedup_label} depending on optimizers across iterations", subtitle=f"With {models_label}"
    )

    mo.md(f"""
    {mo.as_html(energy_speedup_per_optimizer_over_iterations_table)}""")
    return


@app.cell
def _(models_label, summaries_by_optimizer_and_iteration):
    bep_per_optimizer_over_iterations_pivot = summaries_by_optimizer_and_iteration.with_columns(
        pl.col("runs_necessary_to_justify_optimization_no_outliers").round(3)
    ).pivot(
        on="iteration",
        index=["optimizer"],
        values=[
            "runs_necessary_to_justify_optimization_no_outliers",
        ],
    )

    _rename = {f"{i}": f"Iter-{i}" for i in range(0, 5)}

    bep_per_optimizer_over_iterations_table = (
        GT(bep_per_optimizer_over_iterations_pivot)
        .tab_spanner(label=f"BEP", columns=pl.selectors.exclude("optimizer")).fmt_number(pl.selectors.float(), decimals=0)
        .cols_label(_rename)
    )
    print(bep_per_optimizer_over_iterations_table.as_latex().replace("\\toprule", f"\\toprule % {models_label}"))

    bep_per_optimizer_over_iterations_table = bep_per_optimizer_over_iterations_table.tab_header(
        title=f"BEP depending on optimizers across iterations", subtitle=f"With {models_label}"
    )

    mo.md(f"""
    {mo.as_html(bep_per_optimizer_over_iterations_table)}""")
    return


@app.cell
def _():
    mo.md(r"""## Validity analysis""")
    return


@app.cell
def _(summaries_pretty):
    program_evaluation_idle_describe = summaries_pretty["idle_power"].describe()
    idle_coefficient_of_variation = summaries_pretty["idle_power"].std() / summaries_pretty["idle_power"].mean()
    mo.md(f"""
    When measuring the energy consumption of the samples, our idle measures had a coefficient of variation of {idle_coefficient_of_variation:.3f}
    """)
    return


@app.cell
def _(energy_debug_df):
    _corr_df = []
    for _key, _df in energy_debug_df.partition_by(["model", "optimizer", "iteration"], as_dict=True).items():
        model, optimizer, iteration = _key
        _corr = (
            _df.select("results", "original_result_file")
            .explode("results")
            .unnest("results")
            .unnest("energy_measures")
            .select("num_cpu_instructions", "total_energy")
            .to_pandas()
            .corr()["total_energy"]
            .iloc[0]
        )
        _corr_df.append({"model": model, "optimizer": optimizer, "iteration": iteration, "correlation": _corr})

    corr_df_energy_cpu = pl.DataFrame(_corr_df)
    return (corr_df_energy_cpu,)


@app.cell
def _(corr_df_energy_cpu):
    _table = corr_df_energy_cpu.group_by("optimizer").agg(
        pl.col("correlation").mean().round(3).alias("correlation_mean"),
        pl.col("correlation").std().round(3).alias("correlation_std"),
        pl.col("correlation").min().round(3).alias("correlation_min"),
        pl.col("correlation").max().round(3).alias("correlation_max"),
    )

    mo.md(f"""
    Correlation between the number of CPU instructions and the measured energy consumption of the samples, by optimizer. A low correlation could indicate too much bias and error in the energy measures. Interestingly, Simple10 has a low minimum, indicating that one iteration might have too much noise in the measures.

    {mo.as_html(_table)}
    """)
    return


@app.cell
def _(corr_df_energy_cpu):
    _table = corr_df_energy_cpu.group_by("model").agg(
        pl.col("correlation").mean().round(3).alias("correlation_mean"),
        pl.col("correlation").std().round(3).alias("correlation_std"),
        pl.col("correlation").min().round(3).alias("correlation_min"),
        pl.col("correlation").max().round(3).alias("correlation_max"),
    )
    mo.md(f"""
    Correlation between the number of CPU instructions and the measured energy consumption of the samples, by model. A low correlation could indicate too much bias and error in the energy measures. Interestingly, DeepSeek-R1-Distill-Qwen-14B has a low minimum, indicating that one iteration might have too much noise in the measures.

    {mo.as_html(_table)}
    """)
    return


@app.cell
def _(energy_debug_df):
    overall_corr = (
        energy_debug_df.select("results", "original_result_file")
        .explode("results")
        .unnest("results")
        .unnest("energy_measures")
        .select("num_cpu_instructions", "total_energy")
        .to_pandas()
        .corr()["total_energy"]
        .iloc[0]
    )
    mo.md(f"""Overall, the correlation between the number of CPU instruction and the measured energy consumption of the samples is {overall_corr:.3f}""")
    return


@app.cell
def _(summaries_pretty):
    _dps_delta = (pl.col("dps_norm") - pl.col("dps_norm_baseline")).round(3).alias("dps_norm_delta")

    _corr = summaries_pretty.with_columns(_dps_delta).select(
        "average_energy_per_execution_reduction_no_outliers", "dps_norm_delta"
    ).drop_nulls().to_pandas().corr()["dps_norm_delta"].iloc[0]

    mo.md(f"""The correlation between a configuration's optimization capacity ($DPS_{{norm}}\\Delta$) and its Energy Reduction is {_corr:.3f}, which is pretty weak in comparison of the correlation at the sample level between the CPU instructions and energy consumption.""") 
    return


@app.cell
def _():
    mo.md(
        r"""
    ### Validation of the DPS metric in regard to the performance of the code

    How much does CPU time correlates to CPU instructions? Is CPU instructions correlated to DPS?
    """
    )
    return


@app.cell
def _(samples_eval_results):
    def corr(df: pl.DataFrame):
        return pl.from_pandas(df.to_pandas().corr())

    def get_pivot_for_col(cols):
        expr = None
        for col in cols:
            if expr is None:
                expr = pl.when(pl.col(col) == 1.0).then(pl.lit(col))
            else:
                expr = expr.when(pl.col(col) == 1.0).then(pl.lit(col))
        return expr.alias("corr_pivot")


    corr_cols = ["dps", "dps_norm", "num_cpu_instructions", "cpu_time"]

    sample_df_by_task_id = (
        samples_eval_results.filter("profiled").select(pl.col(corr_cols), pl.col("task_id")).partition_by("task_id")
    )

    corrs_df = (
        pl.concat(
            [
                corr(sample_df_by_task_id[0].select(pl.col(corr_cols))).with_columns(
                    pl.lit(df["task_id"].first()).alias("task_id"),
                    get_pivot_for_col(corr_cols),
                )
                for df in sample_df_by_task_id
            ],
            how="vertical",
        )
        .group_by("corr_pivot")
        .agg(pl.col(corr_cols).mean())
    )
    sns.heatmap(corrs_df.to_pandas().set_index("corr_pivot"), annot=True)
    return (corrs_df,)


@app.cell(hide_code=True)
def _(corrs_df, samples_eval_results):
    def _keep_only_min_and_max_of_each_group(df, group, value_col):
        return df.filter(
            ((pl.col(value_col) == pl.col(value_col).max()) | (pl.col(value_col) == pl.col(value_col).min())).over(group)
        )


    _profiled_samples_min_max = _keep_only_min_and_max_of_each_group(
        samples_eval_results.filter("profiled"),
        ["task_id", "matching_cluster_idx"],
        "num_cpu_instructions",
    )

    cpu_instructions_by_cluster_chart = (
        alt.Chart(_profiled_samples_min_max)
        .mark_circle(size=60)
        .encode(
            alt.X("matching_cluster_idx:O"),
            alt.Y("num_cpu_instructions").scale(type="log"),
        )
        .properties(width=200, height=180)
        .facet(facet="task_id:N", columns=4)
        .properties(title="Number of cpu instructions by their cluster. Only min and max of each cluster are shown")
    )

    dps_cpu_corr = corrs_df.filter(pl.col("corr_pivot") == "dps_norm")["num_cpu_instructions"].first()

    mo.md(
        f"""
        #### Is the order between the number of cpu instructions and DPS consistent?

        That is if A has a higher DPS than B, then A _must_ have less cpu instructions than B

        As we can see in the figure below, it _looks_ like most of it is okay. 
        It is also validated by the correlation coefficient between dps and num_cpu_instructions of {round(dps_cpu_corr, 2)}

        {mo.accordion({"Figure: Number of cpu instructions by their cluster": mo.lazy(mo.as_html(cpu_instructions_by_cluster_chart), show_loading_indicator=True)})}

        """
    )
    return


@app.cell
def _(summaries_raw):
    # We should only compare temperature between identical configuration. For now, we'll only take optimizers that have been tested on all temperatures
    _summaries_full_temperature = (
        summaries_raw.group_by("optimizer")
        .agg(n_temperatures=pl.col("temperature").n_unique())
        .filter(pl.col("n_temperatures") >= 5)
        .select("optimizer")
        .join(summaries_raw, on="optimizer")
    )

    _temperature_effect_chart = (
        alt.Chart(_summaries_full_temperature)
        .mark_boxplot()
        .encode(
            alt.X("temperature:O").scale(domain=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            alt.Y(alt.repeat("column"), type="quantitative").scale(domain=(0, 100)),
        )
        .repeat(column=["dps", "dps_norm", "pass@1"])
    )

    mo.md(f"""### What is the effect of the temperature on the DPS and the pass@1?

    Presenting results coming from these optimizers {_summaries_full_temperature["optimizer"].unique().to_list()}

    {mo.as_html(_temperature_effect_chart)}
    """)
    return


@app.cell
def _(task_summaries):
    _task_energy_samples = task_summaries.select(
        pl.col("average_energy_per_execution", "nb_profiled", "nb_samples", "task_id", "energy_per_processed_sample")
    ).drop_nulls()

    _all_correlations = [
        df.select("average_energy_per_execution", "nb_samples").corr()["nb_samples"][0]
        for df in _task_energy_samples.partition_by("task_id")
    ]
    mean_correlation_energy_per_execution_number_of_samples = np.mean(_all_correlations)
    _all_correlations = [
        df.select("energy_per_processed_sample", "nb_samples").corr()["nb_samples"][0]
        for df in _task_energy_samples.partition_by("task_id")
    ]
    mean_correlation_energy_per_opti_number_of_samples = np.mean(_all_correlations)

    scatterplot_energy_nb_samples_correlation = mo.ui.altair_chart(
        alt.Chart(_task_energy_samples)
        .mark_point()
        .encode(alt.X(alt.repeat("row"), type="quantitative"), alt.Y(alt.repeat("column"), type="quantitative"))
        .properties(width=300, height=150)
        .repeat(column=["average_energy_per_execution", "energy_per_processed_sample"], row=["nb_profiled", "nb_samples"])
        .properties(title="Correlations between the energy measurements and the number of samples being processed")
    )
    scatterplot_energy_nb_samples_correlation

    mo.md(f"""
    ### Is our energy measuring setup affected by the number of samples currently being measured?
    Our energy profiling could be affected by the number of samples being measured. This could show by having a strong correlation between the energy measured and the number of samples. If that were the case, this would invalidate most of our results.

    We measured the correlation between the number of samples being processed and the energy per execution of the code, as well as with the energy of optimizing the code. These correlation are respectively of ${mean_correlation_energy_per_execution_number_of_samples:.2f}$ and ${mean_correlation_energy_per_opti_number_of_samples:.2f}$. As the correlations between these variables are sufficiently low, we deem our measuring setup sufficiently reliable.

    {scatterplot_energy_nb_samples_correlation}

    """)
    return


@app.cell
def _(analysis_options, task_summaries):
    nb_potential_baselines = len(task_summaries.filter(pl.col("is_potential_baseline") == True))
    nb_baselines = len(task_summaries.filter(pl.col("is_baseline") == True))
    nb_tasks_total_no_potential_baseline = len(task_summaries.filter(~pl.col("is_potential_baseline")))
    nb_tasks_with_no_speedup_because_no_baseline = len(
        task_summaries.filter(~pl.col("is_potential_baseline"), pl.col("cpu_time_baseline").is_null())
    )
    nb_tasks_with_no_speedup = len(
        task_summaries.filter(~pl.col("is_potential_baseline"), pl.col("cpu_time_speedup").is_null())
    )

    mo.md(f"""
    When calculating baselines. Out of the {nb_potential_baselines} tasks eligible for being the baseline, {nb_potential_baselines - nb_baselines} had not enough passed samples (Must be $\\geq${analysis_options["min_profiled"].value}) to be used as baselines. Because of this, {nb_tasks_with_no_speedup_because_no_baseline} out of {nb_tasks_total_no_potential_baseline} ({nb_tasks_with_no_speedup_because_no_baseline / nb_tasks_total_no_potential_baseline * 100:.1f}%) task results won't have any speedup calculated. In total, including tasks that have not enough data, {nb_tasks_with_no_speedup} of the tasks ({nb_tasks_with_no_speedup / nb_tasks_total_no_potential_baseline * 100:.1f}%) don't won't have any speedup metrics.
    """)
    return


@app.cell
def _(task_summaries):
    ## Distributing the optimization cost down to the task level.
    _task_summaries_with_energy = task_summaries
    _energy_error = (
        _task_summaries_with_energy.group_by("file")
        .agg(pl.col("^energy_exp.*$").sum(), pl.col("total_energy").first())
        .with_columns(
            energy_error=(pl.col("energy_expenditure_bottom_up") - pl.col("total_energy")).truediv(pl.col("total_energy"))
        )["energy_error"]
        .mean()
    )

    # This aggregation could be done better if instead of using "energy_per_processed_sample" we used the energy per processed sample per phase. However it would take longer to implement.

    mo.md(f"""
    When using a semi-bottom up approach to aggregate the energy at the task level, we obtain an energy that is not the same as the ground truth. It is on average {round(abs(_energy_error) * 100, 2)}% off from the ground truth, which we deem acceptable.
    """)
    return


@app.cell
def _(summaries, task_summaries):
    ### Assertions about the results


    def warn_if_trigger(trigger, warning):
        if trigger:
            return f"\n{mo.callout(warning, kind='warn')}\n"
        return ""


    _warnings = ""

    _total_tasks = task_summaries["task_id"].unique()
    _are_tasks_equally_present = (
        task_summaries.group_by("file").agg(task_id=pl.col("task_id").n_unique())["task_id"].n_unique() == 1
    )
    # _are_tasks_equally_present = task_summaries["task_id"].value_counts()["count"].n_unique() == 1
    _warnings += warn_if_trigger(
        not _are_tasks_equally_present,
        "Some tasks may be missing, please verify the integrity of the files in case some parts of the results are missing.",
    )

    _warnings += warn_if_trigger(
        summaries["iteration"].value_counts()["count"].n_unique() != 1,
        "Some global summaries may be missing, please verify the integrity of the files in case some parts of the results are missing.",
    )

    _size_difference = len(task_summaries) - len(task_summaries.unique(subset=["file", "task_id", "batch_id"]))
    _warnings += warn_if_trigger(
        _size_difference != 0, f"Some tasks are duplicated: We found {_size_difference} duplicated rows"
    )

    _all_iterations_present = summaries["iteration"].n_unique() == 5
    _warnings += warn_if_trigger(not _all_iterations_present, "Not all iterations are present on the summaries dataset")

    _all_iterations_present = task_summaries["iteration"].n_unique() == 5
    _warnings += warn_if_trigger(
        not _all_iterations_present, "Not all iterations are present on the task summaries dataset"
    )

    _task_summaries_pass_1_dps_coherence = task_summaries.filter((pl.col("pass_1_global") == 0) & (pl.col("dps_norm") > 0))
    _warnings += warn_if_trigger(
        len(_task_summaries_pass_1_dps_coherence) > 0, "There are tasks with a null pass_1 but a non zero dps"
    )

    _special_optimizers = ["llm4effi", "eoh"]
    iteration_0_results_expected = (
        len(
            summaries.filter(iteration=0)
            .with_columns(
                optimizer=pl.when(pl.col("optimizer").is_in(_special_optimizers)).then(pl.col("optimizer")).otherwise(None)
            )
            .group_by(["optimizer", "model"])
            .agg()
        )
        * 118
    )
    iteration_0_results_actual = len(task_summaries.filter(iteration=0))
    _warnings += warn_if_trigger(
        iteration_0_results_expected != iteration_0_results_actual,
        f"Expected to find {iteration_0_results_expected} tasks results at iteration 0, but found {iteration_0_results_actual}",
    )

    _dps_norm_improvement_inconsistent = summaries.filter(
        (pl.col("dps_norm_baseline") - pl.col("dps_norm")) > 0.1, pl.col("dps_norm_improvement") >= 1
    ).select("model", "optimizer", "iteration", "^dps.*$")
    _warnings += warn_if_trigger(
        len(_dps_norm_improvement_inconsistent) > 0,
        f"There are {len(_dps_norm_improvement_inconsistent)} summaries with a DPS  improvement higher than one while having a DPSNorm lower than their baseline",
    )

    warnings = None
    if _warnings != "":
        warnings = mo.md(_warnings)
    warnings
    return (warnings,)


@app.cell
def _(warnings):
    assert warnings == None, "See the warnings on the cell above"
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Utilities""")
    return


@app.cell
def _(energy_by_phase, samples_eval_results, show_if_debug_mode, summaries):
    # Unused
    def _get_energy_per_token_expr(col_list):
        for col in col_list:
            yield pl.col("total_energy").truediv(pl.col("completion_tokens"))


    _samples_generation_details_by_file = (
        samples_eval_results.group_by("file").agg(pl.col("^*tokens$").sum()).join(summaries, on="file")
    )

    energy_per_token = (
        energy_by_phase.filter(pl.col("phase") == "generation")
        .join(_samples_generation_details_by_file, on="file")
        .drop("^*right$")
        .with_columns(
            (
                pl.col("total_energy").truediv(pl.col(col)).alias(f"energy_per_{col}")
                for col in ["completion_tokens", "prompt_tokens", "total_tokens"]
            )
        )
        .select(
            pl.col(
                "energy_per_completion_tokens",
                "energy_per_prompt_tokens",
                "energy_per_total_tokens",
            )
        )
    )

    show_if_debug_mode(energy_per_token)
    return


@app.cell
def _(summaries):
    all_optimizers = summaries["optimizer"].unique()
    all_models = summaries["model"].unique()

    opti_model_combinations = list(itertools.product(all_models, all_optimizers))
    opti_model_combinations_str = [";".join(l) for l in opti_model_combinations]
    present_combinations = summaries.select(
        opti_model_str=pl.concat_str(pl.col("model"), pl.col("optimizer"), separator=";")
    )
    not_present_combinations = list(
        filter(lambda combi: combi not in present_combinations["opti_model_str"].unique(), opti_model_combinations_str)
    )
    not_present_combinations
    mo.md(f"""
    The following model-optimizer combinations have not been done yet: {mo.as_html(not_present_combinations)}
    """)
    return


if __name__ == "__main__":
    app.run()
