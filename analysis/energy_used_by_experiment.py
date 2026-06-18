

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import polars as pl
    import os


@app.cell
def _():
    ENERGY_LOGS_PATH = "./analysis/energy_logs"


    def find_all_gpu_files(root):
        return list(filter(lambda s: s.endswith("gpu.csv"), os.listdir(root)))


    def find_all_cpu_files(root):
        return list(filter(lambda s: s.endswith("cpu.csv"), os.listdir(root)))
    return ENERGY_LOGS_PATH, find_all_cpu_files, find_all_gpu_files


@app.cell
def _(ENERGY_LOGS_PATH, find_all_cpu_files, find_all_gpu_files):
    gpu_files = find_all_gpu_files(ENERGY_LOGS_PATH)
    cpu_files = find_all_cpu_files(ENERGY_LOGS_PATH)
    files_enum = pl.Enum(gpu_files + cpu_files)
    energy_type = pl.Enum(["cpu", "gpu"])


    from io import TextIOBase


    class CommaSpaceStripper(TextIOBase):
        def __init__(self, file_obj):
            self.file_obj = file_obj

        def readline(self):
            line = self.file_obj.readline()
            if not line:
                return line
            # Replace ", " with "," while preserving newline
            return line.replace(", ", ",")

        def read(self, size=-1):
            # If size is -1 or not provided, read until EOF
            data = self.file_obj.read(size)
            if not data:
                return data
            return data.replace(", ", ",")

        def close(self):
            self.file_obj.close()


    def load_gpu_df(gpu_file, ENERGY_LOGS_PATH):
        gpu_path = os.path.join(ENERGY_LOGS_PATH, gpu_file)
        with open(gpu_path, "r") as f:
            gpu_df = (
                (
                    pl.scan_csv(CommaSpaceStripper(f))
                    .rename({"power.draw [W]": "powerdraw", "index": "gpu_index"})
                    .with_columns(pl.col("timestamp").str.to_datetime("%Y/%m/%d %H:%M:%S%.f", strict=False))
                    .drop_nulls("timestamp")
                    .group_by("gpu_index")
                    .agg(
                        pl.col("timestamp").min().alias("start"),
                        pl.col("timestamp").max().alias("end"),
                        pl.col("powerdraw").mean(),
                    )
                    .with_columns(
                        duration=(pl.col("end") - pl.col("start")).dt.total_seconds().round(2),
                        file=pl.lit(gpu_file).cast(files_enum),
                    )
                    .with_columns(total_consumption=(pl.col("powerdraw") * pl.col("duration")).round(2))
                )
                .group_by("file")
                .agg(pl.col("start", "duration").first(), pl.col("total_consumption").sum())
            )
        return gpu_df


    def load_all_gpu_data(ENERGY_LOGS_PATH):
        _all_gpu_dfs = []
        for gpu_file in gpu_files:
            gpu_df = load_gpu_df(gpu_file, ENERGY_LOGS_PATH)
            _all_gpu_dfs.append(gpu_df)
        return (
            pl.concat(_all_gpu_dfs)
            .with_columns(pl.col("start").dt.date(), pl.lit("gpu", dtype=energy_type).alias("type")).collect()
        )


    def load_cpu_df(cpu_file, ENERGY_LOGS_PATH):
        cpu_path = os.path.join(ENERGY_LOGS_PATH, cpu_file)
        with open(cpu_path, "r") as f:
            cpu_df = (
                pl.scan_csv(
                    f,
                    separator=";",
                    has_header=False,
                    comment_prefix="#",
                    skip_rows=2,
                    schema=pl.Schema({"time_since_start": pl.Float64, "energy_consumed_in_interval": pl.Float64}),
                    truncate_ragged_lines=True,
                )
                .with_columns(pl.lit(cpu_file).cast(files_enum).alias("file"))
                .group_by("file")
                .agg(
                    pl.col("energy_consumed_in_interval").sum().alias("total_consumption").round(2),
                    pl.col("time_since_start").max().round(2).alias("duration"),
                )
            )
        return cpu_df


    def load_all_cpu_data(ENERGY_LOGS_PATH):
        _all_cpu_dfs = []
        for cpu_file in cpu_files:
            cpu_df = load_cpu_df(cpu_file, ENERGY_LOGS_PATH)
            _all_cpu_dfs.append(cpu_df)

        return pl.concat(_all_cpu_dfs).with_columns(
            pl.col("file").cast(pl.String).str.split(by="_").list.first().str.to_date().alias("start"),
            pl.lit("cpu", dtype=energy_type).alias("type"),
        ).collect()
    return load_all_cpu_data, load_all_gpu_data


@app.cell
def _(ENERGY_LOGS_PATH, load_all_cpu_data, load_all_gpu_data):
    power_df = pl.concat([load_all_gpu_data(ENERGY_LOGS_PATH), load_all_cpu_data(ENERGY_LOGS_PATH)], how="diagonal_relaxed").with_columns(
        total_consumption_wh = (pl.col("total_consumption") / 3600).round(2)
    )
    # power_by_day_df = power_df.group_by("start").agg(
    #     pl.col("^total_consumption.*$").sum(),
    #     pl.col("duration")
    # )
    # power_by_day_df
    return (power_df,)


@app.cell
def _(power_df):
    power_df
    return


@app.cell
def _(power_df):
    import altair as alt
    from okabe_ito_theme import okabe_ito_theme
    alt.themes.register("okabe-ito", okabe_ito_theme)
    alt.themes.enable("okabe-ito")

    mo.ui.altair_chart(alt.Chart(power_df).mark_bar(width=20).encode(
        alt.X("start"),
        alt.Y("total_consumption_wh")
    ).properties(title="Consumption per day"))
    return


@app.cell
def _(power_df):
    total_consumption_kwh = power_df["total_consumption_wh"].sum() / 10**3
    co2_by_kwh_france = 56  # grams - https://ourworldindata.org/grapher/carbon-intensity-electricity?tab=chart&country=~FRA
    total_carbon = total_consumption_kwh * co2_by_kwh_france / 10**3

    mo.md(f"""
    In total, this experiment consummed {total_consumption_kwh:.2f} KWh. Which amounted to {total_carbon:.2f} kgs of CO2eq
    """)
    return


if __name__ == "__main__":
    app.run()
