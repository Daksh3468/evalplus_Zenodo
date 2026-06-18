import functools

import polars as pl


@pl.api.register_expr_namespace("crinkles")
class Crinkles:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def struct_to_list(self) -> pl.Expr:
        return self._expr.map_batches(
            lambda x: (
                x.to_frame()  # Convert series into a one column dataframe
                .unnest(x.name)  # Unnest the dictionnary
                .select(pl.concat_list(pl.all()))  # Concat all the list columns into a single column
                .to_series()  # Turn it back into a series
            )
        )  # Courtesy of https://stackoverflow.com/questions/77456832/how-to-extract-a-struct-values-to-a-list-in-polars

    def explode_struct_into_list(self, key) -> pl.Expr:
        return self._expr.map_elements(functools.partial(explode_dict_into_list, key="phase"))


def explode_dict_into_list(d: dict, key) -> list[dict]:
    res = []
    for old_key in d:
        res.append({key: old_key} | d[old_key])
    return res


def remove_outliers(df, col, percentage):
    first_val = df[col].quantile(percentage)
    last_val = df[col].quantile(1 - percentage)
    return df.filter((pl.col(col) > first_val) & (pl.col(col) < last_val))


def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def load_library():  # Necessary for import to register the class
    pass
