import polars as pl
from polars import col
from typing import Literal


def rank_signal(
    signal_df: pl.DataFrame,
    signal_col: str,
    *,
    group_col: str = "signal_date",
    descending: bool = True,
    method: Literal[
        "average",
        "min",
        "max",
        "dense",
        "ordinal",
        "random",
    ] = "ordinal",
) -> pl.DataFrame:
    """
    Note: when descending = True, the larger number it is, the smaller number of rank it receives.
    For instance, the largest number gets a rank of 1.
    """
    return signal_df.with_columns(
        col(signal_col)
        .rank(method=method, descending=descending)
        .over(group_col)
        .alias(f"{signal_col} rank")
    )


def filter_top_ranked(
    ranked_signal_df: pl.DataFrame, rank_col: str, top_n: int
) -> pl.DataFrame:
    return ranked_signal_df.filter(col(rank_col) <= top_n)


def sort_rankings(filtered_signal_df: pl.DataFrame, rank_col: str) -> pl.DataFrame:
    """
    For visualization only, not needed for other purposes.
    """
    return filtered_signal_df.sort(["signal_date", rank_col])


