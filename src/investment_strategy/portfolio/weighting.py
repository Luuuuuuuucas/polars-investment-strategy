import polars as pl
from polars import col
from typing import Literal


def calculate_equal_weight(
    filtered_signal_df: pl.DataFrame,
    *,
    group_col: str = "rebalance_date",
) -> pl.DataFrame:
    """
    A helper function for construct_portfolio_weights.
    """
    return filtered_signal_df.with_columns(
        (1 / pl.len().over(group_col)).alias("portfolio_weight")
    )


def calculate_signal_weight(
    filtered_signal_df: pl.DataFrame,
    signal_col: str,
    *,
    group_col: str = "rebalance_date",
) -> pl.DataFrame:
    """
    A helper function for construct_portfolio_weights.
    """
    return filtered_signal_df.with_columns(
        (col(signal_col) / col(signal_col).sum().over(group_col)).alias(
            "portfolio_weight"
        )
    )


def construct_portfolio_weights(
    filtered_signal_df: pl.DataFrame,
    weighting_method: Literal[
        "equal_weighted",
        "signal_weighted",
    ],
    *,
    signal_col: str | None = None,
    group_col: str = "rebalance_date",
) -> pl.DataFrame:
    if weighting_method == "equal_weighted":
        return calculate_equal_weight(
            filtered_signal_df,
            group_col=group_col,
        )

    if weighting_method == "signal_weighted":
        if signal_col is None:
            raise ValueError(
                "signal_col must be provided when weighting_method='signal_weighted'."
            )

        return calculate_signal_weight(
            filtered_signal_df,
            signal_col,
            group_col=group_col,
        )

    raise ValueError(f"Unsupported weighting_method: {weighting_method}")