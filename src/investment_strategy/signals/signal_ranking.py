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
    Rank the signal based on the signal column.

    Arguments:
        signal_df: factor_reference_table with additional one or more signal columns.

        signal_col: The name of the signal column to be ranked.

        group_col: The column used to group observations before ranking.
            By default, stocks are ranked within each signal date.

        descending: Whether to rank in descending order. Defaults to True.

        method: The ranking method. Defaults to "ordinal".

    Example:
        >>> signal_df = pl.DataFrame(
        ...     {
        ...         "signal_date": [
        ...             date(2024, 1, 31),
        ...             date(2024, 1, 31),
        ...             date(2024, 1, 31),
        ...             date(2024, 2, 29),
        ...             date(2024, 2, 29),
        ...             date(2024, 2, 29),
        ...         ],
        ...         "ticker": ["A", "B", "C", "A", "B", "C"],
        ...         "momentum_6mo": [0.20, 0.10, 0.15, 0.05, 0.25, 0.10],
        ...     }
        ... )

        >>> rank_signal(
        ...     signal_df=signal_df,
        ...     signal_col="momentum_6mo",
        ... ).select(
        ...     "signal_date",
        ...     "ticker",
        ...     "momentum_6mo",
        ...     "momentum_6mo_rank",
        ... )
        shape: (6, 4)
        ┌─────────────┬────────┬──────────────┬───────────────────┐
        │ signal_date │ ticker │ momentum_6mo │ momentum_6mo_rank │
        │ ---         │ ---    │ ---          │ ---               │
        │ date        │ str    │ f64          │ u32               │
        ╞═════════════╪════════╪══════════════╪═══════════════════╡
        │ 2024-01-31  │ A      │ 0.20         │ 1                 │
        │ 2024-01-31  │ B      │ 0.10         │ 3                 │
        │ 2024-01-31  │ C      │ 0.15         │ 2                 │
        │ 2024-02-29  │ A      │ 0.05         │ 3                 │
        │ 2024-02-29  │ B      │ 0.25         │ 1                 │
        │ 2024-02-29  │ C      │ 0.10         │ 2                 │
        └─────────────┴────────┴──────────────┴───────────────────┘

    Note:
        When `descending=True`, larger signal values receive smaller rank numbers.
        For example, the largest value receives a rank of 1.
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
    """
    Keep stocks with rank values less than or equal to top_n for each signal date.

    Arguments:
        ranked_signal_df: signal DataFrame with an additional rank column.

        rank_col: The column name of the rank column.

        top_n: The number of the highest ranked stocks to be filtered.
    
    Returns:
        ranked_signal_df with only top_n stocks kept in each signal date.

    Example:
        >>> filtered = filter_top_ranked(
        ...     ranked_signal_df=ranked,
        ...     rank_col="momentum_6mo_rank",
        ...     top_n=2,
        ... )

        >>> filtered.select(
        ...     "signal_date",
        ...     "ticker",
        ...     "momentum_6mo",
        ...     "momentum_6mo_rank",
        ... )
        shape: (4, 4)
        ┌─────────────┬────────┬──────────────┬───────────────────┐
        │ signal_date │ ticker │ momentum_6mo │ momentum_6mo_rank │
        │ ---         │ ---    │ ---          │ ---               │
        │ date        │ str    │ f64          │ u32               │
        ╞═════════════╪════════╪══════════════╪═══════════════════╡
        │ 2024-01-31  │ A      │ 0.20         │ 1                 │
        │ 2024-01-31  │ C      │ 0.15         │ 2                 │
        │ 2024-02-29  │ B      │ 0.25         │ 1                 │
        │ 2024-02-29  │ C      │ 0.10         │ 2                 │
        └─────────────┴────────┴──────────────┴───────────────────┘
    
    """
    return ranked_signal_df.filter(col(rank_col) <= top_n)


def sort_rankings(filtered_signal_df: pl.DataFrame, rank_col: str) -> pl.DataFrame:
    """
    Sort the filtered signal DataFrame by signal date and rank.

    Arguments:
        filtered_signal_df: A ranked signal DataFrame containing only the selected stocks.

        rank_col: The name of the rank column used for sorting.

    Returns:
        The sorted filtered signal DataFrame.

    Example:
        >>> sorted_rankings = sort_rankings(
        ...     filtered_signal_df=filtered,
        ...     rank_col="momentum_6mo_rank",
        ... )

        >>> sorted_rankings.select(
        ...     "signal_date",
        ...     "ticker",
        ...     "momentum_6mo_rank",
        ... )
        shape: (4, 3)
        ┌─────────────┬────────┬───────────────────┐
        │ signal_date │ ticker │ momentum_6mo_rank │
        │ ---         │ ---    │ ---               │
        │ date        │ str    │ u32               │
        ╞═════════════╪════════╪═══════════════════╡
        │ 2024-01-31  │ A      │ 1                 │
        │ 2024-01-31  │ C      │ 2                 │
        │ 2024-02-29  │ B      │ 1                 │
        │ 2024-02-29  │ C      │ 2                 │
        └─────────────┴────────┴───────────────────┘

    Note:
        This function is intended for visualization purposes only.
    """
    return filtered_signal_df.sort(["signal_date", rank_col])