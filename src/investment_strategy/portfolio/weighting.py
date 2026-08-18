import polars as pl
from polars import col
from typing import Literal


def calculate_equal_weight(
    portfolio_basket: pl.DataFrame,
    *,
    group_col: str = "rebalance_date",
) -> pl.DataFrame:
    """
    Calculate equal portfolio weights within each group.

    Arguments:
        portfolio_basket: A ranked signal DataFrame containing only the selected stocks.

        group_col: The column used to define portfolio groups for weight calculation. Defaults to "rebalance_date",
            so equal weights are calculated independently for each rebalance date.

    Returns:
        portfolio_basket with an additional column: portfolio_weight.

    Example:
        >>> portfolio_basket = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 2, 1),
        ...             date(2024, 2, 1),
        ...             date(2024, 3, 1),
        ...             date(2024, 3, 1),
        ...         ],
        ...         "ticker": ["A", "C", "B", "C"],
        ...     }
        ... )

        >>> calculate_equal_weight(portfolio_basket)
        shape: (4, 3)
        ┌────────────────┬────────┬──────────────────┐
        │ rebalance_date │ ticker │ portfolio_weight │
        │ ---            │ ---    │ ---              │
        │ date           │ str    │ f64              │
        ╞════════════════╪════════╪══════════════════╡
        │ 2024-02-01     │ A      │ 0.5              │
        │ 2024-02-01     │ C      │ 0.5              │
        │ 2024-03-01     │ B      │ 0.5              │
        │ 2024-03-01     │ C      │ 0.5              │
        └────────────────┴────────┴──────────────────┘

    Note: This is a helper function used by construct_portfolio_weights.
    """
    return portfolio_basket.with_columns(
        (1 / pl.len().over(group_col)).alias("portfolio_weight")
    )


def calculate_signal_weight(
    portfolio_basket: pl.DataFrame,
    signal_col: str,
    *,
    group_col: str = "rebalance_date",
) -> pl.DataFrame:
    """
    Calculate signal-based portfolio weights within each group.

    Arguments:
        portfolio_basket: A ranked signal DataFrame containing only the selected stocks.

        signal_col: The name of the signal column for which will be to calculate the weights.

        group_col: The column used to define portfolio groups for weight calculation. Defaults to "rebalance_date",
            so signal-based weights are calculated independently for each rebalance date.

    Returns:
        portfolio_basket with an additional column: portfolio_weight.

    Example:
        >>> portfolio_basket = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 2, 1),
        ...             date(2024, 2, 1),
        ...             date(2024, 3, 1),
        ...             date(2024, 3, 1),
        ...         ],
        ...         "ticker": ["A", "B", "A", "B"],
        ...         "momentum_6mo": [0.30, 0.10, 0.20, 0.20],
        ...     }
        ... )

        >>> calculate_signal_weight(
        ...     portfolio_basket=portfolio_basket,
        ...     signal_col="momentum_6mo",
        ... )
        shape: (4, 4)
        ┌────────────────┬────────┬──────────────┬──────────────────┐
        │ rebalance_date │ ticker │ momentum_6mo │ portfolio_weight │
        │ ---            │ ---    │ ---          │ ---              │
        │ date           │ str    │ f64          │ f64              │
        ╞════════════════╪════════╪══════════════╪══════════════════╡
        │ 2024-02-01     │ A      │ 0.3          │ 0.75             │
        │ 2024-02-01     │ B      │ 0.1          │ 0.25             │
        │ 2024-03-01     │ A      │ 0.2          │ 0.5              │
        │ 2024-03-01     │ B      │ 0.2          │ 0.5              │
        └────────────────┴────────┴──────────────┴──────────────────┘

    Note: This is a helper function used by construct_portfolio_weights.
    """
    return portfolio_basket.with_columns(
        (col(signal_col) / col(signal_col).sum().over(group_col)).alias(
            "portfolio_weight"
        )
    )


def construct_portfolio_weights(
    portfolio_basket: pl.DataFrame,
    weighting_method: Literal[
        "equal_weighted",
        "signal_weighted",
    ],
    *,
    signal_col: str | None = None,
    group_col: str = "rebalance_date",
) -> pl.DataFrame:
    """
    Construct portfolio weights within each group.

    Arguments:
        portfolio_basket: A ranked signal DataFrame containing only the selected stocks.

        weighting_method: The method used for constructing portfolio weights.

        signal_col: The name of the signal column used to calculate signal-based portfolio weights.
            Required when weighting_method="signal_weighted".
            Ignored when weighting_method="equal_weighted".

        group_col: The column used to define portfolio groups for weight calculation.
            Defaults to "rebalance_date", so portfolio weights are calculated
            independently for each rebalance date.

    Example:
        >>> equal_weighted = construct_portfolio_weights(
        ...     portfolio_basket=portfolio_basket,
        ...     weighting_method="equal_weighted",
        ... )

        >>> signal_weighted = construct_portfolio_weights(
        ...     portfolio_basket=portfolio_basket,
        ...     weighting_method="signal_weighted",
        ...     signal_col="momentum_6mo",
        ... )
        
        >>> "portfolio_weight" in equal_weighted.columns
        True
        >>> "portfolio_weight" in signal_weighted.columns
        True

    Returns:
        portfolio_basket with an additional column: portfolio_weight.
    """
    if weighting_method == "equal_weighted":
        return calculate_equal_weight(
            portfolio_basket,
            group_col=group_col,
        )

    if weighting_method == "signal_weighted":
        if signal_col is None:
            raise ValueError(
                "signal_col must be provided when weighting_method='signal_weighted'."
            )

        return calculate_signal_weight(
            portfolio_basket,
            signal_col,
            group_col=group_col,
        )

    raise ValueError(f"Unsupported weighting_method: {weighting_method}")
