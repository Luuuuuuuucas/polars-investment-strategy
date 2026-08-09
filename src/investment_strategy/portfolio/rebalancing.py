import polars as pl
from polars import col
from datetime import date


def prepare_rebalance_allocation_df(
    weighted_portfolio_signal_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Prepare the rebalance allocation DataFrame used for portfolio calculations.

    Arguments:
        weighted_portfolio_signal_df: A Polars DataFrame containing the selected stocks with portfolio weights assigned.

    Returns:
        A Polars DataFrame containing columns: rebalance_date, ticker, rebalance_open, portfolio_weight.
    """
    return weighted_portfolio_signal_df.select(
        col("rebalance_date"),
        col("ticker"),
        col("rebalance_open"),
        col("portfolio_weight"),
    )


def calculate_position_sizing(
    rebalance_allocation_df: pl.DataFrame,
    current_rebalance_date: date,
    investable_value: int | float,
) -> pl.DataFrame:
    """
    Calculate integer share quantities for the current rebalance date using investable value.

    Arguments:
        rebalance_allocation_df: A Polars DataFrame containing columns:
            rebalance_date, ticker, rebalance_open, portfolio_weight.

        current_rebalance_date: The current rebalance date in the loop.

        investable_value: The amount of capital available for investment.
    
    Returns:
        A Polars DataFrame containing columns: rebalance_date, ticker, shares.

    Example:
        >>> rebalance_allocation_df = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 2, 1),
        ...             date(2024, 2, 1),
        ...             date(2024, 3, 1),
        ...         ],
        ...         "ticker": ["A", "B", "A"],
        ...         "rebalance_open": [50.0, 100.0, 60.0],
        ...         "portfolio_weight": [0.6, 0.4, 1.0],
        ...     }
        ... )

        >>> calculate_position_sizing(
        ...     rebalance_allocation_df=rebalance_allocation_df,
        ...     current_rebalance_date=date(2024, 2, 1),
        ...     investable_value=10000,
        ... )
        shape: (2, 3)
        ┌────────────────┬────────┬────────┐
        │ rebalance_date │ ticker │ shares │
        │ ---            │ ---    │ ---    │
        │ date           │ str    │ i64    │
        ╞════════════════╪════════╪════════╡
        │ 2024-02-01     │ A      │ 120    │
        │ 2024-02-01     │ B      │ 40     │
        └────────────────┴────────┴────────┘
    
    Note: This is a helper function used by run_rebalance_simulation.
    """
    return rebalance_allocation_df.filter(
        col("rebalance_date") == current_rebalance_date
    ).select(
        col("rebalance_date"),
        col("ticker"),
        (
            (investable_value * col("portfolio_weight") / col("rebalance_open"))
            .floor()
            .cast(pl.Int64)
            .alias("shares")
        ),
    )


def calculate_reference_position_value(
    factor_reference_table: pl.DataFrame,
    current_positions: pl.DataFrame,
    reference_date: date,
) -> float:
    """
    Value the current positions at the specified reference date.

    Arguments:
        factor_reference_table: A Polars DataFrame containing columns: lookback_date, lag_base_date, signal_date,
            rebalance_date, ticker, lookback_close, lag_base_close, signal_close, rebalance_open.

        current_positions: A Polars DataFrame containing the current holdings.
            At minimum, the columns, ticker and shares, must be present.

        reference_date: The specific date where the valuation occurs.

    Returns:
        A float representing the position value.

    Example:
        >>> current_positions = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 2),
        ...         ],
        ...         "ticker": ["A", "B"],
        ...         "shares": [120, 40],
        ...     }
        ... )

        >>> factor_reference_table = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 2, 1),
        ...             date(2024, 2, 1),
        ...         ],
        ...         "ticker": ["A", "B"],
        ...         "rebalance_open": [50.0, 100.0],
        ...     }
        ... )

        >>> calculate_reference_position_value(
        ...     factor_reference_table=factor_reference_table,
        ...     current_positions=current_positions,
        ...     reference_date=date(2024, 2, 1),
        ... )
        10000.0

    Note: This is a helper function used by run_rebalance_simulation. The positions are valued before and after each rebalance.
    """
    valued_positions = (
        current_positions.select(col("ticker"), col("shares"))
        .with_columns(pl.lit(reference_date).alias("rebalance_date"))
        .join(
            factor_reference_table.select(
                col("rebalance_date"),
                col("ticker"),
                col("rebalance_open"),
            ),
            on=["rebalance_date", "ticker"],
            how="left",
            maintain_order="left"
        )
    )

    return valued_positions.select((col("shares") * col("rebalance_open")).sum()).item()


def run_rebalance_simulation(
    factor_reference_table: pl.DataFrame,
    rebalance_allocation_df: pl.DataFrame,
    initial_capital: int | float,
    rebalance_dates: pl.Series,
) -> dict[str, pl.DataFrame]:
    """
    Simulate all the rebalance occurrences. Record portfolio value, cash residual,
        and number of shares of each asset within each rebalance date.

    Arguments:
        factor_reference_table: A Polars DataFrame containing columns: lookback_date, lag_base_date, signal_date,
            rebalance_date, ticker, lookback_close, lag_base_close, signal_close, rebalance_open.
        
        rebalance_allocation_df: A Polars DataFrame containing columns:
            rebalance_date, ticker, rebalance_open, portfolio_weight.

        initial_capital: The initial amount of capital to invest in the portfolio.

        rebalance_dates: The predefined rebalance dates.

    Example:
        >>> rebalance_dates = date_mapping_df.get_column("rebalance_date")

        >>> simulation_result = run_rebalance_simulation(
        ...     factor_reference_table=factor_reference_table,
        ...     rebalance_allocation_df=rebalance_allocation_df,
        ...     initial_capital=100_000,
        ...     rebalance_dates=rebalance_dates,
        ... )

        >>> rebalance_level_table = simulation_result["rebalance_level_table"]
        >>> position_level_table = simulation_result["position_level_table"]

        >>> rebalance_level_table.head()
        >>> position_level_table.head()
        
    Returns:
        A dictionary containing:
            "rebalance_level_table":
                A Polars DataFrame containing columns: rebalance_date, portfolio_value, cash_residual.

            "position_level_table":
                A Polars DataFrame containing columns: rebalance_date, ticker, shares.
    """
    current_cash_residual = float(initial_capital)
    current_positions: pl.DataFrame | None = None

    cash_residual_record = []
    portfolio_values_record = []
    positions_record = []

    for rebalance_date in rebalance_dates:
        if current_positions is None:
            positions_value_before_rebalance = 0.0
        else:
            positions_value_before_rebalance = calculate_reference_position_value(
                factor_reference_table,
                current_positions,
                rebalance_date,
            )

        total_value_before_rebalance = (
            current_cash_residual + positions_value_before_rebalance
        )

        investable_value = total_value_before_rebalance

        current_positions = calculate_position_sizing(
            rebalance_allocation_df,
            rebalance_date,
            investable_value,
        )

        positions_value_after_rebalance = calculate_reference_position_value(
            factor_reference_table,
            current_positions,
            rebalance_date,
        )

        current_cash_residual = investable_value - positions_value_after_rebalance

        portfolio_value_after_rebalance = (
            positions_value_after_rebalance + current_cash_residual
        )

        cash_residual_record.append(current_cash_residual)
        portfolio_values_record.append(portfolio_value_after_rebalance)

        positions_record.append(current_positions)

    rebalance_level_table = pl.DataFrame(
        {
            "rebalance_date": rebalance_dates,
            "portfolio_value": portfolio_values_record,
            "cash_residual": cash_residual_record,
        }
    )
    position_level_table = pl.concat(positions_record, how="vertical")

    return {
        "rebalance_level_table": rebalance_level_table,
        "position_level_table": position_level_table,
    }