import polars as pl
from polars import col
from datetime import date


def get_backtest_period_close_prices(
    cleaned_close_prices_dataset: pl.DataFrame,
    backtest_start_date: date,
    backtest_end_date: date,
) -> pl.DataFrame:
    """
    Filter the cleaned close price dataset to the backtest date range.
    It is used for get_daily_position_value_table.

    Arguments:
        cleaned_close_prices_dataset: A cleaned Polars DataFrame containing columns: date, ticker, close.

        backtest_start_date: The start date of the backtest.

        backtest_end_date: The end date of the backtest.

    Returns:
        cleaned_close_prices_dataset with dates in the backtest date range.

    Example:
        >>> cleaned_close_prices_dataset = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2023, 12, 29),
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 3),
        ...             date(2024, 1, 4),
        ...             date(2024, 1, 5),
        ...         ],
        ...         "ticker": ["A", "A", "A", "A", "A"],
        ...         "close": [98.0, 100.0, 101.0, 102.0, 103.0],
        ...     }
        ... )

        >>> get_backtest_period_close_prices(
        ...     cleaned_close_prices_dataset=cleaned_close_prices_dataset,
        ...     backtest_start_date=date(2024, 1, 2),
        ...     backtest_end_date=date(2024, 1, 4),
        ... )
        shape: (3, 3)
        ┌────────────┬────────┬───────┐
        │ date       │ ticker │ close │
        │ ---        │ ---    │ ---   │
        │ date       │ str    │ f64   │
        ╞════════════╪════════╪═══════╡
        │ 2024-01-02 │ A      │ 100.0 │
        │ 2024-01-03 │ A      │ 101.0 │
        │ 2024-01-04 │ A      │ 102.0 │
        └────────────┴────────┴───────┘
    """
    return cleaned_close_prices_dataset.filter(
        col("date").is_between(backtest_start_date, backtest_end_date)
    )


def get_next_date_matched_rebalance_level_table(
    rebalance_level_table: pl.DataFrame,
) -> pl.DataFrame:
    """
    Match the current rebalance date to the next rebalance date.
    It is used for get_daily_position_value_table and get_daily_portfolio_table.

    Arguments:
        rebalance_level_table: A Polars DataFrame containing columns: rebalance_date, pre_rebalance_portfolio_value,
            post_rebalance_portfolio_value, cash_residual, transaction_cost.

    Returns:
        rebalance_level_table with an additional column: next_rebalance_date.

    Example:
        >>> rebalance_level_table = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 2, 1),
        ...             date(2024, 3, 1),
        ...         ],
        ...         "pre_rebalance_portfolio_value": [
        ...             100000.0,
        ...             103500.0,
        ...             102000.0,
        ...         ],
        ...         "post_rebalance_portfolio_value": [
        ...             99850.0,
        ...             103300.0,
        ...             101820.0,
        ...         ],
        ...         "cash_residual": [200.0, 150.0, 180.0],
        ...         "transaction_cost": [150.0, 200.0, 180.0],
        ...     }
        ... )

        >>> get_next_date_matched_rebalance_level_table(
        ...     rebalance_level_table
        ... )
        shape: (3, 6)
        ┌────────────────┬───────────────────────────────┬────────────────────────────────┬───────────────┬──────────────────┬─────────────────────┐
        │ rebalance_date │ pre_rebalance_portfolio_value │ post_rebalance_portfolio_value │ cash_residual │ transaction_cost │ next_rebalance_date │
        │ ---            │ ---                           │ ---                            │ ---           │ ---              │ ---                 │
        │ date           │ f64                           │ f64                            │ f64           │ f64              │ date                │
        ╞════════════════╪═══════════════════════════════╪════════════════════════════════╪═══════════════╪══════════════════╪═════════════════════╡
        │ 2024-01-02     │ 100000.0                      │ 99850.0                        │ 200.0         │ 150.0            │ 2024-02-01          │
        │ 2024-02-01     │ 103500.0                      │ 103300.0                       │ 150.0         │ 200.0            │ 2024-03-01          │
        │ 2024-03-01     │ 102000.0                      │ 101820.0                       │ 180.0         │ 180.0            │ null                │
        └────────────────┴───────────────────────────────┴────────────────────────────────┴───────────────┴──────────────────┴─────────────────────┘
    """
    return rebalance_level_table.with_columns(
        col("rebalance_date").shift(-1).alias("next_rebalance_date")
    )


def get_daily_position_value_table(
    backtest_period_close_prices: pl.DataFrame,
    next_date_matched_rebalance_level_table: pl.DataFrame,
    position_level_table: pl.DataFrame,
) -> pl.DataFrame:
    """
    Create a daily position-value table with reference to rebalance level table and position level table.

    Arguments:
        backtest_period_close_prices: A Polars DataFrame with dates in the backtest date range, containing columns:
            date, ticker, close.
        
        next_date_matched_rebalance_level_table: A Polars DataFrame containing columns: rebalance_date, pre_rebalance_portfolio_value,
            post_rebalance_portfolio_value, cash_residual, next_rebalance_date.

        position_level_table: A Polars DataFrame containing columns: rebalance_date, ticker, shares.

    Returns:
        A Polars DataFrame containing columns: date, rebalance_date, ticker, shares, close, position_value.

    Example:
        >>> daily_position_value_table = get_daily_position_value_table(
        ...     backtest_period_close_prices=backtest_period_close_prices,
        ...     next_date_matched_rebalance_level_table=next_date_matched_rebalance_level_table,
        ...     position_level_table=position_level_table,
        ... )

        >>> daily_position_value_table.head()
    """
    position_level_table = position_level_table.rename({"ticker": "portfolio_ticker"})

    date_matched_position_level_table = position_level_table.join(
        next_date_matched_rebalance_level_table.select(
            col("rebalance_date"), col("next_rebalance_date")
        ),
        on="rebalance_date",
        how="left",
        maintain_order="left"
    )

    return (
        backtest_period_close_prices.join_where(
            date_matched_position_level_table,
            col("rebalance_date") <= col("date"),
            (
                (col("date") < col("next_rebalance_date"))
                | (col("next_rebalance_date").is_null())
            ),
            col("ticker") == col("portfolio_ticker"),
        )
        .select(
            col("date"),
            col("rebalance_date"),
            col("ticker"),
            col("shares"),
            col("close"),
        )
        .sort(["rebalance_date", "date", "ticker"])
        .with_columns((col("shares") * col("close")).alias("position_value"))
    )


def get_daily_portfolio_table(
    next_date_matched_rebalance_level_table: pl.DataFrame,
    daily_position_value_table: pl.DataFrame,
    initial_capital: float
) -> pl.DataFrame:
    """
    Create a daily portfolio table with reference to rebalance level table and daily position-value table.

    Arguments:
        next_date_matched_rebalance_level_table: A Polars DataFrame containing columns: rebalance_date, pre_rebalance_portfolio_value,
            post_rebalance_portfolio_value, cash_residual, next_rebalance_date.
        
        daily_position_value_table: A Polars DataFrame containing columns: date, rebalance_date, ticker,
            shares, close, position_value.

        initial_capital:
            The initial amount of capital to invest in the portfolio.
    Returns:
        A Polars DataFrame containing columns: date, positions_value, cash_residual, portfolio_value, daily_return.

    Example:
        >>> daily_portfolio_table = get_daily_portfolio_table(
        ...     next_date_matched_rebalance_level_table=next_date_matched_rebalance_level_table,
        ...     daily_position_value_table=daily_position_value_table,
        ...     initial_capital=initial_capital
        ... )

        >>> daily_portfolio_table.head()

    Note:
        The first daily return is calculated from the initial capital at the first rebalance open to the portfolio value
        at the end of the first trading day. Subsequent daily returns are calculated using the percentage change in
        portfolio value.
    """
    sorted_daily_position_values = (
        daily_position_value_table.select(
            col("date"), col("position_value").alias("positions_value")
        )
        .group_by("date", maintain_order=True)
        .sum()
    )

    return (
        sorted_daily_position_values.join_where(
            next_date_matched_rebalance_level_table,
            col("rebalance_date") <= col("date"),
            (
                (col("date") < col("next_rebalance_date"))
                | col("next_rebalance_date").is_null()
            ),
        )
        .select(
            col("date"),
            col("positions_value"),
            col("cash_residual"),
        )
        .sort("date")
        .with_columns(
            (col("positions_value") + col("cash_residual")).alias("portfolio_value")
        )
        .with_columns(
            col("portfolio_value").pct_change().fill_null(col("portfolio_value") / initial_capital - 1).alias("daily_return")
        )
    )