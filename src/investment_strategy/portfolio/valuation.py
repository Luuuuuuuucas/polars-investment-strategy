import polars as pl
from polars import col
from datetime import date


def get_backtest_period_close_prices(
    cleaned_close_prices_dataset: pl.DataFrame,
    backtest_start_date: date,
    backtest_end_date: date,
) -> pl.DataFrame:
    """
    Creates a filtered close prices dataset, which will be used in get_daily_position_value_table.
    """
    return cleaned_close_prices_dataset.filter(
        (col("date") >= backtest_start_date) & (col("date") <= backtest_end_date)
    )


def get_next_date_matched_rebalance_level_table(
    rebalance_level_table: pl.DataFrame,
) -> pl.DataFrame:
    """
    Creates a filtered close prices dataset, which will be used in get_daily_position_value_table and get_daily_portfolio_table.
    """
    return rebalance_level_table.with_columns(
        col("rebalance_date").shift(-1).alias("next_rebalance_date")
    )


def get_daily_position_value_table(
    backtest_period_close_prices: pl.DataFrame,
    next_date_matched_rebalance_level_table: pl.DataFrame,
    position_level_table: pl.DataFrame,
) -> pl.DataFrame:
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
) -> pl.DataFrame:
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
            col("portfolio_value").pct_change().alias("daily_return")
        )
    )