import polars as pl
from polars import col
from datetime import date
from typing import Literal


def get_backtest_end_date(
    backtest_start_date: date,
    backtest_period: int,
    backtest_period_unit: Literal["d", "w", "mo", "q", "y"],
) -> date:
    return (
        pl.Series([backtest_start_date])
        .dt.offset_by(f"{backtest_period}{backtest_period_unit}")
        .item()
    )


def get_trading_calendar(cleaned_close_prices_dataset: pl.DataFrame) -> pl.DataFrame:
    return cleaned_close_prices_dataset.select("date").unique().sort("date")


def create_date_mapping(
    trading_calendar: pl.DataFrame,
    rebalance_frequency: int,
    rebalance_freq_unit: Literal["d", "w", "mo", "q", "y"],
    backtest_start_date: date,
    backtest_end_date: date,
    lookback_period_for_total_returns: int,
    lookback_period_unit: Literal["d", "w", "mo", "q", "y"],
) -> pl.DataFrame:
    """
    Theoretical rebalance dates are aligned to the next available trading date.
    Theoretical lookback dates are aligned to the most recent trading date on or before the target date,
    so the realized lookback period is never shorter than requested.

    The first rebalance date represents the initial portfolio formation.
    Subsequent rebalance dates represent periodic portfolio rebalancing.
    """
    target_rebalance_dates = (
        pl.date_range(
            start=backtest_start_date,
            end=backtest_end_date,
            interval=f"{rebalance_frequency}{rebalance_freq_unit}",
            eager=True,
        )
        .alias("target_date")
        .to_frame()
    )

    rebalance_dates = target_rebalance_dates.join_asof(
        trading_calendar,
        left_on="target_date",
        right_on="date",
        strategy="forward",
    ).select(col("date").alias("rebalance_date"))

    signal_and_rebalance_dates = rebalance_dates.join(
        trading_calendar.sort("date").with_columns(
            col("date").shift(1).alias("signal_date")
        ),
        left_on="rebalance_date",
        right_on="date",
        how="left",
    )

    target_lookback_dates = signal_and_rebalance_dates.select(
        col("signal_date")
        .dt.offset_by(f"-{lookback_period_for_total_returns}{lookback_period_unit}")
        .alias("target_date")
    )

    lookback_dates = target_lookback_dates.join_asof(
        trading_calendar,
        left_on="target_date",
        right_on="date",
        strategy="backward",
    ).select(col("date").alias("lookback_date"))

    return pl.concat(
        [
            lookback_dates,
            signal_and_rebalance_dates,
        ],
        how="horizontal_extend",
    ).select(col("lookback_date"), col("signal_date"), col("rebalance_date"))


def get_prices_for_date_mapping(
    cleaned_close_prices_dataset: pl.DataFrame,
    cleaned_dataset: pl.DataFrame,
    date_mapping_df: pl.DataFrame,
) -> pl.DataFrame:
    signal_table = cleaned_close_prices_dataset.join(
        date_mapping_df,
        left_on="date",
        right_on="signal_date",
        how="semi",
    ).rename(
        {
            "date": "signal_date",
            "close": "signal_close",
        }
    )

    lookback_table = cleaned_close_prices_dataset.join(
        date_mapping_df,
        left_on="date",
        right_on="lookback_date",
        how="semi",
    ).rename(
        {
            "date": "lookback_date",
            "close": "lookback_close",
        }
    )

    rebalance_table = (
        cleaned_dataset.select(
            "date",
            "ticker",
            "open",
        )
        .join(
            date_mapping_df,
            left_on="date",
            right_on="rebalance_date",
            how="semi",
        )
        .rename(
            {
                "date": "rebalance_date",
                "open": "rebalance_open",
            }
        )
    )

    matched_table = (
        signal_table.join(
            date_mapping_df,
            on="signal_date",
            how="left",
        )
        .join(
            lookback_table,
            on=["lookback_date", "ticker"],
            how="left",
        )
        .join(
            rebalance_table,
            on=["rebalance_date", "ticker"],
            how="left",
        )
    )

    return matched_table


def calculate_past_returns(
    date_price_mapping_df: pl.DataFrame,
    lookback_period_for_total_returns: int,
    lookback_period_unit: Literal["d", "w", "mo", "q", "y"]
) -> pl.DataFrame:
    return date_price_mapping_df.with_columns(
        (col("signal_close") / col("lookback_close") - 1).alias(
            f"past {lookback_period_for_total_returns}{lookback_period_unit} total returns"
        )
    )


def calculate_past_returns_std(
    cleaned_close_prices_dataset: pl.DataFrame,
    trading_calendar: pl.DataFrame,
    date_mapping_df: pl.DataFrame,
    rolling_window_days: int,
    backtest_end_date: date,
) -> pl.DataFrame:
    first_signal_date = date_mapping_df.get_column("signal_date").min()

    window_start_date = (
        trading_calendar.filter(col("date") <= first_signal_date)
        .tail(rolling_window_days + 1)
        .get_column("date")
        .first()
    )

    filtered_dataset = cleaned_close_prices_dataset.filter(
        (col("date") >= window_start_date) & (col("date") <= backtest_end_date)
    ).sort(["ticker", "date"])

    dataset_with_returns = filtered_dataset.with_columns(
        col("close").shift(1).over("ticker").alias("last_day_close")
    ).with_columns(
        (col("close") / col("last_day_close") - 1).alias("daily_return")
    )

    dataset_with_std = dataset_with_returns.with_columns(
        (
            col("daily_return").rolling_std(rolling_window_days).over("ticker")
            * (252**0.5)
        ).alias(f"past {rolling_window_days} trading days std")
    )

    return dataset_with_std.join(
        date_mapping_df,
        left_on="date",
        right_on="signal_date",
        how="semi",
    ).sort(["date", "ticker"])


def get_full_date_price_std_table(
    dataset_with_past_returns: pl.DataFrame, dataset_with_past_std: pl.DataFrame
) -> pl.DataFrame:
    return (
        dataset_with_past_std.join(
            dataset_with_past_returns,
            left_on=["date", "ticker"],
            right_on=["signal_date", "ticker"],
            how="left",
        )
        .drop(["close", "last_day_close", "daily_return"])
        .rename({"date": "signal_date"})
    )


def get_risk_adjusted_return(
    full_date_price_std_table: pl.DataFrame,
    lookback_period_for_total_returns: int,
    lookback_period_unit: Literal["d", "w", "mo", "q", "y"],
    rolling_window_days: int
) -> pl.DataFrame:
    return full_date_price_std_table.with_columns(
        (col(f"past {lookback_period_for_total_returns}{lookback_period_unit} total returns") / col(f"past {rolling_window_days} trading days std"))
        .alias(f"risk_adjusted past {lookback_period_for_total_returns}{lookback_period_unit} total return")
    )

