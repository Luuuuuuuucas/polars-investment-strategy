import polars as pl
from polars import col
from datetime import date
from typing import Literal


def get_backtest_end_date(
    backtest_start_date: date,
    backtest_period: int,
    backtest_period_unit: Literal["w", "mo", "q", "y"],
) -> date:
    return (
        pl.Series([backtest_start_date])
        .dt.offset_by(f"{backtest_period}{backtest_period_unit}")
        .item()
    )


def get_trading_calendar(cleaned_close_prices_dataset: pl.DataFrame) -> pl.DataFrame:
    return cleaned_close_prices_dataset.select("date").unique(maintain_order=True)


def create_date_mapping(
    trading_calendar: pl.DataFrame,
    rebalance_frequency: int,
    rebalance_freq_unit: Literal["w", "mo", "q", "y"],
    backtest_start_date: date,
    backtest_end_date: date,
    lookback_period_for_total_returns: int,
    lookback_period_unit: Literal["w", "mo", "q", "y"],
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
        maintain_order="left",
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

    target_lag_base_dates = signal_and_rebalance_dates.select(
        col("signal_date").dt.offset_by("-1mo").alias("target_date")
    )

    lag_base_dates = target_lag_base_dates.join_asof(
        trading_calendar, left_on="target_date", right_on="date", strategy="backward"
    ).select(col("date").alias("lag_base_date"))

    return pl.concat(
        [
            lookback_dates,
            lag_base_dates,
            signal_and_rebalance_dates,
        ],
        how="horizontal_extend",
    ).select(
        col("lookback_date"),
        col("lag_base_date"),
        col("signal_date"),
        col("rebalance_date"),
    )


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
        maintain_order="left",
    ).rename(
        {
            "date": "signal_date",
            "close": "signal_close",
        }
    )

    lag_base_table = cleaned_close_prices_dataset.join(
        date_mapping_df,
        left_on="date",
        right_on="lag_base_date",
        how="semi",
        maintain_order="left",
    ).rename({"date": "lag_base_date", "close": "lag_base_close"})

    lookback_table = cleaned_close_prices_dataset.join(
        date_mapping_df,
        left_on="date",
        right_on="lookback_date",
        how="semi",
        maintain_order="left",
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
            maintain_order="left",
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
            date_mapping_df, on="signal_date", how="left", maintain_order="left"
        )
        .join(
            lookback_table,
            on=["lookback_date", "ticker"],
            how="left",
            maintain_order="left",
        )
        .join(
            lag_base_table,
            on=["lag_base_date", "ticker"],
            how="left",
            maintain_order="left",
        )
        .join(
            rebalance_table,
            on=["rebalance_date", "ticker"],
            how="left",
            maintain_order="left",
        )
    )

    return matched_table


def calculate_past_returns(
    date_price_mapping_df: pl.DataFrame,
    lookback_period_for_total_returns: int,
    lookback_period_unit: Literal["w", "mo", "q", "y"],
) -> pl.DataFrame:
    return date_price_mapping_df.with_columns(
        (col("lag_base_close") / col("lookback_close") - 1).alias(
            f"momentum_{lookback_period_for_total_returns}{lookback_period_unit}"
        )
    )


def calculate_past_returns_std(
    cleaned_close_prices_dataset: pl.DataFrame,
    factor_reference_table: pl.DataFrame,
    date_mapping_df: pl.DataFrame,
    lookback_period_for_total_returns: int,
    lookback_period_unit: Literal["w", "mo", "q", "y"],
) -> pl.DataFrame:
    start_date = date_mapping_df.get_column("lookback_date").first()
    end_date = date_mapping_df.get_column("lag_base_date").last()

    vol = (
        cleaned_close_prices_dataset.with_columns(
            col("close").pct_change().over("ticker").alias("daily_return")
        )
        .filter(col("date").is_between(start_date, end_date))
        .join_where(
            date_mapping_df.select(
                "lookback_date",
                "lag_base_date",
                "signal_date",
            ),
            col("date") > col("lookback_date"),
            col("date") <= col("lag_base_date"),
        )
        .group_by(["ticker", "signal_date"], maintain_order=True)
        .agg(
            col("daily_return")
            .std()
            .alias(f"vol_{lookback_period_for_total_returns}{lookback_period_unit}")
        )
    )

    return factor_reference_table.join(
        vol,
        on=["ticker", "signal_date"],
        how="left",
        maintain_order="left",
    )


def get_risk_adjusted_return(
    factor_reference_table: pl.DataFrame,
    lookback_period_for_total_returns: int,
    lookback_period_unit: Literal["w", "mo", "q", "y"],
) -> pl.DataFrame:
    return factor_reference_table.with_columns(
        (
            col(f"momentum_{lookback_period_for_total_returns}{lookback_period_unit}")
            / col(f"vol_{lookback_period_for_total_returns}{lookback_period_unit}")
        ).alias(
            f"risk_adjusted momentum_{lookback_period_for_total_returns}{lookback_period_unit}"
        )
    )
