import polars as pl
from polars import col
from datetime import date
from typing import Literal


def get_backtest_end_date(
    backtest_start_date: date,
    backtest_period: int,
    backtest_period_unit: Literal["w", "mo", "q", "y"],
) -> date:
    """
    Calculate the backtest end date from the start date and backtest period.

    Arguments:
        backtest_start_date: The start date of the backtest.

        backtest_period: The length of the backtest period.

        backtest_period_unit: The unit of the backtest horizon:
            "w": weeks,
            "mo": months,
            "q": quarters,
            "y": years

    Returns:
        The ending date of the backtest horizon.

    Example:
        >>> get_backtest_end_date(
        ...     backtest_start_date=date(2020, 1, 1),
        ...     backtest_period=5,
        ...     backtest_period_unit="y",
        ... )
        datetime.date(2025, 1, 1)
    """
    return (
        pl.Series([backtest_start_date])
        .dt.offset_by(f"{backtest_period}{backtest_period_unit}")
        .item()
    )


def get_trading_calendar(cleaned_close_prices_dataset: pl.DataFrame) -> pl.DataFrame:
    """
    Get all available unique trading dates in the dataset.

    Arguments:
        cleaned_close_prices_dataset: A cleaned Polars DataFrame containing columns: date, ticker, close.

    Returns:
        A Polars DataFrame containing one row per unique trading date.

    Example:
        >>> prices = pl.DataFrame({
        ...     "date": [
        ...         date(2026, 1, 2),
        ...         date(2026, 1, 2),
        ...         date(2026, 1, 5),
        ...     ],
        ...     "ticker": ["AAPL", "MSFT", "AAPL"],
        ...     "close": [250.0, 480.0, 252.0],
        ... })
        >>> get_trading_calendar(prices)
        shape: (2, 1)
        ┌────────────┐
        │ date       │
        │ ---        │
        │ date       │
        ╞════════════╡
        │ 2026-01-02 │
        │ 2026-01-05 │
        └────────────┘
    """
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
    Define rebalance dates, signal dates, lag base dates, and lookback dates.
    Create a date-mapping Polars DataFrame mapping lookback dates, lag base dates, and signal dates
    to the corresponding rebalance dates.

    Arguments:
        trading_calendar: A Polars DataFrame containing one row per unique trading date.

        rebalance_frequency: The frequency of rebalancing in the backtest period.

        rebalance_freq_unit: The unit of the rebalancing frequency.
            "w": weeks,
            "mo": months,
            "q": quarters,
            "y": years

        backtest_start_date: The start date of the backtest.

        backtest_end_date: The end date of the backtest.

        lookback_period_for_total_returns: The length of the lookback period.

        lookback_period_unit: The unit of the lookback period.

    Returns:
        A Polars DataFrame containing columns: lookback_date, lag_base_date, signal_date, rebalance_date.

    Example:
        >>> trading_calendar = pl.DataFrame(
        ...     {
        ...         "date": pl.date_range(
        ...             date(2020, 1, 1),
        ...             date(2025, 12, 31),
        ...             interval="1d",
        ...             eager=True,
        ...         )
        ...     }
        ... )
        >>> date_mapping = create_date_mapping(
        ...     trading_calendar=trading_calendar,
        ...     rebalance_frequency=1,
        ...     rebalance_freq_unit="mo",
        ...     backtest_start_date=date(2020, 1, 1),
        ...     backtest_end_date=date(2025, 1, 1),
        ...     lookback_period_for_total_returns=6,
        ...     lookback_period_unit="mo",
        ... )

    Note:
    - Theoretical rebalance dates are aligned to the next available trading date.
    - Theoretical lookback dates are aligned to the most recent trading date on or before the target date,
      so the realized lookback period is never shorter than requested.
    - Theoretical lag base dates are aligned to the most recent trading date on or before the target date,
      so the realized lookback period is never shorter than requested.
    - The first rebalance date represents the initial portfolio formation.
    - Subsequent rebalance dates represent periodic portfolio rebalancing.
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
    cleaned_stock_OHLCV: pl.DataFrame,
    date_mapping_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Get close prices for each of lookback dates, lag base dates, and signal dates. Get open prices for each rebalance date.

    Arguments:
        cleaned_close_prices_dataset: A cleaned Polars DataFrame containing columns: date, ticker, close.

        cleaned_stock_OHLCV: A Polars DataFrame with daily stock OHLCV data missing-value handled using predefined rules.

        date_mapping_df: A date-mapping Polars DataFrame containing columns: lookback_date, lag_base_date, signal_date, rebalance_date.

    Returns:
        A Polars DataFrame containing columns: lookback_date, lag_base_date, signal_date, rebalance_date, ticker, lookback_close,
        lag_base_close, signal_close, rebalance_open.

    Example:
        >>> prices_for_mapping = get_prices_for_date_mapping(
        ...     cleaned_close_prices_dataset=cleaned_close_prices_dataset,
        ...     cleaned_stock_OHLCV=cleaned_stock_OHLCV,
        ...     date_mapping_df=date_mapping_df,
        ... )
    """
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
        cleaned_stock_OHLCV.select(
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

    return (
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


def calculate_momentum(
    factor_reference_table: pl.DataFrame,
    lookback_period_for_total_returns: int,
    lookback_period_unit: Literal["w", "mo", "q", "y"],
) -> pl.DataFrame:
    """
    Calculate the percentage change of the lag base close relative to the lookback close.

    Arguments:
        factor_reference_table: A Polars DataFrame containing columns: lookback_date, lag_base_date, signal_date,
            rebalance_date, ticker, lookback_close, lag_base_close, signal_close, rebalance_open.

        lookback_period_for_total_returns: The length of the lookback period.

        lookback_period_unit: The unit of the lookback period.
            "w": weeks,
            "mo": months,
            "q": quarters,
            "y": years

    Returns:
        factor_reference_table with an additional column: momentum_{lookback_period_for_total_returns}{lookback_period_unit}

    Example:
        >>> factor_reference_table = calculate_momentum(
        ...     factor_reference_table=factor_reference_table,
        ...     lookback_period_for_total_returns=6,
        ...     lookback_period_unit="mo",
        ... )

        >>> "momentum_6mo" in factor_reference_table.columns
        True
    """
    return factor_reference_table.with_columns(
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
    """
    Calculate return volatility over the lookback date and the lag base date.
    
    Arguments:
        cleaned_close_prices_dataset: A cleaned Polars DataFrame containing columns: date, ticker, close.

        factor_reference_table: A Polars DataFrame containing columns: lookback_date, lag_base_date, signal_date,
            rebalance_date, ticker, lookback_close, lag_base_close, signal_close, rebalance_open.

        date_mapping_df: A date-mapping Polars DataFrame containing columns: lookback_date, lag_base_date, signal_date, rebalance_date.

        lookback_period_for_total_returns: The length of the lookback period.

        lookback_period_unit: The unit of the lookback period.

    Returns:
        factor_reference_table with an additional column: vol_{lookback_period_for_total_returns}{lookback_period_unit}

    Example:
    >>> factor_reference_table = calculate_past_returns_std(
    ...     cleaned_close_prices_dataset=cleaned_close_prices_dataset,
    ...     factor_reference_table=factor_reference_table,
    ...     date_mapping_df=date_mapping_df,
    ...     lookback_period_for_total_returns=6,
    ...     lookback_period_unit="mo",
    ... )

    >>> "vol_6mo" in factor_reference_table.columns
    True

    Note:
    - This function uses simple returns rather than log returns.
    - This function is designed to support sequential factor construction.
      Existing columns in `factor_reference_table` are preserved, and the volatility factor is appended as a new column.

    """
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
        .sort(["signal_date"])
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
    factor_reference_with_momentum_and_vol: pl.DataFrame,
    lookback_period_for_total_returns: int,
    lookback_period_unit: Literal["w", "mo", "q", "y"],
) -> pl.DataFrame:
    """
    Calculate risk-adjusted momentum.

    Arguments:
        factor_reference_with_momentum_and_vol: A Polars DataFrame containing columns: lookback_date, lag_base_date,
            signal_date, rebalance_date, ticker, lookback_close, lag_base_close, signal_close, rebalance_open,
            momentum_{lookback_period_for_total_returns}{lookback_period_unit}, and
            vol_{lookback_period_for_total_returns}{lookback_period_unit}
        
        lookback_period_for_total_returns: The length of the lookback period.

        lookback_period_unit: The unit of the lookback period.

    Returns:
        The input DataFrame with an additional column:
            risk_adjusted_momentum_{lookback_period_for_total_returns}{lookback_period_unit}.

    Example:
        >>> factor_reference_table = get_risk_adjusted_return(
        ...     factor_reference_with_momentum_and_vol=factor_reference_table,
        ...     lookback_period_for_total_returns=6,
        ...     lookback_period_unit="mo",
        ... )

        >>> "risk_adjusted_momentum_6mo" in factor_reference_table.columns
        True
            
    Note:
        This function is designed to support sequential factor construction.
        Existing columns are preserved, and the risk-adjusted momentum factor is appended as a new column.
    """
    return factor_reference_with_momentum_and_vol.with_columns(
        (
            col(f"momentum_{lookback_period_for_total_returns}{lookback_period_unit}")
            / col(f"vol_{lookback_period_for_total_returns}{lookback_period_unit}")
        ).alias(
            f"risk_adjusted_momentum_{lookback_period_for_total_returns}{lookback_period_unit}"
        )
    )
