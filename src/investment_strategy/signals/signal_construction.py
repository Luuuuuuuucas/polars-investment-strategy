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
) -> pl.DataFrame:
    """
    Define rebalance dates and signal dates.
    Create a date-mapping Polars DataFrame mapping signal dates and rebalance dates.

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

    Returns:
        A Polars DataFrame containing columns: signal_date, rebalance_date.

    Example:
        >>> trading_calendar = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2025, 12, 31),
        ...             date(2026, 1, 2),
        ...             date(2026, 1, 5),
        ...             date(2026, 1, 30),
        ...             date(2026, 2, 2),
        ...             date(2026, 2, 27),
        ...             date(2026, 3, 2),
        ...         ]
        ...     }
        ... )

        >>> date_mapping = create_date_mapping(
        ...     trading_calendar=trading_calendar,
        ...     rebalance_frequency=1,
        ...     rebalance_freq_unit="mo",
        ...     backtest_start_date=date(2026, 1, 2),
        ...     backtest_end_date=date(2026, 3, 2),
        ... )

        >>> date_mapping
        shape: (3, 2)
        ┌─────────────┬────────────────┐
        │ signal_date ┆ rebalance_date │
        │ ---         ┆ ---            │
        │ date        ┆ date           │
        ╞═════════════╪════════════════╡
        │ 2025-12-31  ┆ 2026-01-02     │
        │ 2026-01-30  ┆ 2026-02-02     │
        │ 2026-02-27  ┆ 2026-03-02     │
        └─────────────┴────────────────┘

    Note:
    - Theoretical rebalance dates are aligned to the next available trading date.
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

    return rebalance_dates.join(
        trading_calendar.with_columns(
            col("date").shift(1).alias("signal_date")
        ),
        left_on="rebalance_date",
        right_on="date",
        how="left",
        maintain_order="left",
    ).select(
        col("signal_date"),
        col("rebalance_date")
    )


def get_prices_for_date_mapping(
    cleaned_close_prices_dataset: pl.DataFrame,
    cleaned_stock_OHLCV: pl.DataFrame,
    date_mapping_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Get close prices for signal date. Get open prices for each rebalance date.

    Arguments:
        cleaned_close_prices_dataset: A cleaned Polars DataFrame containing columns: date, ticker, close.

        cleaned_stock_OHLCV: A Polars DataFrame with daily stock OHLCV data missing-value handled using predefined rules.

        date_mapping_df: A date-mapping Polars DataFrame containing columns: signal_date, rebalance_date.

    Returns:
        A Polars DataFrame containing columns: signal_date, rebalance_date, ticker, signal_close, rebalance_open.

    Example:
        >>> cleaned_close_prices_dataset = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2025, 12, 31),
        ...             date(2025, 12, 31),
        ...             date(2026, 1, 30),
        ...             date(2026, 1, 30),
        ...             date(2026, 2, 27),
        ...             date(2026, 2, 27),
        ...         ],
        ...         "ticker": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
        ...         "close": [250.0, 420.0, 255.0, 430.0, 260.0, 440.0],
        ...     }
        ... )

        >>> cleaned_stock_OHLCV = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2026, 1, 2),
        ...             date(2026, 1, 2),
        ...             date(2026, 2, 2),
        ...             date(2026, 2, 2),
        ...             date(2026, 3, 2),
        ...             date(2026, 3, 2),
        ...         ],
        ...         "ticker": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
        ...         "open": [251.0, 422.0, 256.0, 432.0, 262.0, 442.0],
        ...     }
        ... )

        >>> date_mapping_df = pl.DataFrame(
        ...     {
        ...         "signal_date": [
        ...             date(2025, 12, 31),
        ...             date(2026, 1, 30),
        ...             date(2026, 2, 27),
        ...         ],
        ...         "rebalance_date": [
        ...             date(2026, 1, 2),
        ...             date(2026, 2, 2),
        ...             date(2026, 3, 2),
        ...         ],
        ...     }
        ... )

        >>> prices_for_date_mapping = get_prices_for_date_mapping(
        ...     cleaned_close_prices_dataset=cleaned_close_prices_dataset,
        ...     cleaned_stock_OHLCV=cleaned_stock_OHLCV,
        ...     date_mapping_df=date_mapping_df,
        ... )

        >>> prices_for_date_mapping
        shape: (6, 5)
        ┌─────────────┬────────┬──────────────┬────────────────┬────────────────┐
        │ signal_date ┆ ticker ┆ signal_close ┆ rebalance_date ┆ rebalance_open │
        │ ---         ┆ ---    ┆ ---          ┆ ---            ┆ ---            │
        │ date        ┆ str    ┆ f64          ┆ date           ┆ f64            │
        ╞═════════════╪════════╪══════════════╪════════════════╪════════════════╡
        │ 2025-12-31  ┆ AAPL   ┆ 250.0        ┆ 2026-01-02     ┆ 251.0          │
        │ 2025-12-31  ┆ MSFT   ┆ 420.0        ┆ 2026-01-02     ┆ 422.0          │
        │ 2026-01-30  ┆ AAPL   ┆ 255.0        ┆ 2026-02-02     ┆ 256.0          │
        │ 2026-01-30  ┆ MSFT   ┆ 430.0        ┆ 2026-02-02     ┆ 432.0          │
        │ 2026-02-27  ┆ AAPL   ┆ 260.0        ┆ 2026-03-02     ┆ 262.0          │
        │ 2026-02-27  ┆ MSFT   ┆ 440.0        ┆ 2026-03-02     ┆ 442.0          │
        └─────────────┴────────┴──────────────┴────────────────┴────────────────┘
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

    return signal_table.join(
        date_mapping_df, on="signal_date", how="left", maintain_order="left"
    ).join(
        rebalance_table,
        on=["rebalance_date", "ticker"],
        how="left",
        maintain_order="left",
    )