import polars as pl
from polars import col
from typing import Literal


def create_momentum_date_mapping(
    date_mapping_df: pl.DataFrame,
    trading_calendar: pl.DataFrame,
    lookback_period_for_total_returns: int,
    lookback_period_unit: Literal["w", "mo", "q", "y"],
) -> pl.DataFrame:
    """
    Define lag base dates and lookback dates.
    Create a date-mapping Polars DataFrame mapping lookback dates, lag base dates, and signal dates
    to the corresponding rebalance dates.

    Arguments:
        date_mapping_df: A Polars DataFrame containing columns: signal_date, rebalance_date.

        trading_calendar: A Polars DataFrame containing one row per unique trading date.

        lookback_period_for_total_returns: The length of the lookback period.

        lookback_period_unit: The unit of the lookback period.

    Returns:
        A Polars DataFrame containing columns: lookback_date, lag_base_date, signal_date, rebalance_date.

    Example:
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

        >>> trading_calendar = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2025, 6, 30),
        ...             date(2025, 7, 30),
        ...             date(2025, 8, 27),
        ...             date(2025, 11, 28),
        ...             date(2025, 12, 30),
        ...             date(2026, 1, 27),
        ...             date(2025, 12, 31),
        ...             date(2026, 1, 30),
        ...             date(2026, 2, 27),
        ...         ]
        ...     }
        ... ).sort("date")

        >>> momentum_date_mapping = create_momentum_date_mapping(
        ...     date_mapping_df=date_mapping_df,
        ...     trading_calendar=trading_calendar,
        ...     lookback_period_for_total_returns=6,
        ...     lookback_period_unit="mo",
        ... )

        >>> momentum_date_mapping
        shape: (3, 4)
        ┌───────────────┬───────────────┬─────────────┬────────────────┐
        │ lookback_date ┆ lag_base_date ┆ signal_date ┆ rebalance_date │
        │ ---           ┆ ---           ┆ ---         ┆ ---            │
        │ date          ┆ date          ┆ date        ┆ date           │
        ╞═══════════════╪═══════════════╪═════════════╪════════════════╡
        │ 2025-06-30    ┆ 2025-11-28    ┆ 2025-12-31  ┆ 2026-01-02     │
        │ 2025-07-30    ┆ 2025-12-30    ┆ 2026-01-30  ┆ 2026-02-02     │
        │ 2025-08-27    ┆ 2026-01-27    ┆ 2026-02-27  ┆ 2026-03-02     │
        └───────────────┴───────────────┴─────────────┴────────────────┘

    Note:
    - Theoretical rebalance dates are aligned to the next available trading date.
    - Theoretical lookback dates are aligned to the most recent trading date on or before the target date,
      so the realized lookback period is never shorter than requested.
    - Theoretical lag base dates are aligned to the most recent trading date on or before the target date,
      so the realized lookback period is never shorter than requested.
    - The first rebalance date represents the initial portfolio formation.
    - Subsequent rebalance dates represent periodic portfolio rebalancing.
    - This is a helper function for calculate_momentum and calculate_risk_adjusted_momentum.
    """
    target_lookback_dates = date_mapping_df.select(
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

    target_lag_base_dates = date_mapping_df.select(
        col("signal_date").dt.offset_by("-1mo").alias("target_date")
    )

    lag_base_dates = target_lag_base_dates.join_asof(
        trading_calendar, left_on="target_date", right_on="date", strategy="backward"
    ).select(col("date").alias("lag_base_date"))

    return pl.concat(
        [
            lookback_dates,
            lag_base_dates,
            date_mapping_df,
        ],
        how="horizontal_extend",
    ).select(
        col("lookback_date"),
        col("lag_base_date"),
        col("signal_date"),
        col("rebalance_date"),
    )


def get_prices_for_momentum_date_mapping(
    factor_reference_table: pl.DataFrame,
    momentum_date_mapping_df: pl.DataFrame,
    cleaned_close_prices_dataset: pl.DataFrame,
) -> pl.DataFrame:
    """
    Add lag base dates and signal dates with close prices in additional to factor_reference_table.

    Arguments:
        factor_reference_table: A Polars DataFrame containing columns: signal_date, rebalance_date, ticker,
            signal_close, rebalance_open.

        momentum_date_mapping_df: A Polars DataFrame containing columns: lookback_date, lag_base_date,
            signal_date, rebalance_date.

        cleaned_close_prices_dataset: A cleaned Polars DataFrame containing columns: date, ticker, close.

    Returns:
        A Polars DataFrame containing columns: lookback_date, lag_base_date, signal_date, rebalance_date, ticker,
            lookback_close, lag_base_close, signal_close, rebalance_open.

    Example:
        >>> factor_reference_table = pl.DataFrame(
        ...     {
        ...         "signal_date": [
        ...             date(2025, 12, 31),
        ...             date(2025, 12, 31),
        ...             date(2026, 1, 30),
        ...             date(2026, 1, 30),
        ...         ],
        ...         "rebalance_date": [
        ...             date(2026, 1, 2),
        ...             date(2026, 1, 2),
        ...             date(2026, 2, 2),
        ...             date(2026, 2, 2),
        ...         ],
        ...         "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
        ...         "signal_close": [250.0, 420.0, 255.0, 430.0],
        ...         "rebalance_open": [251.0, 422.0, 256.0, 432.0],
        ...     }
        ... )

        >>> momentum_date_mapping_df = pl.DataFrame(
        ...     {
        ...         "lookback_date": [
        ...             date(2025, 6, 30),
        ...             date(2025, 7, 30),
        ...         ],
        ...         "lag_base_date": [
        ...             date(2025, 11, 28),
        ...             date(2025, 12, 30),
        ...         ],
        ...         "signal_date": [
        ...             date(2025, 12, 31),
        ...             date(2026, 1, 30),
        ...         ],
        ...         "rebalance_date": [
        ...             date(2026, 1, 2),
        ...             date(2026, 2, 2),
        ...         ],
        ...     }
        ... )

        >>> cleaned_close_prices_dataset = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2025, 6, 30),
        ...             date(2025, 6, 30),
        ...             date(2025, 7, 30),
        ...             date(2025, 7, 30),
        ...             date(2025, 11, 28),
        ...             date(2025, 11, 28),
        ...             date(2025, 12, 30),
        ...             date(2025, 12, 30),
        ...         ],
        ...         "ticker": [
        ...             "AAPL", "MSFT",
        ...             "AAPL", "MSFT",
        ...             "AAPL", "MSFT",
        ...             "AAPL", "MSFT",
        ...         ],
        ...         "close": [
        ...             200.0, 350.0,
        ...             205.0, 360.0,
        ...             240.0, 400.0,
        ...             245.0, 410.0,
        ...         ],
        ...     }
        ... )

        >>> momentum_reference_table = get_prices_for_momentum_date_mapping(
        ...     factor_reference_table=factor_reference_table,
        ...     momentum_date_mapping_df=momentum_date_mapping_df,
        ...     cleaned_close_prices_dataset=cleaned_close_prices_dataset,
        ... )

        >>> momentum_reference_table
        shape: (4, 9)
        ┌───────────────┬───────────────┬─────────────┬────────────────┬────────┬────────────────┬────────────────┬──────────────┬────────────────┐
        │ lookback_date ┆ lag_base_date ┆ signal_date ┆ rebalance_date ┆ ticker ┆ lookback_close ┆ lag_base_close ┆ signal_close ┆ rebalance_open │
        │ ---           ┆ ---           ┆ ---         ┆ ---            ┆ ---    ┆ ---            ┆ ---            ┆ ---          ┆ ---            │
        │ date          ┆ date          ┆ date        ┆ date           ┆ str    ┆ f64            ┆ f64            ┆ f64          ┆ f64            │
        ╞═══════════════╪═══════════════╪═════════════╪════════════════╪════════╪════════════════╪════════════════╪══════════════╪════════════════╡
        │ 2025-06-30    ┆ 2025-11-28    ┆ 2025-12-31  ┆ 2026-01-02     ┆ AAPL   ┆ 200.0          ┆ 240.0          ┆ 250.0        ┆ 251.0          │
        │ 2025-06-30    ┆ 2025-11-28    ┆ 2025-12-31  ┆ 2026-01-02     ┆ MSFT   ┆ 350.0          ┆ 400.0          ┆ 420.0        ┆ 422.0          │
        │ 2025-07-30    ┆ 2025-12-30    ┆ 2026-01-30  ┆ 2026-02-02     ┆ AAPL   ┆ 205.0          ┆ 245.0          ┆ 255.0        ┆ 256.0          │
        │ 2025-07-30    ┆ 2025-12-30    ┆ 2026-01-30  ┆ 2026-02-02     ┆ MSFT   ┆ 360.0          ┆ 410.0          ┆ 430.0        ┆ 432.0          │
        └───────────────┴───────────────┴─────────────┴────────────────┴────────┴────────────────┴────────────────┴──────────────┴────────────────┘

        Note:
        - The column order in the actual output may differ from the example, but the order does not
          affect the resulting data or subsequent calculations.
        - This is a helper function for calculate_momentum and calculate_risk_adjusted_momentum.
    """
    lag_base_table = cleaned_close_prices_dataset.join(
        momentum_date_mapping_df,
        left_on="date",
        right_on="lag_base_date",
        how="semi",
        maintain_order="left",
    ).rename({"date": "lag_base_date", "close": "lag_base_close"})

    lookback_table = cleaned_close_prices_dataset.join(
        momentum_date_mapping_df,
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

    return (
        factor_reference_table.join(
            momentum_date_mapping_df,
            on=["rebalance_date", "signal_date"],
            how="left",
            maintain_order="left",
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
    )


def calculate_momentum(
    factor_reference_table: pl.DataFrame,
    cleaned_close_prices_dataset: pl.DataFrame,
    date_mapping_df: pl.DataFrame,
    trading_calendar: pl.DataFrame,
    lookback_period_for_total_returns: int,
    lookback_period_unit: Literal["w", "mo", "q", "y"],
) -> pl.DataFrame:
    """
    Calculate the percentage change of the lag base close relative to the lookback close.

    Arguments:
        factor_reference_table: A Polars DataFrame containing columns: signal_date, rebalance_date, ticker,
            signal_close, rebalance_open.

        cleaned_close_prices_dataset: A cleaned Polars DataFrame containing columns: date, ticker, close.

        date_mapping_df: A Polars DataFrame containing columns: signal_date, rebalance_date.

        trading_calendar: A Polars DataFrame containing one row per unique trading date.

        lookback_period_for_total_returns: The length of the lookback period.

        lookback_period_unit: The unit of the lookback period.
            "w": weeks,
            "mo": months,
            "q": quarters,
            "y": years

    Returns:
        factor_reference_table with an additional column: momentum_{lookback_period_for_total_returns}{lookback_period_unit}

    Example:
        >>> factor_reference_table = pl.DataFrame(
        ...     {
        ...         "signal_date": [
        ...             date(2025, 12, 31),
        ...             date(2025, 12, 31),
        ...             date(2026, 1, 30),
        ...             date(2026, 1, 30),
        ...         ],
        ...         "rebalance_date": [
        ...             date(2026, 1, 2),
        ...             date(2026, 1, 2),
        ...             date(2026, 2, 2),
        ...             date(2026, 2, 2),
        ...         ],
        ...         "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
        ...         "signal_close": [250.0, 420.0, 255.0, 430.0],
        ...         "rebalance_open": [251.0, 422.0, 256.0, 432.0],
        ...     }
        ... )

        >>> cleaned_close_prices_dataset = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2025, 6, 30),
        ...             date(2025, 6, 30),
        ...             date(2025, 7, 30),
        ...             date(2025, 7, 30),
        ...             date(2025, 11, 28),
        ...             date(2025, 11, 28),
        ...             date(2025, 12, 30),
        ...             date(2025, 12, 30),
        ...         ],
        ...         "ticker": [
        ...             "AAPL", "MSFT",
        ...             "AAPL", "MSFT",
        ...             "AAPL", "MSFT",
        ...             "AAPL", "MSFT",
        ...         ],
        ...         "close": [
        ...             200.0, 350.0,
        ...             205.0, 360.0,
        ...             240.0, 400.0,
        ...             245.0, 410.0,
        ...         ],
        ...     }
        ... )

        >>> date_mapping_df = pl.DataFrame(
        ...     {
        ...         "signal_date": [
        ...             date(2025, 12, 31),
        ...             date(2026, 1, 30),
        ...         ],
        ...         "rebalance_date": [
        ...             date(2026, 1, 2),
        ...             date(2026, 2, 2),
        ...         ],
        ...     }
        ... )

        >>> trading_calendar = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2025, 6, 30),
        ...             date(2025, 7, 30),
        ...             date(2025, 11, 28),
        ...             date(2025, 12, 30),
        ...             date(2025, 12, 31),
        ...             date(2026, 1, 30),
        ...         ]
        ...     }
        ... ).sort("date")

        >>> factor_reference_table = calculate_momentum(
        ...     factor_reference_table=factor_reference_table,
        ...     cleaned_close_prices_dataset=cleaned_close_prices_dataset,
        ...     date_mapping_df=date_mapping_df,
        ...     trading_calendar=trading_calendar,
        ...     lookback_period_for_total_returns=6,
        ...     lookback_period_unit="mo",
        ... )

        >>> factor_reference_table
        shape: (4, 6)
        ┌─────────────┬────────────────┬────────┬──────────────┬────────────────┬──────────────┐
        │ signal_date ┆ rebalance_date ┆ ticker ┆ signal_close ┆ rebalance_open ┆ momentum_6mo │
        │ ---         ┆ ---            ┆ ---    ┆ ---          ┆ ---            ┆ ---          │
        │ date        ┆ date           ┆ str    ┆ f64          ┆ f64            ┆ f64          │
        ╞═════════════╪════════════════╪════════╪══════════════╪════════════════╪══════════════╡
        │ 2025-12-31  ┆ 2026-01-02     ┆ AAPL   ┆ 250.0        ┆ 251.0          ┆ 0.2          │
        │ 2025-12-31  ┆ 2026-01-02     ┆ MSFT   ┆ 420.0        ┆ 422.0          ┆ 0.142857     │
        │ 2026-01-30  ┆ 2026-02-02     ┆ AAPL   ┆ 255.0        ┆ 256.0          ┆ 0.195122     │
        │ 2026-01-30  ┆ 2026-02-02     ┆ MSFT   ┆ 430.0        ┆ 432.0          ┆ 0.138889     │
        └─────────────┴────────────────┴────────┴──────────────┴────────────────┴──────────────┘

    Note:
    - The column order in the actual output may differ from the example, but the order does not
      affect the resulting data or subsequent calculations.
    - This function is designed to support sequential factor construction. Existing columns are
      preserved, and the risk-adjusted momentum factor is appended as a new column.
    """
    momentum_date_mapping_df = create_momentum_date_mapping(
        date_mapping_df=date_mapping_df,
        trading_calendar=trading_calendar,
        lookback_period_for_total_returns=lookback_period_for_total_returns,
        lookback_period_unit=lookback_period_unit,
    )

    momentum_reference_table = get_prices_for_momentum_date_mapping(
        factor_reference_table=factor_reference_table,
        momentum_date_mapping_df=momentum_date_mapping_df,
        cleaned_close_prices_dataset=cleaned_close_prices_dataset,
    )

    momentum = momentum_reference_table.select(
        col("signal_date"),
        col("ticker"),
        (col("lag_base_close") / col("lookback_close") - 1).alias(
            f"momentum_{lookback_period_for_total_returns}{lookback_period_unit}"
        ),
    )

    return factor_reference_table.join(
        momentum, on=["signal_date", "ticker"], how="left", maintain_order="left"
    )


def calculate_risk_adjusted_momentum(
    factor_reference_table: pl.DataFrame,
    cleaned_close_prices_dataset: pl.DataFrame,
    date_mapping_df: pl.DataFrame,
    trading_calendar: pl.DataFrame,
    lookback_period_for_total_returns: int,
    lookback_period_unit: Literal["w", "mo", "q", "y"],
) -> pl.DataFrame:
    """
    Calculate the percentage change of the lag base close relative to the lookback close
        with adjustment to the corresponding volatility.

    Arguments:
        factor_reference_table: A Polars DataFrame containing columns: signal_date, rebalance_date, ticker,
            signal_close, rebalance_open.

        cleaned_close_prices_dataset: A cleaned Polars DataFrame containing columns: date, ticker, close.

        date_mapping_df: A Polars DataFrame containing columns: signal_date, rebalance_date.

        trading_calendar: A Polars DataFrame containing one row per unique trading date.

        lookback_period_for_total_returns: The length of the lookback period.

        lookback_period_unit: The unit of the lookback period.
            "w": weeks,
            "mo": months,
            "q": quarters,
            "y": years

    Returns:
        The input DataFrame with an additional column:
            risk_adjusted_momentum_{lookback_period_for_total_returns}{lookback_period_unit}.
    
    Example:
        >>> factor_reference_table = pl.DataFrame(
        ...     {
        ...         "signal_date": [
        ...             date(2025, 12, 31),
        ...             date(2025, 12, 31),
        ...         ],
        ...         "rebalance_date": [
        ...             date(2026, 1, 2),
        ...             date(2026, 1, 2),
        ...         ],
        ...         "ticker": ["AAPL", "MSFT"],
        ...         "signal_close": [250.0, 420.0],
        ...         "rebalance_open": [251.0, 422.0],
        ...     }
        ... )

        >>> cleaned_close_prices_dataset = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2025, 6, 30),
        ...             date(2025, 6, 30),
        ...             date(2025, 7, 1),
        ...             date(2025, 7, 1),
        ...             date(2025, 9, 1),
        ...             date(2025, 9, 1),
        ...             date(2025, 11, 28),
        ...             date(2025, 11, 28),
        ...         ],
        ...         "ticker": [
        ...             "AAPL", "MSFT",
        ...             "AAPL", "MSFT",
        ...             "AAPL", "MSFT",
        ...             "AAPL", "MSFT",
        ...         ],
        ...         "close": [
        ...             200.0, 350.0,
        ...             220.0, 370.0,
        ...             230.0, 385.0,
        ...             240.0, 400.0,
        ...         ],
        ...     }
        ... )

        >>> date_mapping_df = pl.DataFrame(
        ...     {
        ...         "signal_date": [date(2025, 12, 31)],
        ...         "rebalance_date": [date(2026, 1, 2)],
        ...     }
        ... )

        >>> trading_calendar = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2025, 6, 30),
        ...             date(2025, 7, 1),
        ...             date(2025, 9, 1),
        ...             date(2025, 11, 28),
        ...             date(2025, 12, 31),
        ...             date(2026, 1, 2),
        ...         ]
        ...     }
        ... )

        >>> factor_reference_table = calculate_risk_adjusted_momentum(
        ...     factor_reference_table=factor_reference_table,
        ...     cleaned_close_prices_dataset=cleaned_close_prices_dataset,
        ...     date_mapping_df=date_mapping_df,
        ...     trading_calendar=trading_calendar,
        ...     lookback_period_for_total_returns=6,
        ...     lookback_period_unit="mo",
        ... )

        >>> factor_reference_table
        shape: (2, 6)
        ┌─────────────┬────────────────┬────────┬──────────────┬────────────────┬────────────────────────────┐
        │ signal_date ┆ rebalance_date ┆ ticker ┆ signal_close ┆ rebalance_open ┆ risk_adjusted_momentum_6mo │
        │ ---         ┆ ---            ┆ ---    ┆ ---          ┆ ---            ┆ ---                        │
        │ date        ┆ date           ┆ str    ┆ f64          ┆ f64            ┆ f64                        │
        ╞═════════════╪════════════════╪════════╪══════════════╪════════════════╪════════════════════════════╡
        │ 2025-12-31  ┆ 2026-01-02     ┆ AAPL   ┆ 250.0        ┆ 251.0          ┆ 6.683532                   │
        │ 2025-12-31  ┆ 2026-01-02     ┆ MSFT   ┆ 420.0        ┆ 422.0          ┆ 14.868901                  │
        └─────────────┴────────────────┴────────┴──────────────┴────────────────┴────────────────────────────┘

    Note:
    - The column order in the actual output may differ from the example, but the order does not
      affect the resulting data or subsequent calculations.
    - This function is designed to support sequential factor construction. Existing columns are
      preserved, and the risk-adjusted momentum factor is appended as a new column.
    """
    momentum_date_mapping_df = create_momentum_date_mapping(
        date_mapping_df=date_mapping_df,
        trading_calendar=trading_calendar,
        lookback_period_for_total_returns=lookback_period_for_total_returns,
        lookback_period_unit=lookback_period_unit,
    )

    momentum_reference_table = get_prices_for_momentum_date_mapping(
        factor_reference_table=factor_reference_table,
        momentum_date_mapping_df=momentum_date_mapping_df,
        cleaned_close_prices_dataset=cleaned_close_prices_dataset,
    )

    momentum = momentum_reference_table.select(
        col("signal_date"),
        col("ticker"),
        (col("lag_base_close") / col("lookback_close") - 1).alias(
            f"momentum_{lookback_period_for_total_returns}{lookback_period_unit}"
        ),
    )

    start_date = momentum_date_mapping_df.get_column("lookback_date").first()
    end_date = momentum_date_mapping_df.get_column("lag_base_date").last()

    vol = (
        cleaned_close_prices_dataset.with_columns(
            col("close").log().diff().over("ticker").alias("daily_return")
        )
        .filter(col("date").is_between(start_date, end_date))
        .join_where(
            momentum_date_mapping_df.select(
                "lookback_date",
                "lag_base_date",
                "signal_date",
            ),
            col("date").is_between(
                col("lookback_date"), col("lag_base_date"), closed="right"
            ),
        )
        .sort(["signal_date"])
        .group_by(["ticker", "signal_date"], maintain_order=True)
        .agg(
            col("daily_return")
            .std()
            .alias(f"vol_{lookback_period_for_total_returns}{lookback_period_unit}")
        )
    )

    risk_adjusted_vol = momentum.join(
        vol, on=["signal_date", "ticker"], how="left", maintain_order="left"
    ).select(
        col("signal_date"),
        col("ticker"),
        (
            col(f"momentum_{lookback_period_for_total_returns}{lookback_period_unit}")
            / col(f"vol_{lookback_period_for_total_returns}{lookback_period_unit}")
        ).alias(
            f"risk_adjusted_momentum_{lookback_period_for_total_returns}{lookback_period_unit}"
        ),
    )

    return factor_reference_table.join(
        risk_adjusted_vol,
        on=["signal_date", "ticker"],
        how="left",
        maintain_order="left",
    )
