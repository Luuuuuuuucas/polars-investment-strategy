import polars as pl


def fill_OHLCV_missing_values(
    stock_OHLCV: pl.DataFrame,
) -> pl.DataFrame:
    """
    Fill missing values in stock_OHLCV using predefined rules:
    - Forward-fills missing close and open prices by at most one observation per ticker.
    - Replaces missing volume with 0.
    - Replaces missing high and low prices with the corresponding close price at the same date.

    Arguments:
        stock_OHLCV: Raw, uncleaned stock OHLCV data.

    Returns:
        A Polars DataFrame with daily stock OHLCV data missing-value handled using predefined rules.

    Example:
        >>> stock_OHLCV = pl.DataFrame({
        ...     "ticker": ["AAPL", "AAPL", "AAPL"],
        ...     "open": [100.0, None, None],
        ...     "high": [102.0, None, 105.0],
        ...     "low": [99.0, None, 103.0],
        ...     "close": [101.0, None, None],
        ...     "volume": [1000, None, 1200],
        ... })
        >>> fill_OHLCV_missing_values(stock_OHLCV)
        shape: (3, 6)
        ┌────────┬───────┬───────┬───────┬───────┬────────┐
        │ ticker ┆ open  ┆ high  ┆ low   ┆ close ┆ volume │
        │ ---    ┆ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---    │
        │ str    ┆ f64   ┆ f64   ┆ f64   ┆ f64   ┆ i64    │
        ╞════════╪═══════╪═══════╪═══════╪═══════╪════════╡
        │ AAPL   ┆ 100.0 ┆ 102.0 ┆ 99.0  ┆ 101.0 ┆ 1000   │
        │ AAPL   ┆ 100.0 ┆ 101.0 ┆ 101.0 ┆ 101.0 ┆ 0      │
        │ AAPL   ┆ null  ┆ 105.0 ┆ 103.0 ┆ null  ┆ 1200   │
        └────────┴───────┴───────┴───────┴───────┴────────┘

    Note: Missing close prices are forward-filled before missing high and low prices are replaced.
        This means that high and low prices may use a close price from the previous observation.
    """
    return stock_OHLCV.with_columns(
        pl.col("close").forward_fill(limit=1).over("ticker"),
        pl.col("open").forward_fill(limit=1).over("ticker"),
        pl.col("volume").fill_null(0),
    ).with_columns(
        pl.col("high").fill_null(pl.col("close")),
        pl.col("low").fill_null(pl.col("close")),
    )
