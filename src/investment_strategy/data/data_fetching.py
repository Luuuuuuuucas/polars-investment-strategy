from pathlib import Path

import pandas as pd
import polars as pl
import yfinance as yf


def download_yfinance_prices(
    tickers: list[str] | str,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """
    Download daily auto-adjusted OHLCV data from Yahoo Finance and return a normalized long-format Polars DataFrame.

    Arguments:
        tickers
            Yahoo Finance ticker symbols.
        start_date
            Inclusive start date in YYYY-MM-DD format.
        end_date
            Exclusive end date in YYYY-MM-DD format.

    Returns:
        A Polars DataFrame with columns: date, ticker, open, high, low, close, volume.
        The DataFrame is sorted by date and ticker.

    Example:
        >>> prices = download_yfinance_prices(
        ...     tickers=["AAPL", "MSFT"],
        ...     start_date="2026-01-05",
        ...     end_date="2026-01-08",
        ... )
        >>> prices
        shape: (6, 7)
        ┌────────────┬────────┬────────┬────────┬────────┬────────┬──────────┐
        │ date       ┆ ticker ┆ open   ┆ high   ┆ low    ┆ close  ┆ volume   │
        │ ---        ┆ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---      │
        │ date       ┆ str    ┆ f64    ┆ f64    ┆ f64    ┆ f64    ┆ i64      │
        ╞════════════╪════════╪════════╪════════╪════════╪════════╪══════════╡
        │ 2026-01-05 ┆ AAPL   ┆ ...    ┆ ...    ┆ ...    ┆ ...    ┆ ...      │
        │ 2026-01-05 ┆ MSFT   ┆ ...    ┆ ...    ┆ ...    ┆ ...    ┆ ...      │
        │ 2026-01-06 ┆ AAPL   ┆ ...    ┆ ...    ┆ ...    ┆ ...    ┆ ...      │
        │ 2026-01-06 ┆ MSFT   ┆ ...    ┆ ...    ┆ ...    ┆ ...    ┆ ...      │
        │ 2026-01-07 ┆ AAPL   ┆ ...    ┆ ...    ┆ ...    ┆ ...    ┆ ...      │
        │ 2026-01-07 ┆ MSFT   ┆ ...    ┆ ...    ┆ ...    ┆ ...    ┆ ...      │
        └────────────┴────────┴────────┴────────┴────────┴────────┴──────────┘

    Note: OHLC prices are adjusted for stock splits and cash dividends because auto_adjust=True.
    """

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        group_by="column",
        multi_level_index=True,
        progress=False,
        threads=True,
    )

    long_pd = (
        raw.stack(level="Ticker", future_stack=True)
        .rename_axis(index=["date", "ticker"])
        .reset_index()
    )

    prices = (
        pl.from_pandas(long_pd)
        .rename(
            {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        .select(
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
        )
        .cast(
            {
                "date": pl.Date,
                "ticker": pl.String,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            },
            strict=False,
        )
        .sort(["date", "ticker"])
    )

    return prices


if __name__ == "__main__":
    # Download S&P 500 stocks:
    constituents_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

    tickers = (
        pd.read_csv(constituents_url)["Symbol"]
        .str.replace(".", "-", regex=False)
        .drop_duplicates()
        .tolist()
    )

    sp500_market_data = download_yfinance_prices(
        tickers=tickers,
        start_date="2018-12-31",
        end_date="2026-06-01",
    )

    output_path = Path("data/raw/sp500_market_data.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sp500_market_data.write_parquet(output_path)

    # Download benchmarks:
    benchmark_market_data = download_yfinance_prices(
        tickers=[
            "^GSPC",  # S&P 500
            "^IXIC",  # Nasdaq Composite
            "^NDX",  # Nasdaq-100
            "^DJI",  # Dow Jones Industrial Average
            "^RUT",  # Russell 2000
        ],
        start_date="2018-12-31",
        end_date="2026-06-01",
    )

    output_path = Path("data/raw/benchmark_market_data.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark_market_data.write_parquet(output_path)
