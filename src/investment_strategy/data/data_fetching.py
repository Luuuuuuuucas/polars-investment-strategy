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

    Parameters
    ----------
    tickers
        Yahoo Finance ticker symbols.
    start_date
        Inclusive start date in YYYY-MM-DD format.
    end_date
        Exclusive end date in YYYY-MM-DD format.

    Returns
    -------
    pl.DataFrame
        Columns:
        date, ticker, open, high, low, close, volume.

        OHLC prices are adjusted for stock splits and cash dividends because auto_adjust=True.
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
