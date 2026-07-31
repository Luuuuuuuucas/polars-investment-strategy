import polars as pl


def fill_OHLCV_missing_values(
    stock_OHLCV: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame:
    cleaned_stock_OHLCV = (
        stock_OHLCV.with_columns(
            pl.col("close").forward_fill().over("ticker"),
            pl.col("open").forward_fill().over("ticker"),
            pl.col("volume").fill_null(0),
        )
        .with_columns(
            pl.col("high").fill_null(pl.col("close")),
            pl.col("low").fill_null(pl.col("close")),
        )
        .drop_nulls()
    )
    return cleaned_stock_OHLCV