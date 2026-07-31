import polars as pl
from polars import col


def calculate_daily_return_std(daily_portfolio_price_return_df: pl.DataFrame) -> float:
    return daily_portfolio_price_return_df.get_column("daily_return").std()


def calculate_annualized_volatility(
    daily_portfolio_price_return_df: pl.DataFrame,
) -> float:
    return daily_portfolio_price_return_df.get_column("daily_return").std() * (252**0.5)


def calculate_drawdown(daily_portfolio_price_return_df: pl.DataFrame) -> pl.DataFrame:
    return (
        daily_portfolio_price_return_df.with_columns(
            (col("portfolio_value").cum_max()).alias("running_max")
        )
        .with_columns(
            (col("portfolio_value") / col("running_max") - 1).alias("drawdown")
        )
        .drop("running_max")
    )


def calculate_max_drawdown(drawdown_table: pl.DataFrame) -> float:
    return drawdown_table.get_column("drawdown").min()


def calculate_sharpe_ratio(
    mean_daily_return: float,
    annualized_volatility: float,
    *,
    risk_free_rate: float = 0.0,
) -> float:
    return (mean_daily_return * 252 - risk_free_rate) / annualized_volatility
