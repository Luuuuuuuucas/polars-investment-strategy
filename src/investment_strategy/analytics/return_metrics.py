import polars as pl
from polars import col


def prepare_daily_portfolio_price_return_df(
    daily_portfolio_table: pl.DataFrame,
) -> pl.DataFrame:
    return daily_portfolio_table.select(
        col("date"), col("portfolio_value"), col("daily_return")
    )


def calculate_total_return(daily_portfolio_price_return_df: pl.DataFrame) -> float:
    portfolio_values = daily_portfolio_price_return_df.get_column("portfolio_value")
    return portfolio_values.last() / portfolio_values.first() - 1


def calculate_annualized_return_CAGR(
    daily_portfolio_price_return_df: pl.DataFrame,
) -> float:
    portfolio_values = daily_portfolio_price_return_df.get_column("portfolio_value")
    n_periods = portfolio_values.len() - 1
    return (portfolio_values.last() / portfolio_values.first()) ** (252 / n_periods) - 1


def calculate_mean_daily_return(daily_portfolio_price_return_df: pl.DataFrame) -> float:
    daily_returns = daily_portfolio_price_return_df.get_column(
        "daily_return"
    ).drop_nulls()
    return daily_returns.mean()


def calculate_annualized_mean_return(
    daily_portfolio_price_return_df: pl.DataFrame,
) -> float:
    daily_returns = daily_portfolio_price_return_df.get_column(
        "daily_return"
    ).drop_nulls()
    return (1 + daily_returns.mean()) ** 252 - 1

