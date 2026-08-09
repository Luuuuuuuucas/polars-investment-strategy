import polars as pl
from polars import col


def calculate_daily_return_std(daily_portfolio_value_return_df: pl.DataFrame) -> float:
    """
    Calculate the standard deviation of daily portfolio returns.

    Arguments:
        daily_portfolio_value_return_df: A Polars DataFrame containing columns: date, portfolio_value, daily_return.

    Returns:
        A float representing the volatility of daily portfolio returns.

    Example:
        >>> daily_portfolio_value_return_df = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 3),
        ...             date(2024, 1, 4),
        ...             date(2024, 1, 5),
        ...         ],
        ...         "portfolio_value": [
        ...             100000.0,
        ...             101000.0,
        ...             100495.0,
        ...             101499.95,
        ...         ],
        ...         "daily_return": [
        ...             None,
        ...             0.01,
        ...             -0.005,
        ...             0.01,
        ...         ],
        ...     }
        ... )

        >>> calculate_daily_return_std(
        ...     daily_portfolio_value_return_df
        ... )
        0.008660254037844387
    """
    return daily_portfolio_value_return_df.get_column("daily_return").std()


def calculate_annualized_volatility(
    daily_portfolio_value_return_df: pl.DataFrame,
) -> float:
    """
    Calculate annualized portfolio return volatility assuming 252 trading days per year.

    Arguments:
        daily_portfolio_value_return_df: A Polars DataFrame containing columns: date, portfolio_value, daily_return.

    Returns:
        A float representing the annualized portfolio return volatility assuming 252 trading days per year.

    Example:
        >>> daily_portfolio_value_return_df = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 3),
        ...             date(2024, 1, 4),
        ...             date(2024, 1, 5),
        ...         ],
        ...         "portfolio_value": [
        ...             100000.0,
        ...             101000.0,
        ...             100495.0,
        ...             101499.95,
        ...         ],
        ...         "daily_return": [
        ...             None,
        ...             0.01,
        ...             -0.005,
        ...             0.01,
        ...         ],
        ...     }
        ... )

        >>> calculate_annualized_volatility(
        ...     daily_portfolio_value_return_df
        ... )
        0.137...
    """
    return daily_portfolio_value_return_df.get_column("daily_return").std() * (252**0.5)


def calculate_drawdown(daily_portfolio_value_return_df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate daily portfolio drawdown from the running maximum portfolio value.

    Arguments:
        daily_portfolio_value_return_df: A Polars DataFrame containing columns: date, portfolio_value, daily_return.
    
    Returns:
        daily_portfolio_value_return_df with an additional column: "drawdown".

    Example:
        >>> daily_portfolio_value_return_df = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 3),
        ...             date(2024, 1, 4),
        ...             date(2024, 1, 5),
        ...         ],
        ...         "portfolio_value": [
        ...             100000.0,
        ...             105000.0,
        ...             102000.0,
        ...             108000.0,
        ...         ],
        ...         "daily_return": [
        ...             None,
        ...             0.05,
        ...             -0.028571,
        ...             0.058824,
        ...         ],
        ...     }
        ... )

        >>> calculate_drawdown(
        ...     daily_portfolio_value_return_df
        ... )
        shape: (4, 4)
        ┌────────────┬─────────────────┬──────────────┬───────────┐
        │ date       │ portfolio_value │ daily_return │ drawdown  │
        │ ---        │ ---             │ ---          │ ---       │
        │ date       │ f64             │ f64          │ f64       │
        ╞════════════╪═════════════════╪══════════════╪═══════════╡
        │ 2024-01-02 │ 100000.0        │ null         │ 0.0       │
        │ 2024-01-03 │ 105000.0        │ 0.05         │ 0.0       │
        │ 2024-01-04 │ 102000.0        │ -0.028571    │ -0.028571 │
        │ 2024-01-05 │ 108000.0        │ 0.058824     │ 0.0       │
        └────────────┴─────────────────┴──────────────┴───────────┘
    """
    return (
        daily_portfolio_value_return_df.with_columns(
            (col("portfolio_value").cum_max()).alias("running_max")
        )
        .with_columns(
            (col("portfolio_value") / col("running_max") - 1).alias("drawdown")
        )
        .drop("running_max")
    )


def calculate_max_drawdown(drawdown_table: pl.DataFrame) -> float:
    """
    Calculate the maximum portfolio drawdown.

    Arguments:
        drawdown_table: A Polars DataFrame containing columns: date, portfolio_value, daily_return, drawdown.

    Returns:
        A float representing the maximum portfolio drawdown.

    Example:
        >>> drawdown_table = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 3),
        ...             date(2024, 1, 4),
        ...             date(2024, 1, 5),
        ...         ],
        ...         "portfolio_value": [
        ...             100000.0,
        ...             105000.0,
        ...             102000.0,
        ...             108000.0,
        ...         ],
        ...         "daily_return": [
        ...             None,
        ...             0.05,
        ...             -0.028571,
        ...             0.058824,
        ...         ],
        ...         "drawdown": [
        ...             0.0,
        ...             0.0,
        ...             -0.028571,
        ...             0.0,
        ...         ],
        ...     }
        ... )

        >>> calculate_max_drawdown(drawdown_table)
        -0.028571
    """
    return drawdown_table.get_column("drawdown").min()


def calculate_sharpe_ratio(
    mean_daily_return: float,
    annualized_volatility: float,
    *,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate the portfolio Sharpe ratio.

    Arguments:
        mean_daily_return: The arithmetic average return of the portfolio.

        annualized_volatility: The annualized portfolio return volatility assuming 252 trading days per year.

        risk_free_rate: The annualized risk-free rate. Defaults to 0.0.

    Returns:
        A float represting the portfolio Sharpe ratio.

    Example:
        >>> calculate_sharpe_ratio(
        ...     mean_daily_return=0.0005,
        ...     annualized_volatility=0.15,
        ...     risk_free_rate=0.03,
        ... )
        0.64
    """
    return (mean_daily_return * 252 - risk_free_rate) / annualized_volatility


def calculate_calmer_ratio(
    annualized_return_cagr: float,
    max_drawdown: float
) -> float:
    """
    Calculate the Calmar ratio of the portfolio.

    Arguments:
        annualized_return_cagr: The annualized return (CAGR) assuming 252 trading days per year.

        max_drawdown: The maximum portfolio drawdown.

    Returns:
        A float representing the Calmar ratio of the portfolio.

    Example:
        >>> calculate_calmer_ratio(
        ...     annualized_return_cagr=0.20,
        ...     max_drawdown=-0.10,
        ... )
        2.0
    """
    return annualized_return_cagr / abs(max_drawdown)