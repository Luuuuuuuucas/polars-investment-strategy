import polars as pl
from polars import col


def prepare_daily_portfolio_value_return_df(
    daily_portfolio_table: pl.DataFrame,
) -> pl.DataFrame:
    """
    Prepare the daily portfolio value return DataFrame used for return metrics calculations.

    Arguments:
        daily_portfolio_table: A Polars DataFrame containing columns: date, positions_value,
            cash_residual, portfolio_value, daily_return.

    Returns:
        A Polars DataFrame containing columns: date, portfolio_value, daily_return.

    Example:
        >>> daily_portfolio_table = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 3),
        ...             date(2024, 1, 4),
        ...         ],
        ...         "positions_value": [99800.0, 100500.0, 101200.0],
        ...         "cash_residual": [200.0, 200.0, 200.0],
        ...         "portfolio_value": [100000.0, 100700.0, 101400.0],
        ...         "daily_return": [None, 0.007, 0.006951],
        ...     }
        ... )

        >>> prepare_daily_portfolio_value_return_df(
        ...     daily_portfolio_table=daily_portfolio_table,
        ... )
        shape: (3, 3)
        ┌────────────┬─────────────────┬──────────────┐
        │ date       │ portfolio_value │ daily_return │
        │ ---        │ ---             │ ---          │
        │ date       │ f64             │ f64          │
        ╞════════════╪═════════════════╪══════════════╡
        │ 2024-01-02 │ 100000.0        │ null         │
        │ 2024-01-03 │ 100700.0        │ 0.007        │
        │ 2024-01-04 │ 101400.0        │ 0.006951     │
        └────────────┴─────────────────┴──────────────┘

    Note:
        Daily portfolio values are based on closing prices. Therefore, total return is measured from
        the first available close-valued portfolio observation to the final close-valued portfolio
        observation, rather than from the initial rebalance open value.
    """
    return daily_portfolio_table.select(
        col("date"), col("portfolio_value"), col("daily_return")
    )


def calculate_total_return(
    daily_portfolio_value_return_df: pl.DataFrame, initial_capital: int | float
) -> float:
    """
    Calculate the percentage change of the ending portfolio relative to the beginning portfolio value.

    Arguments:
        daily_portfolio_value_return_df: A Polars DataFrame containing columns: date, portfolio_value, daily_return.

        initial_capital: The initial amount of capital to invest in the portfolio.

    Returns:
        A float representing the percentage change of the ending portfolio relative to the beginning portfolio value.

    Example:
        >>> daily_portfolio_value_return_df = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 3),
        ...             date(2024, 1, 4),
        ...         ],
        ...         "portfolio_value": [100000.0, 102000.0, 105000.0],
        ...         "daily_return": [None, 0.02, 0.029412],
        ...     }
        ... )

        >>> calculate_total_return(
        ...     daily_portfolio_value_return_df,
        ...     initial_capital=100_000.0
        ... )
        0.05
    """
    portfolio_values = daily_portfolio_value_return_df.get_column("portfolio_value")
    return portfolio_values.last() / initial_capital - 1


def calculate_annualized_return_cagr(
    daily_portfolio_value_return_df: pl.DataFrame, initial_capital: int | float
) -> float:
    """
    Calculate the annualized return (CAGR) assuming 252 trading days per year.

    Arguments:
        daily_portfolio_value_return_df: A Polars DataFrame containing columns: date, portfolio_value, daily_return.

        initial_capital: The initial amount of capital to invest in the portfolio.

    Returns:
        A float representing the annualized return assuming 252 trading days per year.

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
        ...             102000.0,
        ...             103000.0,
        ...         ],
        ...         "daily_return": [
        ...             None,
        ...             0.01,
        ...             0.009901,
        ...             0.009804,
        ...         ],
        ...     }
        ... )

        >>> calculate_annualized_return_cagr(
        ...     daily_portfolio_value_return_df,
        ...     initial_capital=100_000.0
        ... )
        10.98...
    """
    portfolio_values = daily_portfolio_value_return_df.get_column("portfolio_value")
    n_periods = portfolio_values.len() - 1
    return (portfolio_values.last() / initial_capital) ** (252 / n_periods) - 1


def calculate_mean_daily_return(daily_portfolio_value_return_df: pl.DataFrame) -> float:
    """
    Calculate arithmetic average return.

    Arguments:
        daily_portfolio_value_return_df: A Polars DataFrame containing columns: date, portfolio_value, daily_return.

    Returns:
        A float representing arithmetic average return of the portfolio.

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

        >>> calculate_mean_daily_return(
        ...     daily_portfolio_value_return_df
        ... )
        0.005
    """
    daily_returns = daily_portfolio_value_return_df.get_column("daily_return")
    return daily_returns.mean()


def calculate_annualized_mean_return(
    daily_portfolio_value_return_df: pl.DataFrame,
) -> float:
    """
    Calculate the annualized arithmetic mean return assuming 252 trading days per year.

    Arguments:
        daily_portfolio_value_return_df: A Polars DataFrame containing columns: date, portfolio_value, daily_return.

    Returns:
        A float representing annualized mean return of the portfolio.

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

        >>> calculate_annualized_mean_return(
        ...     daily_portfolio_value_return_df
        ... )
        2.51...
    """
    daily_returns = daily_portfolio_value_return_df.get_column("daily_return")
    return (1 + daily_returns.mean()) ** 252 - 1
