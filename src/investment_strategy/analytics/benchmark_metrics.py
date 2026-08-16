import polars as pl
from polars import col
from datetime import date


def get_daily_benchmark_table(
    benchmark_data: pl.DataFrame,
    benchmark_ticker: str,
    backtest_start_date: date,
    backtest_end_date: date,
) -> pl.DataFrame:
    """
    Prepare a benchmark daily table.

    Arguments:
        benchmark_data: A Polars DataFrame with all available benchmarks containing columns: date, ticker,
            open, high, low.

        benchmark_ticker: The selected benchmark ticker.

        backtest_start_date: The start date of the backtest.

        backtest_end_date: The end date of the backtest.

    Returns:
        A Polars DataFrame containing columns: date, ticker, open, close, daily_return.

    Example:
        >>> daily_benchmark_table = get_daily_benchmark_table(
        ...     benchmark_data=benchmark_data,
        ...     benchmark_ticker="^GSPC",
        ...     backtest_start_date=date(2024, 1, 2),
        ...     backtest_end_date=date(2024, 1, 5),
        ... )

        >>> daily_benchmark_table
        shape: (4, 5)
        ┌────────────┬────────┬────────┬────────┬──────────────┐
        │ date       ┆ ticker ┆ open   ┆ close  ┆ daily_return │
        │ ---        ┆ ---    ┆ ---    ┆ ---    ┆ ---          │
        │ date       ┆ str    ┆ f64    ┆ f64    ┆ f64          │
        ╞════════════╪════════╪════════╪════════╪══════════════╡
        │ 2024-01-02 ┆ ^GSPC  ┆ 470.00 ┆ 472.00 ┆ 0.004255     │
        │ 2024-01-03 ┆ ^GSPC  ┆ 472.50 ┆ 471.00 ┆ -0.002119    │
        │ 2024-01-04 ┆ ^GSPC  ┆ 471.20 ┆ 474.00 ┆ 0.006369     │
        │ 2024-01-05 ┆ ^GSPC  ┆ 474.50 ┆ 476.00 ┆ 0.004219     │
        └────────────┴────────┴────────┴────────┴──────────────┘

    Note: The first daily return is calculated as `close / open - 1` to align the benchmark return series
        with the portfolio return series from the start of the backtest. This ensures that both series
        cover the same return period and can be directly compared when calculating metrics such as tracking error.
    """
    return (
        benchmark_data.filter(
            (col("ticker") == benchmark_ticker)
            & col("date").is_between(backtest_start_date, backtest_end_date)
        )
        .select(
            "date",
            "ticker",
            "open",
            "close",
        )
        .with_columns(
            col("close")
            .pct_change()
            .fill_null(col("close") / col("open") - 1)
            .alias("daily_return")
        )
    )


def get_portfolio_benchmark_returns_table(
    daily_benchmark_table: pl.DataFrame,
    daily_portfolio_value_return_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Prepare a portfolio-benchmark returns table.

    Arguments:
        daily_benchmark_table: A Polars DataFrame containing columns: date, ticker, open, close, daily_return,
            for benchmark ticker.

        daily_portfolio_value_return_df: A Polars DataFrame containing columns: date, portfolio_value,
            daily_return.

    Returns:
        A Polars DataFrame containing columns: date, portfolio_return, benchmark_return.

    Example:
        >>> daily_benchmark_table = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 3),
        ...             date(2024, 1, 4),
        ...         ],
        ...         "ticker": ["^GSPC"] * 3,
        ...         "open": [4700.0, 4740.0, 4720.0],
        ...         "close": [4747.0, 4710.0, 4767.2],
        ...         "daily_return": [0.01, -0.007793, 0.012144],
        ...     }
        ... )

        >>> daily_portfolio_value_return_df = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 3),
        ...             date(2024, 1, 4),
        ...         ],
        ...         "portfolio_value": [101500.0, 100992.5, 102507.3875],
        ...         "daily_return": [0.015, -0.005, 0.015],
        ...     }
        ... )

        >>> get_portfolio_benchmark_returns_table(
        ...     daily_benchmark_table=daily_benchmark_table,
        ...     daily_portfolio_value_return_df=daily_portfolio_value_return_df,
        ... )
        shape: (3, 3)
        ┌────────────┬──────────────────┬──────────────────┐
        │ date       ┆ portfolio_return ┆ benchmark_return │
        │ ---        ┆ ---              ┆ ---              │
        │ date       ┆ f64              ┆ f64              │
        ╞════════════╪══════════════════╪══════════════════╡
        │ 2024-01-02 ┆ 0.015            ┆ 0.01             │
        │ 2024-01-03 ┆ -0.005           ┆ -0.007793        │
        │ 2024-01-04 ┆ 0.015            ┆ 0.012144         │
        └────────────┴──────────────────┴──────────────────┘

    Note: This function prepares a table supporting benchmark metrics calculations.
    """
    return daily_portfolio_value_return_df.select(
        col("date"), col("daily_return").alias("portfolio_return")
    ).join(
        daily_benchmark_table.select(
            col("date"), col("daily_return").alias("benchmark_return")
        ),
        on="date",
        how="left",
        maintain_order="left",
    )


def calculate_portfolio_beta(portfolio_benchmark_returns_table: pl.DataFrame) -> float:
    """
    Calculate portfolio beta.

    Arguments:
        portfolio_benchmark_returns_table: A Polars DataFrame containing columns:
            date, portfolio_return, benchmark_return.

    Returns:
        A float representing portfolio beta.

    Example:
        >>> portfolio_benchmark_returns_table = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 3),
        ...             date(2024, 1, 4),
        ...             date(2024, 1, 5),
        ...         ],
        ...         "portfolio_return": [0.015, 0.030, -0.015, 0.045],
        ...         "benchmark_return": [0.010, 0.020, -0.010, 0.030],
        ...     }
        ... )

        >>> calculate_portfolio_beta(
        ...     portfolio_benchmark_returns_table=portfolio_benchmark_returns_table,
        ... )
        1.5
    """
    return (
        portfolio_benchmark_returns_table.select(
            pl.cov("portfolio_return", "benchmark_return")
        ).item()
    ) / (portfolio_benchmark_returns_table.get_column("benchmark_return").var())


def calculate_jensens_alpha(
    portfolio_benchmark_returns_table: pl.DataFrame,
    portfolio_beta: float,
    *,
    annual_rf: float = 0.0,
) -> float:
    """
    Calculate portfolio Jensen's alpha.

    Arguments:
        portfolio_benchmark_returns_table: A Polars DataFrame containing columns:
            date, portfolio_return, benchmark_return.

        portfolio_beta: Price volatility of the portfolio relative to the market.

        annual_rf: Annualized risk free rate. Default to 0.0.

    Returns:
        A float representing portfolio Jensen's alpha.

    Example:
        >>> portfolio_benchmark_returns_table = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 3),
        ...             date(2024, 1, 4),
        ...             date(2024, 1, 5),
        ...         ],
        ...         "portfolio_return": [0.015, 0.030, -0.015, 0.045],
        ...         "benchmark_return": [0.010, 0.020, -0.010, 0.030],
        ...     }
        ... )

        >>> calculate_jensens_alpha(
        ...     portfolio_benchmark_returns_table=portfolio_benchmark_returns_table,
        ...     portfolio_beta=1.5,
        ...     annual_rf=0.03,
        ... )
        0.01478026794314058
    """
    daily_rf = (1 + annual_rf) ** (1 / 252) - 1

    excess_returns_df = portfolio_benchmark_returns_table.with_columns(
        pl.lit(daily_rf).alias("daily_rf")
    ).with_columns(
        (col("portfolio_return") - col("daily_rf")).alias("portfolio_excess"),
        (col("benchmark_return") - col("daily_rf")).alias("benchmark_excess"),
    )

    daily_alpha = (
        excess_returns_df.get_column("portfolio_excess").mean()
        - portfolio_beta * excess_returns_df.get_column("benchmark_excess").mean()
    )

    return daily_alpha * 252


def calculate_r_squared(portfolio_benchmark_returns_table: pl.DataFrame) -> float:
    """
    Calculate portfolio R-squared.

    Arguments:
        portfolio_benchmark_returns_table: A Polars DataFrame containing columns:
            date, portfolio_return, benchmark_return.

    Returns:
        A float representing portfolio R-squared.

    Example:
        >>> portfolio_benchmark_returns_table = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 3),
        ...             date(2024, 1, 4),
        ...             date(2024, 1, 5),
        ...         ],
        ...         "portfolio_return": [0.015, 0.030, -0.015, 0.045],
        ...         "benchmark_return": [0.010, 0.020, -0.010, 0.030],
        ...     }
        ... )

        >>> calculate_R_squared(
        ...     portfolio_benchmark_returns_table=portfolio_benchmark_returns_table,
        ... )
        1.0
    """
    return (
        (
            portfolio_benchmark_returns_table.select(
                pl.cov("portfolio_return", "benchmark_return")
            ).item()
        )
        ** 2
    ) / (
        portfolio_benchmark_returns_table.get_column("benchmark_return").var()
        * portfolio_benchmark_returns_table.get_column("portfolio_return").var()
    )


def calculate_annualized_tracking_error(
    portfolio_benchmark_returns_table: pl.DataFrame,
) -> float:
    """
    Calculate annualized tracking error.

    Arguments:
        portfolio_benchmark_returns_table: A Polars DataFrame containing columns:
            date, portfolio_return, benchmark_return.

    Returns:
        A float representing annualized tracking error.

    Example:
        >>> portfolio_benchmark_returns_table = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 3),
        ...             date(2024, 1, 4),
        ...             date(2024, 1, 5),
        ...         ],
        ...         "portfolio_return": [0.015, 0.030, -0.015, 0.045],
        ...         "benchmark_return": [0.010, 0.020, -0.010, 0.030],
        ...     }
        ... )

        >>> calculate_annualized_tracking_error(
        ...     portfolio_benchmark_returns_table=portfolio_benchmark_returns_table,
        ... )
        0.135557957701
    """
    return (
        portfolio_benchmark_returns_table.get_column("portfolio_return")
        - portfolio_benchmark_returns_table.get_column("benchmark_return")
    ).std() * (252) ** 0.5


def calculate_information_ratio(
    portfolio_benchmark_returns_table: pl.DataFrame,
) -> float:
    """
    Calculate portfolio information ratio.

    Arguments:
        portfolio_benchmark_returns_table: A Polars DataFrame containing columns:
            date, portfolio_return, benchmark_return.

    Returns:
        A float representing porfolio information ratio.

    Example:
        >>> portfolio_benchmark_returns_table = pl.DataFrame(
        ...     {
        ...         "date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 3),
        ...             date(2024, 1, 4),
        ...             date(2024, 1, 5),
        ...         ],
        ...         "portfolio_return": [0.015, 0.030, -0.015, 0.045],
        ...         "benchmark_return": [0.010, 0.020, -0.010, 0.030],
        ...     }
        ... )

        >>> calculate_information_ratio(
        ...     portfolio_benchmark_returns_table=portfolio_benchmark_returns_table,
        ... )
        11.618950038622252
    """
    active_return = portfolio_benchmark_returns_table.get_column(
        "portfolio_return"
    ) - portfolio_benchmark_returns_table.get_column("benchmark_return")

    annualized_active_return = active_return.mean() * 252
    annualized_tracking_error = active_return.std() * 252**0.5

    return annualized_active_return / annualized_tracking_error


def calculate_treynor_ratio(
    daily_portfolio_value_return_df: pl.DataFrame,
    portfolio_beta: float,
    *,
    annual_rf: float = 0.0,
) -> float:
    """
    Calculate portfolio treynor ratio.

    Arguments:
        daily_portfolio_value_return_df: A Polars DataFrame containing columns: date, portfolio_value,
            daily_return.

        portfolio_beta: Price volatility of the portfolio relative to the market.

        annual_rf: Annualized risk free rate. Default to 0.0.

    Returns:
        A float representing portfolio treynor ratio.

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
        ...             101500.0,
        ...             104545.0,
        ...             102976.825,
        ...             107610.782125,
        ...         ],
        ...         "daily_return": [0.015, 0.030, -0.015, 0.045],
        ...     }
        ... )

        >>> calculate_treynor_ratio(
        ...     daily_portfolio_value_return_df=daily_portfolio_value_return_df,
        ...     portfolio_beta=1.5,
        ...     annual_rf=0.03,
        ... )
        3.1302929760758125
    """
    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    return (
        daily_portfolio_value_return_df.get_column("daily_return").mean() - daily_rf
    ) * 252 / portfolio_beta
