import polars as pl
from polars import col
from typing import Literal


def get_portfolio_turnover_table(
    rebalance_level_table: pl.DataFrame, trade_level_table: pl.DataFrame
) -> pl.DataFrame:
    """
    Prepare a portfolio turnover table using rebalance level table and trade level table.

    Arguments:
        rebalance_level_table: A Polars DataFrame containing columns: rebalance_date, pre_rebalance_portfolio_value,
            post_rebalance_portfolio_value, cash_residual, transaction_cost.

        trade_level_table: A Polars DataFrame containing columns: rebalance_date, ticker, shares_traded, rebalance_open, trade_value,
            execution_cost, commission, transaction_cost, cash_flows.

    Returns:
        A Polars DataFrame containing columns: rebalance_date, buy_value, sell_value, pre_rebalance_portfolio_value,
            post_rebalance_portfolio_value, cash_residual, transaction_cost, one_way_turnover, transaction_cost_ratio.

    Example:
        >>> rebalance_level_table = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 2, 1),
        ...             date(2024, 3, 1),
        ...         ],
        ...         "pre_rebalance_portfolio_value": [
        ...             1000.0,
        ...             1049.02,
        ...         ],
        ...         "post_rebalance_portfolio_value": [
        ...             999.02,
        ...             1048.85,
        ...         ],
        ...         "cash_residual": [
        ...             149.02,
        ...             118.85,
        ...         ],
        ...         "transaction_cost": [
        ...             0.98,
        ...             0.17,
        ...         ],
        ...     }
        ... )

        >>> trade_level_table = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 2, 1),
        ...             date(2024, 2, 1),
        ...             date(2024, 3, 1),
        ...             date(2024, 3, 1),
        ...         ],
        ...         "ticker": ["A", "B", "A", "B"],
        ...         "shares_traded": [9, 4, -1, 1],
        ...         "rebalance_open": [50.0, 100.0, 60.0, 90.0],
        ...         "trade_value": [450.0, 400.0, 60.0, 90.0],
        ...         "execution_cost": [0.45, 0.40, 0.06, 0.09],
        ...         "commission": [0.09, 0.04, 0.01, 0.01],
        ...         "transaction_cost": [0.54, 0.44, 0.07, 0.10],
        ...         "cash_flows": [-450.54, -400.44, 59.93, -90.10],
        ...     }
        ... )

        >>> get_portfolio_turnover_table(
        ...     rebalance_level_table=rebalance_level_table,
        ...     trade_level_table=trade_level_table,
        ... )
        shape: (2, 9)
        ┌────────────────┬───────────┬────────────┬────────────────────────────────┬─────────────────────────────────┬───────────────┬──────────────────┬──────────────────┬────────────────────────┐
        │ rebalance_date │ buy_value │ sell_value │ pre_rebalance_portfolio_value  │ post_rebalance_portfolio_value  │ cash_residual │ transaction_cost │ one_way_turnover │ transaction_cost_ratio │
        │ ---            │ ---       │ ---        │ ---                            │ ---                             │ ---           │ ---              │ ---              │ ---                    │
        │ date           │ f64       │ f64        │ f64                            │ f64                             │ f64           │ f64              │ f64              │ f64                    │
        ╞════════════════╪═══════════╪════════════╪════════════════════════════════╪═════════════════════════════════╪═══════════════╪══════════════════╪══════════════════╪════════════════════════╡
        │ 2024-02-01     │ 850.0     │ 0.0        │ 1000.0                         │ 999.02                          │ 149.02        │ 0.98             │ 1.0              │ 0.00098                │
        │ 2024-03-01     │ 90.0      │ 60.0       │ 1049.02                        │ 1048.85                         │ 118.85        │ 0.17             │ 0.085794         │ 0.000162               │
        └────────────────┴───────────┴────────────┴────────────────────────────────┴─────────────────────────────────┴───────────────┴──────────────────┴──────────────────┴────────────────────────┘

    Note:
    - one_way_turnover measures portfolio turnover using the value of purchases relative to the portfolio value before rebalancing:
                            one_way_turnover = buy_value / pre_rebalance_portfolio_value
    - A one-way turnover of 1.0 means that purchases during the rebalance are equal to 100% of the portfolio value before
      rebalancing.
    - transaction_cost_ratio measures transaction cost relative to the portfolio value immediately before rebalancing:
                            transaction_cost_ratio = transaction_cost / pre_rebalance_portfolio_value
      This allows transaction costs to be compared across rebalance dates with different portfolio sizes.
    - For the first rebalance date, sell_value is zero because the portfolio starts entirely in cash. The corresponding one-way
      turnover is assigned a value of 1.0 to represent the initial portfolio construction.
    - The turnover for the initial portfolio construction is kept for completeness, but it should normally be excluded when
      calculating average turnover because it represents initial investment rather than an actual rebalance.
    - In the current version, this function identifies the initial portfolio construction by checking whether sell_value equals
      zero. This is valid because the framework currently assumes a long-only strategy and no external cash flows.
    """
    return pl.concat(
        [
            trade_level_table.group_by("rebalance_date").agg(
                (col("trade_value").filter(col("shares_traded") > 0).sum()).alias(
                    "buy_value"
                ),
                (col("trade_value").filter(col("shares_traded") < 0).sum()).alias(
                    "sell_value"
                ),
            ),
            rebalance_level_table.drop("rebalance_date"),
        ],
        how="horizontal_extend",
    ).with_columns(
        (
            pl.when(col("sell_value") == 0)
            .then(pl.lit(1.0))
            .otherwise(col("buy_value") / col("pre_rebalance_portfolio_value"))
            .alias("one_way_turnover")
        ),
        (col("transaction_cost") / col("pre_rebalance_portfolio_value")).alias(
            "transaction_cost_ratio"
        ),
    )


def calculate_average_turnover(portfolio_turnover_table: pl.DataFrame) -> float:
    """
    Calculate portfolio average turnover using one-way turnover.

    Arguments:
        portfolio_turnover_table: A Polars DataFrame containing columns: rebalance_date, buy_value, sell_value,
            pre_rebalance_portfolio_value, post_rebalance_portfolio_value, cash_residual, transaction_cost,
            one_way_turnover.

    Returns:
        A float representing portfolio average turnover.

    Example:
        >>> portfolio_turnover_table = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 2, 1),
        ...             date(2024, 3, 1),
        ...             date(2024, 4, 1),
        ...         ],
        ...         "buy_value": [
        ...             850.0,
        ...             90.0,
        ...             210.0,
        ...         ],
        ...         "sell_value": [
        ...             0.0,
        ...             60.0,
        ...             200.0,
        ...         ],
        ...         "pre_rebalance_portfolio_value": [
        ...             1000.0,
        ...             1049.02,
        ...             1100.0,
        ...         ],
        ...         "post_rebalance_portfolio_value": [
        ...             999.02,
        ...             1048.85,
        ...             1099.60,
        ...         ],
        ...         "cash_residual": [
        ...             149.02,
        ...             118.85,
        ...             89.60,
        ...         ],
        ...         "transaction_cost": [
        ...             0.98,
        ...             0.17,
        ...             0.40,
        ...         ],
        ...         "one_way_turnover": [
        ...             1.0,
        ...             90.0 / 1049.02,
        ...             210.0 / 1100.0,
        ...         ],
        ...     }
        ... )

        >>> calculate_average_turnover(
        ...     portfolio_turnover_table
        ... )
        0.138351...

    Note: The turnover for initial portfolio construction is excluded in the calculation
        as it represents initial investment rather than an actual rebalance.
    """
    return (
        portfolio_turnover_table.slice(1).select(col("one_way_turnover")).mean().item()
    )


def calculate_annualized_turnover(
    portfolio_turnover_table: pl.DataFrame,
    rebalance_frequency: int,
    rebalance_freq_unit: Literal["w", "mo", "q", "y"],
) -> float:
    """
    Calculate annualized turnover using one-way turnover.

    Arguments:
        portfolio_turnover_table: A Polars DataFrame containing columns: rebalance_date, buy_value, sell_value,
            pre_rebalance_portfolio_value, post_rebalance_portfolio_value, cash_residual, transaction_cost,
            one_way_turnover.

        rebalance_frequency: The frequency of rebalancing in the backtest period.

        rebalance_freq_unit: The unit of the rebalancing frequency.
            "w": weeks,
            "mo": months,
            "q": quarters,
            "y": years

    Returns:
        A float representing portfolio annualized turnover.

    Example:
        >>> portfolio_turnover_table = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 2, 1),
        ...             date(2024, 3, 1),
        ...             date(2024, 4, 1),
        ...         ],
        ...         "buy_value": [850.0, 90.0, 210.0],
        ...         "sell_value": [0.0, 60.0, 200.0],
        ...         "pre_rebalance_portfolio_value": [
        ...             1000.0,
        ...             1049.02,
        ...             1100.0,
        ...         ],
        ...         "post_rebalance_portfolio_value": [
        ...             999.02,
        ...             1048.85,
        ...             1099.60,
        ...         ],
        ...         "cash_residual": [149.02, 118.85, 89.60],
        ...         "transaction_cost": [0.98, 0.17, 0.40],
        ...         "one_way_turnover": [
        ...             1.0,
        ...             90.0 / 1049.02,
        ...             210.0 / 1100.0,
        ...         ],
        ...     }
        ... )

        >>> calculate_annualized_turnover(
        ...     portfolio_turnover_table=portfolio_turnover_table,
        ...     rebalance_frequency=2,
        ...     rebalance_freq_unit="mo",
        ... )
        0.83010...

    Note: This function assumes 52 weeks per year when annualizing weekly rebalance frequencies.
    """
    average_turnover = calculate_average_turnover(portfolio_turnover_table)

    periods_per_year = {"w": 52, "mo": 12, "q": 4, "y": 1}

    rebalances_per_year = periods_per_year[rebalance_freq_unit] / rebalance_frequency

    return average_turnover * rebalances_per_year


def calculate_total_transaction_cost(portfolio_turnover_table: pl.DataFrame) -> float:
    """
    Calculate total transaction cost occured using portfolio turnover table.

    Arguments:
        portfolio_turnover_table: A Polars DataFrame containing columns: rebalance_date, buy_value, sell_value,
            pre_rebalance_portfolio_value, post_rebalance_portfolio_value, cash_residual, transaction_cost,
            one_way_turnover.

    Returns:
        A float representing total transaction cost.

    Example:
        >>> portfolio_turnover_table = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 2, 1),
        ...             date(2024, 3, 1),
        ...             date(2024, 4, 1),
        ...         ],
        ...         "buy_value": [850.0, 90.0, 210.0],
        ...         "sell_value": [0.0, 60.0, 200.0],
        ...         "pre_rebalance_portfolio_value": [
        ...             1000.0,
        ...             1049.02,
        ...             1100.0,
        ...         ],
        ...         "post_rebalance_portfolio_value": [
        ...             999.02,
        ...             1048.85,
        ...             1099.60,
        ...         ],
        ...         "cash_residual": [149.02, 118.85, 89.60],
        ...         "transaction_cost": [0.98, 0.17, 0.40],
        ...         "one_way_turnover": [
        ...             1.0,
        ...             90.0 / 1049.02,
        ...             210.0 / 1100.0,
        ...         ],
        ...     }
        ... )

        >>> calculate_total_transaction_cost(
        ...     portfolio_turnover_table
        ... )
        1.55
    """
    return portfolio_turnover_table.get_column("transaction_cost").sum()


def calculate_total_transaction_cost_ratio(
    portfolio_turnover_table: pl.DataFrame, initial_capital: int | float
) -> float:
    """
    Calculate total transaction cost as a proportion of initial capital.

    Arguments:
        portfolio_turnover_table: A Polars DataFrame containing columns: rebalance_date, buy_value, sell_value,
            pre_rebalance_portfolio_value, post_rebalance_portfolio_value, cash_residual, transaction_cost,
            one_way_turnover.

        initial_capital: The initial amount of capital to invest in the portfolio.

    Returns:
        A float reprenting total transaction cost as a proportion of initial capital.

    Example:
        >>> portfolio_turnover_table = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 2, 1),
        ...             date(2024, 3, 1),
        ...             date(2024, 4, 1),
        ...         ],
        ...         "buy_value": [850.0, 90.0, 210.0],
        ...         "sell_value": [0.0, 60.0, 200.0],
        ...         "pre_rebalance_portfolio_value": [
        ...             1000.0,
        ...             1049.02,
        ...             1100.0,
        ...         ],
        ...         "post_rebalance_portfolio_value": [
        ...             999.02,
        ...             1048.85,
        ...             1099.60,
        ...         ],
        ...         "cash_residual": [149.02, 118.85, 89.60],
        ...         "transaction_cost": [0.98, 0.17, 0.40],
        ...         "one_way_turnover": [
        ...             1.0,
        ...             90.0 / 1049.02,
        ...             210.0 / 1100.0,
        ...         ],
        ...     }
        ... )

        >>> calculate_total_transaction_cost_ratio(
        ...     portfolio_turnover_table=portfolio_turnover_table,
        ...     initial_capital=1000,
        ... )
        0.00155
    """
    return (
        portfolio_turnover_table.get_column("transaction_cost").sum() / initial_capital
    )


def calculate_average_transaction_cost_ratio(
    portfolio_turnover_table: pl.DataFrame,
) -> float:
    """
    Calculate the average transaction cost ratio across all rebalance dates.

    Arguments:
        portfolio_turnover_table: A Polars DataFrame containing columns: rebalance_date, buy_value, sell_value,
            pre_rebalance_portfolio_value, post_rebalance_portfolio_value, cash_residual, transaction_cost,
            one_way_turnover.
        
    Returns:
        A float representing the average transaction cost as a proportion of pre-rebalance portfolio value.

    Example:
        >>> portfolio_turnover_table = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 2, 1),
        ...             date(2024, 3, 1),
        ...             date(2024, 4, 1),
        ...         ],
        ...         "buy_value": [850.0, 90.0, 210.0],
        ...         "sell_value": [0.0, 60.0, 200.0],
        ...         "pre_rebalance_portfolio_value": [
        ...             1000.0,
        ...             1049.02,
        ...             1100.0,
        ...         ],
        ...         "post_rebalance_portfolio_value": [
        ...             999.02,
        ...             1048.85,
        ...             1099.60,
        ...         ],
        ...         "cash_residual": [149.02, 118.85, 89.60],
        ...         "transaction_cost": [0.98, 0.17, 0.40],
        ...         "one_way_turnover": [
        ...             1.0,
        ...             90.0 / 1049.02,
        ...             210.0 / 1100.0,
        ...         ],
        ...         "transaction_cost_ratio": [
        ...             0.98 / 1000.0,
        ...             0.17 / 1049.02,
        ...             0.40 / 1100.0,
        ...         ],
        ...     }
        ... )

        >>> calculate_average_transaction_cost_ratio(
        ...     portfolio_turnover_table
        ... )
        0.000501...
    """
    return portfolio_turnover_table.get_column("transaction_cost_ratio").mean()
