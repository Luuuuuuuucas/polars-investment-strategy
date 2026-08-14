import polars as pl
from polars import col
from datetime import date


def prepare_rebalance_allocation_df(
    weighted_portfolio_signal_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Prepare the rebalance allocation DataFrame used for portfolio calculations.

    Arguments:
        weighted_portfolio_signal_df: A Polars DataFrame containing the selected stocks with portfolio weights assigned.

    Returns:
        A Polars DataFrame containing columns: rebalance_date, ticker, rebalance_open, portfolio_weight.
    """
    return weighted_portfolio_signal_df.select(
        col("rebalance_date"),
        col("ticker"),
        col("rebalance_open"),
        col("portfolio_weight"),
    )


def calculate_position_sizing(
    rebalance_allocation_df: pl.DataFrame,
    current_rebalance_date: date,
    investable_value: int | float,
    execution_cost_rate: float,
    commission_per_share: int | float,
) -> pl.DataFrame:
    """
    Calculate integer share quantities for the current rebalance date using investable value.

    Arguments:
        rebalance_allocation_df: A Polars DataFrame containing columns:
            rebalance_date, ticker, rebalance_open, portfolio_weight.

        current_rebalance_date: The current rebalance date in the loop.

        investable_value: The amount of capital available for investment.

        execution_cost_rate: Proportional execution costs (e.g., bid-ask spread and slippage).

        commission_per_share: Fixed fee charged for each individual share bought or sold.

    Returns:
        A Polars DataFrame containing columns: rebalance_date, ticker, shares.

    Example:
        >>> rebalance_allocation_df = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 2, 1),
        ...             date(2024, 2, 1),
        ...             date(2024, 3, 1),
        ...         ],
        ...         "ticker": ["A", "B", "A"],
        ...         "rebalance_open": [50.0, 100.0, 60.0],
        ...         "portfolio_weight": [0.6, 0.4, 1.0],
        ...     }
        ... )

        >>> calculate_position_sizing(
        ...     rebalance_allocation_df=rebalance_allocation_df,
        ...     current_rebalance_date=date(2024, 2, 1),
        ...     investable_value=10000,
        ...     execution_cost_rate=0.001,
        ...     commission_per_share=0.01,
        ... )
        shape: (2, 3)
        ┌────────────────┬────────┬────────┐
        │ rebalance_date │ ticker │ shares │
        │ ---            │ ---    │ ---    │
        │ date           │ str    │ i64    │
        ╞════════════════╪════════╪════════╡
        │ 2024-02-01     │ A      │ 119    │
        │ 2024-02-01     │ B      │ 39     │
        └────────────────┴────────┴────────┘

    Note: This is a helper function used by run_rebalance_simulation.
    """
    cost_per_share = col("rebalance_open") * execution_cost_rate + commission_per_share
    return rebalance_allocation_df.filter(
        col("rebalance_date") == current_rebalance_date
    ).select(
        col("rebalance_date"),
        col("ticker"),
        (
            investable_value
            * col("portfolio_weight")
            / (cost_per_share + col("rebalance_open"))
        )
        .floor()
        .cast(pl.Int64)
        .alias("shares"),
    )


def calculate_reference_position_value(
    factor_reference_table: pl.DataFrame,
    post_rebalance_positions: pl.DataFrame,
    reference_date: date,
) -> float:
    """
    Value the current positions at the specified reference date.

    Arguments:
        factor_reference_table: A Polars DataFrame containing columns: lookback_date, lag_base_date, signal_date,
            rebalance_date, ticker, lookback_close, lag_base_close, signal_close, rebalance_open.

        post_rebalance_positions: A Polars DataFrame containing the current holdings.
            At minimum, the columns, ticker and shares, must be present.

        reference_date: The specific date where the valuation occurs.

    Returns:
        A float representing the position value.

    Example:
        >>> post_rebalance_positions = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 1, 2),
        ...             date(2024, 1, 2),
        ...         ],
        ...         "ticker": ["A", "B"],
        ...         "shares": [120, 40],
        ...     }
        ... )

        >>> factor_reference_table = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 2, 1),
        ...             date(2024, 2, 1),
        ...         ],
        ...         "ticker": ["A", "B"],
        ...         "rebalance_open": [50.0, 100.0],
        ...     }
        ... )

        >>> calculate_reference_position_value(
        ...     factor_reference_table=factor_reference_table,
        ...     post_rebalance_positions=post_rebalance_positions,
        ...     reference_date=date(2024, 2, 1),
        ... )
        10000.0

    Note: This is a helper function used by run_rebalance_simulation. The positions are valued before and after each rebalance.
    """
    valued_positions = (
        post_rebalance_positions.select(col("ticker"), col("shares"))
        .with_columns(pl.lit(reference_date).alias("rebalance_date"))
        .join(
            factor_reference_table.select(
                col("rebalance_date"),
                col("ticker"),
                col("rebalance_open"),
            ),
            on=["rebalance_date", "ticker"],
            how="left",
            maintain_order="left",
        )
    )

    return valued_positions.select((col("shares") * col("rebalance_open")).sum()).item()


def calculate_trade_cash_flows(
    factor_reference_table: pl.DataFrame,
    pre_rebalance_positions: pl.DataFrame,
    post_rebalance_positions: pl.DataFrame,
    rebalance_date: date,
    execution_cost_rate: float,
    commission_per_share: int | float,
) -> pl.DataFrame:
    """
    Calculate cash flows and transaction costs for a specific rebalance date.

    Arguments:
        factor_reference_table: A Polars DataFrame containing columns: lookback_date, lag_base_date, signal_date,
            rebalance_date, ticker, lookback_close, lag_base_close, signal_close, rebalance_open.

        pre_rebalance_positions: A Polars DataFrame containing columns:
            rebalance_date, ticker, shares, representing portfolio positions before rebalancing.

        post_rebalance_positions: A Polars DataFrame containing columns:
            rebalance_date, ticker, shares, representing target portfolio positions after rebalancing.

        rebalance_date: The current rebalance date.

        execution_cost_rate: Proportional execution costs (e.g., bid-ask spread and slippage).

        commission_per_share: Fixed fee charged for each individual share bought or sold.

    Returns:
        A Polars DataFrame containing columns: rebalance_date, ticker, shares_traded, rebalance_open,
            trade_value, execution_cost, commission, transaction_cost, cash_flows.

    Example:
        >>> factor_reference_table = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 3, 1),
        ...             date(2024, 3, 1),
        ...         ],
        ...         "ticker": ["A", "B"],
        ...         "rebalance_open": [60.0, 90.0],
        ...     }
        ... )

        >>> pre_rebalance_positions = pl.DataFrame(
        ...     {
        ...         "ticker": ["A", "B"],
        ...         "shares": [9, 4],
        ...     }
        ... )

        >>> post_rebalance_positions = pl.DataFrame(
        ...     {
        ...         "ticker": ["A", "B"],
        ...         "shares": [8, 5],
        ...     }
        ... )

        >>> calculate_trade_cash_flows(
        ...     factor_reference_table=factor_reference_table,
        ...     pre_rebalance_positions=pre_rebalance_positions,
        ...     post_rebalance_positions=post_rebalance_positions,
        ...     rebalance_date=date(2024, 3, 1),
        ...     execution_cost_rate=0.001,
        ...     commission_per_share=0.01,
        ... )
        shape: (2, 9)
        ┌────────────────┬────────┬───────────────┬────────────────┬─────────────┬────────────────┬────────────┬──────────────────┬────────────┐
        │ rebalance_date │ ticker │ shares_traded │ rebalance_open │ trade_value │ execution_cost │ commission │ transaction_cost │ cash_flows │
        │ ---            │ ---    │ ---           │ ---            │ ---         │ ---            │ ---        │ ---              │ ---        │
        │ date           │ str    │ i64           │ f64            │ f64         │ f64            │ f64        │ f64              │ f64        │
        ╞════════════════╪════════╪═══════════════╪════════════════╪═════════════╪════════════════╪════════════╪══════════════════╪════════════╡
        │ 2024-03-01     │ A      │ -1            │ 60.0           │ 60.0        │ 0.06           │ 0.01       │ 0.07             │ 59.93      │
        │ 2024-03-01     │ B      │ 1             │ 90.0           │ 90.0        │ 0.09           │ 0.01       │ 0.10             │ -90.10     │
        └────────────────┴────────┴───────────────┴────────────────┴─────────────┴────────────────┴────────────┴──────────────────┴────────────┘

    Note:
    - Positive shares_traded values represent shares bought, while negative values represent shares sold.
    - Trading costs are always recorded as positive values.
    - Buy trades generate negative cash flows, while sell trades generate positive cash flows net of transaction costs.
    - This is a helper function used by run_rebalance_simulation.
    """
    cost_per_share = col("rebalance_open") * execution_cost_rate + commission_per_share

    return (
        pre_rebalance_positions.select(col("ticker"), col("shares"))
        .join(
            post_rebalance_positions.select(col("ticker"), col("shares")),
            on="ticker",
            how="full",
            nulls_equal=True,
            coalesce=True,
            maintain_order="left"
        )
        .with_columns(col("shares").fill_null(0), col("shares_right").fill_null(0))
        .with_columns(
            (col("shares_right") - col("shares")).alias("shares_traded"),
            pl.lit(rebalance_date).alias("rebalance_date"),
        )
        .select(col("ticker"), col("shares_traded"), col("rebalance_date"))
        .join(
            factor_reference_table.select(
                col("rebalance_date"),
                col("ticker"),
                col("rebalance_open"),
            ),
            on=["rebalance_date", "ticker"],
            how="left",
            maintain_order="left",
        )
        .with_columns(
            (
                abs(col("shares_traded")) * col("rebalance_open") * execution_cost_rate
            ).alias("execution_cost"),
            (abs(col("shares_traded")) * commission_per_share).alias("commission"),
            (abs(col("shares_traded")) * cost_per_share).alias("transaction_cost"),
            (abs(col("shares_traded")) * col("rebalance_open")).alias("trade_value"),
        )
        .with_columns(
            (
                -col("shares_traded") * col("rebalance_open") - col("transaction_cost")
            ).alias("cash_flows")
        )
        .select(
            col("rebalance_date"),
            col("ticker"),
            col("shares_traded"),
            col("rebalance_open"),
            col("trade_value"),
            col("execution_cost"),
            col("commission"),
            col("transaction_cost"),
            col("cash_flows"),
        )
    )


def run_rebalance_simulation(
    factor_reference_table: pl.DataFrame,
    rebalance_allocation_df: pl.DataFrame,
    initial_capital: int | float,
    rebalance_dates: pl.Series,
    execution_cost_rate: float,
    commission_per_share: int | float,
) -> dict[str, pl.DataFrame]:
    """
    Simulate all the rebalance occurrences. Record portfolio value, cash residual, number of shares,
        transaction cost, cash flows, and number of shares of each asset within each rebalance date.

    Arguments:
        factor_reference_table: A Polars DataFrame containing columns: lookback_date, lag_base_date, signal_date,
            rebalance_date, ticker, lookback_close, lag_base_close, signal_close, rebalance_open.

        rebalance_allocation_df: A Polars DataFrame containing columns:
            rebalance_date, ticker, rebalance_open, portfolio_weight.

        initial_capital: The initial amount of capital to invest in the portfolio.

        rebalance_dates: The predefined rebalance dates.

        execution_cost_rate: Proportional execution costs (e.g., bid-ask spread and slippage).

        commission_per_share: Fixed fee charged for each individual share bought or sold.

    Example:
        >>> factor_reference_table = pl.DataFrame(
        ...     {
        ...         "lookback_date": [
        ...             date(2023, 7, 1),
        ...             date(2023, 7, 1),
        ...             date(2023, 8, 1),
        ...             date(2023, 8, 1),
        ...         ],
        ...         "lag_base_date": [
        ...             date(2023, 12, 1),
        ...             date(2023, 12, 1),
        ...             date(2024, 1, 1),
        ...             date(2024, 1, 1),
        ...         ],
        ...         "signal_date": [
        ...             date(2024, 1, 31),
        ...             date(2024, 1, 31),
        ...             date(2024, 2, 29),
        ...             date(2024, 2, 29),
        ...         ],
        ...         "rebalance_date": [
        ...             date(2024, 2, 1),
        ...             date(2024, 2, 1),
        ...             date(2024, 3, 1),
        ...             date(2024, 3, 1),
        ...         ],
        ...         "ticker": ["A", "B", "A", "B"],
        ...         "lookback_close": [40.0, 80.0, 45.0, 85.0],
        ...         "lag_base_close": [48.0, 95.0, 55.0, 92.0],
        ...         "signal_close": [49.0, 98.0, 58.0, 91.0],
        ...         "rebalance_open": [50.0, 100.0, 60.0, 90.0],
        ...     }
        ... )

        >>> rebalance_allocation_df = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             date(2024, 2, 1),
        ...             date(2024, 2, 1),
        ...             date(2024, 3, 1),
        ...             date(2024, 3, 1),
        ...         ],
        ...         "ticker": ["A", "B", "A", "B"],
        ...         "rebalance_open": [50.0, 100.0, 60.0, 90.0],
        ...         "portfolio_weight": [0.5, 0.5, 0.5, 0.5],
        ...     }
        ... )

        >>> rebalance_dates = pl.Series(
        ...     "rebalance_date",
        ...     [date(2024, 2, 1), date(2024, 3, 1)],
        ... )

        >>> simulation_result = run_rebalance_simulation(
        ...     factor_reference_table=factor_reference_table,
        ...     rebalance_allocation_df=rebalance_allocation_df,
        ...     initial_capital=1000,
        ...     rebalance_dates=rebalance_dates,
        ...     execution_cost_rate=0.001,
        ...     commission_per_share=0.01,
        ... )

        >>> simulation_result["rebalance_level_table"]
        shape: (2, 5)
        ┌────────────────┬─────────────────────────────────┬──────────────────────────────────┬───────────────┬──────────────────┐
        │ rebalance_date │ pre_rebalance_portfolio_value   │ post_rebalance_portfolio_value   │ cash_residual │ transaction_cost │
        │ ---            │ ---                             │ ---                              │ ---           │ ---              │
        │ date           │ f64                             │ f64                              │ f64           │ f64              │
        ╞════════════════╪═════════════════════════════════╪══════════════════════════════════╪═══════════════╪══════════════════╡
        │ 2024-02-01     │ 1000.00                         │ 999.02                           │ 149.02        │ 0.98             │
        │ 2024-03-01     │ 1049.02                         │ 1048.85                          │ 118.85        │ 0.17             │
        └────────────────┴─────────────────────────────────┴──────────────────────────────────┴───────────────┴──────────────────┘

        >>> simulation_result["position_level_table"]
        shape: (4, 3)
        ┌────────────────┬────────┬────────┐
        │ rebalance_date │ ticker │ shares │
        │ ---            │ ---    │ ---    │
        │ date           │ str    │ i64    │
        ╞════════════════╪════════╪════════╡
        │ 2024-02-01     │ A      │ 9      │
        │ 2024-02-01     │ B      │ 4      │
        │ 2024-03-01     │ A      │ 8      │
        │ 2024-03-01     │ B      │ 5      │
        └────────────────┴────────┴────────┘

        >>> simulation_result["trade_level_table"]
        shape: (4, 9)
        ┌────────────────┬────────┬───────────────┬────────────────┬─────────────┬────────────────┬────────────┬──────────────────┬────────────┐
        │ rebalance_date │ ticker │ shares_traded │ rebalance_open │ trade_value │ execution_cost │ commission │ transaction_cost │ cash_flows │
        │ ---            │ ---    │ ---           │ ---            │ ---         │ ---            │ ---        │ ---              │ ---        │
        │ date           │ str    │ i64           │ f64            │ f64         │ f64            │ f64        │ f64              │ f64        │
        ╞════════════════╪════════╪═══════════════╪════════════════╪═════════════╪════════════════╪════════════╪══════════════════╪════════════╡
        │ 2024-02-01     │ A      │ 9             │ 50.0           │ 450.0       │ 0.45           │ 0.09       │ 0.54             │ -450.54    │
        │ 2024-02-01     │ B      │ 4             │ 100.0          │ 400.0       │ 0.4            │ 0.04       │ 0.44             │ -400.44    │
        │ 2024-03-01     │ A      │ -1            │ 60.0           │ -60.0       │ 0.06           │ 0.01       │ 0.07             │ 59.93      │
        │ 2024-03-01     │ B      │ 1             │ 90.0           │ 90.0        │ 0.09           │ 0.01       │ 0.10             │ -90.10     │
        └────────────────┴────────┴───────────────┴────────────────┴─────────────┴────────────────┴────────────┴──────────────────┴────────────┘

    Returns:
        A dictionary containing:
            "rebalance_level_table":
                A Polars DataFrame containing columns: rebalance_date, pre_rebalance_portfolio_value,
                    post_rebalance_portfolio_value, cash_residual, transaction_cost.

            "position_level_table":
                A Polars DataFrame containing columns: rebalance_date, ticker, shares.

            "trade_level_table":
                A Polars DataFrame containing columns: rebalance_date, ticker, shares_traded, rebalance_open, trade_value,
                    execution_cost, commission, transaction_cost, cash_flows.

    Note:
    - In trade_level_table, positive shares_traded values represent shares bought, while negative values represent shares sold.
    - cash_residual in rebalance_level_table may be negative because transaction costs are deducted after position sizing and
      can cause cash outflows to exceed the available cash balance.
    - investable_value is currently set equal to pre_rebalance_portfolio_value. It is kept as a separate variable so that the
      amount allocated to position sizing can be adjusted independently in future versions, for example by reserving part of
      the portfolio value for estimated transaction costs to reduce or prevent negative cash residuals.
    """
    current_cash_residual = float(initial_capital)
    post_rebalance_positions = pl.DataFrame(schema={"ticker": str, "shares": pl.Int64})

    cash_residual_record = []
    pre_rebalance_portfolio_values_record = []
    positions_record = []
    post_rebalance_portfolio_values_record = []
    transaction_cost_record = []
    cash_flows_record = []

    for i, rebalance_date in enumerate(rebalance_dates):
        if i == 0:
            pre_rebalance_positions_value = 0.0
        else:
            pre_rebalance_positions_value = calculate_reference_position_value(
                factor_reference_table,
                post_rebalance_positions,
                rebalance_date,
            )

        pre_rebalance_portfolio_value = (
            current_cash_residual + pre_rebalance_positions_value
        )

        investable_value = pre_rebalance_portfolio_value

        pre_rebalance_positions = post_rebalance_positions

        post_rebalance_positions = calculate_position_sizing(
            rebalance_allocation_df,
            rebalance_date,
            investable_value,
            execution_cost_rate,
            commission_per_share,
        )

        trading_cash_flows = calculate_trade_cash_flows(
            factor_reference_table,
            pre_rebalance_positions,
            post_rebalance_positions,
            rebalance_date,
            execution_cost_rate,
            commission_per_share,
        )

        post_rebalance_positions_value = calculate_reference_position_value(
            factor_reference_table,
            post_rebalance_positions,
            rebalance_date,
        )

        current_cash_residual = (
            current_cash_residual + trading_cash_flows.get_column("cash_flows").sum()
        )

        post_rebalance_portfolio_value = (
            post_rebalance_positions_value + current_cash_residual
        )

        cash_residual_record.append(current_cash_residual)

        pre_rebalance_portfolio_values_record.append(pre_rebalance_portfolio_value)

        positions_record.append(post_rebalance_positions)

        post_rebalance_portfolio_values_record.append(post_rebalance_portfolio_value)

        transaction_cost_record.append(
            trading_cash_flows.get_column("transaction_cost").sum()
        )

        cash_flows_record.append(trading_cash_flows)

    position_level_table: pl.DataFrame = pl.concat(positions_record, how="vertical")

    rebalance_level_table = pl.DataFrame(
        {
            "rebalance_date": rebalance_dates,
            "pre_rebalance_portfolio_value": pre_rebalance_portfolio_values_record,
            "post_rebalance_portfolio_value": post_rebalance_portfolio_values_record,
            "cash_residual": cash_residual_record,
            "transaction_cost": transaction_cost_record,
        }
    )

    trade_level_table: pl.DataFrame = pl.concat(cash_flows_record, how="vertical")

    return {
        "rebalance_level_table": rebalance_level_table,
        "position_level_table": position_level_table,
        "trade_level_table": trade_level_table,
    }
