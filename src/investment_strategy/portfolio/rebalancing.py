import polars as pl
from polars import col
from datetime import date


def prepare_rebalance_allocation_df(
    sorted_filtered_signal_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Generates rebalance_allocation_df, which will be used as an input for portfolio calculations.
    """
    return sorted_filtered_signal_df.select(
        col("rebalance_date"),
        col("ticker"),
        col("rebalance_open"),
        col("portfolio_weight"),
    )


def calculate_position_sizing(
    rebalance_allocation_df: pl.DataFrame,
    current_rebalance_date: date,
    investable_value: int | float,
) -> pl.DataFrame:
    """
    Calculate integer share quantities for the current rebalance date.
    """
    return rebalance_allocation_df.filter(
        col("rebalance_date") == current_rebalance_date
    ).select(
        col("rebalance_date"),
        col("ticker"),
        (
            (investable_value * col("portfolio_weight") / col("rebalance_open"))
            .floor()
            .cast(pl.Int64)
            .alias("shares")
        ),
    )


def calculate_reference_portfolio_value(
    rebalance_signal_price_df: pl.DataFrame,
    current_positions: pl.DataFrame,
    reference_date: date,
) -> float:
    """
    Value the current positions at the specified reference date.
    """
    valued_positions = (
        current_positions.select(col("ticker"), col("shares"))
        .with_columns(pl.lit(reference_date).alias("rebalance_date"))
        .join(
            rebalance_signal_price_df.select(
                col("rebalance_date"),
                col("ticker"),
                col("rebalance_open"),
            ),
            on=["rebalance_date", "ticker"],
            how="left",
        )
    )

    return valued_positions.select((col("shares") * col("rebalance_open")).sum()).item()


def run_rebalance_simulation(
    rebalance_signal_price_df: pl.DataFrame,
    rebalance_allocation_df: pl.DataFrame,
    initial_capital: int | float,
    rebalance_dates: pl.Series,
) -> dict[str, pl.DataFrame]:
    current_cash_residual = float(initial_capital)
    current_positions: pl.DataFrame | None = None

    cash_residual_record = []
    portfolio_values_record = []
    positions_record = []

    for rebalance_date in rebalance_dates:
        if current_positions is None:
            positions_value_before_rebalance = 0.0
        else:
            positions_value_before_rebalance = calculate_reference_portfolio_value(
                rebalance_signal_price_df,
                current_positions,
                rebalance_date,
            )

        total_value_before_rebalance = (
            current_cash_residual + positions_value_before_rebalance
        )

        investable_value = total_value_before_rebalance

        current_positions = calculate_position_sizing(
            rebalance_allocation_df,
            rebalance_date,
            investable_value,
        )

        positions_value_after_rebalance = calculate_reference_portfolio_value(
            rebalance_signal_price_df,
            current_positions,
            rebalance_date,
        )

        current_cash_residual = investable_value - positions_value_after_rebalance

        portfolio_value_after_rebalance = (
            positions_value_after_rebalance + current_cash_residual
        )

        cash_residual_record.append(current_cash_residual)
        portfolio_values_record.append(portfolio_value_after_rebalance)

        positions_record.append(current_positions)

    rebalance_level_table = pl.DataFrame(
        {
            "rebalance_date": rebalance_dates,
            "portfolio_value": portfolio_values_record,
            "cash_residual": cash_residual_record,
        }
    )
    position_level_table = pl.concat(positions_record, how="vertical")

    return {
        "rebalance_level_table": rebalance_level_table,
        "position_level_table": position_level_table,
    }
