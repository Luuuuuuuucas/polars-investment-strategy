import polars as pl
from polars import col


def construct_portfolio(
    ranked_candidates: pl.DataFrame,
    rebalance_dates: pl.Series,
    rank_col: str,
    top_n: int,
) -> pl.DataFrame:
    """
    Construct portfolio for each rebalance date.

    Arguments:
        ranked_candidates: A Polars DataFrame containing at least the columns: rebalance_date,
            ticker, rebalance_open, and the specified rank column.

        rebalance_dates: The predefined rebalance dates.

        rank_col: The column name of the rank column.

        top_n: The portfolio size.

    Returns:
        A Polars DataFrame containing columns: rebalance_date, ticker, rebalance_open,
            and the specified rank column.

    Example:
        >>> ranked_candidates = pl.DataFrame(
        ...     {
        ...         "rebalance_date": [
        ...             "2025-01-31", "2025-01-31", "2025-01-31", "2025-01-31",
        ...             "2025-02-28", "2025-02-28", "2025-02-28", "2025-02-28",
        ...         ],
        ...         "ticker": [
        ...             "AAPL", "MSFT", "NVDA", "AMZN",
        ...             "AAPL", "MSFT", "NVDA", "AMZN",
        ...         ],
        ...         "rebalance_open": [
        ...             230.0, 410.0, 140.0, 225.0,
        ...             235.0, 415.0, 145.0, 230.0,
        ...         ],
        ...         "momentum_rank": [
        ...             1, 2, 3, 4,
        ...             4, 1, 3, 2,
        ...         ],
        ...     }
        ... ).with_columns(
        ...     col("rebalance_date").str.to_date()
        ... )

        >>> rebalance_dates = ranked_candidates.get_column(
        ...     "rebalance_date"
        ... ).unique(maintain_order=True)

        >>> construct_portfolio(
        ...     ranked_candidates=ranked_candidates,
        ...     rebalance_dates=rebalance_dates,
        ...     rank_col="momentum_rank",
        ...     top_n=3,
        ... )
        shape: (6, 4)
        ┌────────────────┬────────┬────────────────┬───────────────┐
        │ rebalance_date ┆ ticker ┆ rebalance_open ┆ momentum_rank │
        │ ---            ┆ ---    ┆ ---            ┆ ---           │
        │ date           ┆ str    ┆ f64            ┆ i64           │
        ╞════════════════╪════════╪════════════════╪═══════════════╡
        │ 2025-01-31     ┆ AAPL   ┆ 230.0          ┆ 1             │
        │ 2025-01-31     ┆ MSFT   ┆ 410.0          ┆ 2             │
        │ 2025-01-31     ┆ NVDA   ┆ 140.0          ┆ 3             │
        │ 2025-02-28     ┆ AAPL   ┆ 235.0          ┆ 4             │
        │ 2025-02-28     ┆ MSFT   ┆ 415.0          ┆ 1             │
        │ 2025-02-28     ┆ NVDA   ┆ 145.0          ┆ 3             │
        └────────────────┴────────┴────────────────┴───────────────┘
    """
    portfolio_candidates = ranked_candidates.select(
        col("rebalance_date"), col("ticker"), col("rebalance_open"), col(rank_col)
    )

    portfolio_basket = []
    for i, current_date in enumerate(rebalance_dates):
        if i == 0:
            current_basket = portfolio_candidates.filter(
                (col(rank_col)) <= top_n, (col("rebalance_date") == current_date)
            )

            portfolio_basket.append(current_basket)
        else:
            previous_tickers = current_basket.get_column("ticker").to_list()

            current_candidates = portfolio_candidates.filter(
                col("rebalance_date") == current_date
            )

            retained_stocks = current_candidates.filter(
                col("ticker").is_in(previous_tickers)
            )

            replacement_count = top_n - retained_stocks.height

            new_stocks = current_candidates.filter(
                ~col("ticker").is_in(previous_tickers)
            ).bottom_k(
                k=replacement_count,
                by=rank_col,
            )

            current_basket = pl.concat(
                [
                    retained_stocks,
                    new_stocks,
                ]
            )

            portfolio_basket.append(current_basket)

    return pl.concat(portfolio_basket, how="vertical")