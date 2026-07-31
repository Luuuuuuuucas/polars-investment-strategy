# Modular Quantitative Backtesting Framework

A modular backtesting framework built with Polars.

## Features

- Universe Construction
- Signal Generation
- Portfolio Construction
- Portfolio Valuation
- Performance Analytics

## Project Structure

src/
    data/
    signal/
    portfolio/
    analytics/

## Examples

The notebooks demonstrate the complete workflow of the framework,
from signal generation to portfolio construction, valuation,
and performance analytics.

## Data

The project expects a cleaned daily OHLCV dataset stored under:

data/raw/

The example dataset used during development was downloaded from Yahoo Finance and preprocessed locally, so it is not included in this repository.