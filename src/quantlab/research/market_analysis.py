import pandas as pd

from quantlab.data.loader import load_market_data
from quantlab.analytics.returns import simple_returns, log_returns
from quantlab.analytics.statistics import daily_volatility, annualised_volatility, covariance, correlation

def load_assets(tickers: list[str], start: str, end:str) -> pd.DataFrame:
    if not tickers:
        raise ValueError("List of tickers is empty")

    tickers = list(dict.fromkeys(tickers))

    prices = {}

    for ticker in tickers:
        data = load_market_data(ticker, start=start, end=end)
        prices[ticker] = data["Close"]

    prices = pd.DataFrame(prices).dropna()

    if prices.empty:
        raise ValueError("No overlapping market data")

    return prices

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = {}
    for ticker in prices:
        returns[ticker] = simple_returns(prices[ticker])
    return pd.DataFrame(returns).dropna()

def volatility_report(returns: pd.DataFrame) -> pd.DataFrame:
    report = {}

    for ticker in returns.columns:
        report[ticker] = annualised_volatility(returns[ticker])

    return pd.Series(report, name="Annualised Volatility").to_frame()

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    matrix = {}

    for ticker_a in returns.columns:
        matrix[ticker_a] = {}
        for ticker_b in returns.columns:
            matrix[ticker_a][ticker_b] = correlation(returns[ticker_a], returns[ticker_b])

    return pd.DataFrame(matrix)

def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    matrix = {}

    for ticker_a in returns.columns:
        matrix[ticker_a] = {}
        for ticker_b in returns.columns:
            matrix[ticker_a][ticker_b] = covariance(returns[ticker_a], returns[ticker_b])
    return pd.DataFrame(matrix)