import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df
