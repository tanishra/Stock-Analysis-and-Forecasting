import yfinance as yf
import pandas as pd

def load_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Close']].dropna()
    df.index = pd.to_datetime(df.index)
    return df
