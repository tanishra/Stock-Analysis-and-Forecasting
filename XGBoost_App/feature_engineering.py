import pandas as pd
import numpy as np
import ta

def add_features(df: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Technical Indicators
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['Volatility'] = df['Close'].rolling(window=10).std()

    # Lag features
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)

    # Date-based features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # Target
    df['target'] = df['Close'].shift(-1)

    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df
