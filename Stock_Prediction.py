import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def get_data(ticker):
    stock_data = yf.download(ticker, start="2020-01-01")
    stock_data = stock_data[['Close']].dropna()
    stock_data.index = pd.to_datetime(stock_data.index)
    return stock_data

def stationary_check(series):
    result = adfuller(series)
    return result[1]  # p-value

def get_differencing_order(close_price):
    d = 0
    p_value = stationary_check(close_price)
    while p_value > 0.05 and d < 3:  # avoid too many differences
        d += 1
        close_price = close_price.diff().dropna()
        p_value = stationary_check(close_price)
    return d

def fit_model(data, d_order):
    try:
        model = ARIMA(data, order=(5, d_order, 5))  # Stable configuration
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=30)
        return forecast
    except Exception as e:
        raise RuntimeError(f"Model fitting failed: {e}")


def evaluate_model(original_price, d_order):
    train, test = original_price[:-30], original_price[-30:]
    predictions = fit_model(train, d_order)
    predictions.index = test.index  # align for comparison
    rmse = np.sqrt(mean_squared_error(test, predictions))
    return round(rmse, 2)

def get_forecast(original_price, d_order):
    forecast = fit_model(original_price, d_order)
    start_date = original_price.index[-1] + timedelta(days=1)
    forecast_index = pd.bdate_range(start=start_date, periods=30)  # Business days
    forecast_df = pd.DataFrame({'Date': forecast_index, 'Prediction': forecast})
    return forecast_df
