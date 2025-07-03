from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from datetime import timedelta

def train_arima_and_forecast(df, steps=30):
    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=steps, freq='B')
    forecast_df = pd.DataFrame({'Forecast': forecast.values}, index=forecast_dates)
    return forecast_df
