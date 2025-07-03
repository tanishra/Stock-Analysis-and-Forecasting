def get_forecast_csv(forecast_df):
    return forecast_df.reset_index().rename(columns={"index": "Date"}).to_csv(index=False).encode()
