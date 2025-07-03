import pandas as pd

def predict_next_n_days(model, scaler, df: pd.DataFrame, n_days: int = 30):
    df_temp = df.copy()
    preds = []

    for i in range(n_days):
        features = df_temp.drop(columns=['Date', 'target']).iloc[-1:]
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        preds.append(pred)

        # Append new row with shifted values
        new_row = features.copy()
        new_row.iloc[0, new_row.columns.get_loc('lag_1')] = pred
        for j in range(2, 11):  # lag_2 to lag_10
            lag_col = f'lag_{j}'
            prev_col = f'lag_{j-1}'
            if prev_col in new_row.columns and lag_col in new_row.columns:
                new_row.iloc[0, new_row.columns.get_loc(lag_col)] = new_row.iloc[0][prev_col]

        df_temp = pd.concat([df_temp, pd.DataFrame(new_row)], ignore_index=True)

    last_date = pd.to_datetime(df['Date'].iloc[-1])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_days)
    return pd.DataFrame({'Date': future_dates, 'Prediction': preds})
