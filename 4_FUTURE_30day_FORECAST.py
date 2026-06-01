import pandas as pd
import numpy as np
import os
import joblib

BASE_PATH = r"C:\Users\rushi\OneDrive\Desktop\Future Interns\Machine_Learning"

model = joblib.load(os.path.join(BASE_PATH, "sales_model.pkl"))

df = pd.read_csv(os.path.join(BASE_PATH, "processed_sales.csv"))
df['Date'] = pd.to_datetime(df['Date'])

future = []

current = df.copy()

for i in range(30):

    next_date = current['Date'].max() + pd.Timedelta(days=1)

    row = {}

    row['Date'] = next_date
    row['Day'] = next_date.day
    row['Month'] = next_date.month
    row['Weekday'] = next_date.weekday()
    row['WeekOfYear'] = int(next_date.isocalendar().week)

    row['Trend'] = (next_date - df['Date'].min()).days

    row['Lag_1'] = current['Sales'].iloc[-1]
    row['Lag_2'] = current['Sales'].iloc[-2]
    row['Lag_7'] = current['Sales'].iloc[-7]
    row['Lag_14'] = current['Sales'].iloc[-14]
    row['Lag_30'] = current['Sales'].iloc[-30]

    row['Roll_7_mean'] = current['Sales'].tail(7).mean()
    row['Roll_7_std'] = current['Sales'].tail(7).std()
    row['Roll_30_mean'] = current['Sales'].tail(30).mean()

    features = [
        'Day','Month','Weekday','WeekOfYear','Trend',
        'Lag_1','Lag_2','Lag_7','Lag_14','Lag_30',
        'Roll_7_mean','Roll_7_std','Roll_30_mean'
    ]

    X_future = pd.DataFrame([row])[features]

    pred = model.predict(X_future)[0]

    future.append([next_date, pred])

    current = pd.concat([
        current,
        pd.DataFrame([{
            'Date': next_date,
            'Sales': pred
        }])
    ], ignore_index=True)

forecast_df = pd.DataFrame(future, columns=['Date','Forecast'])

forecast_df.to_csv(
    os.path.join(BASE_PATH, "future_forecast.csv"),
    index=False
)

print("DONE: future forecast created")
print(forecast_df.head())