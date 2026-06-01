import pandas as pd
import os

BASE_PATH = r"C:\Users\rushi\OneDrive\Desktop\Future Interns\Machine_Learning"

df = pd.read_csv(os.path.join(BASE_PATH, "daily_sales.csv"))
df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%B')
df['Quarter'] = df['Date'].dt.quarter
df['Weekday'] = df['Date'].dt.day_name()

df['Trend'] = (df['Date'] - df['Date'].min()).dt.days

df['Rolling_7'] = df['Sales'].rolling(7).mean()
df['Rolling_30'] = df['Sales'].rolling(30).mean()

df.to_csv(
    os.path.join(BASE_PATH, "powerbi_sales.csv"),
    index=False
)

print("DONE: Power BI dataset ready")