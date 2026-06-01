import pandas as pd
import os

BASE_PATH = r"C:\Users\rushi\OneDrive\Desktop\Future Interns\Machine_Learning"

print("Loading dataset...")

df = pd.read_excel(
    os.path.join(BASE_PATH, "Online Retail.xlsx")
)

# -----------------------------
# CLEAN DATA
# -----------------------------

df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df.dropna(subset=['CustomerID'])

df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# -----------------------------
# CREATE REVENUE
# -----------------------------

df['Revenue'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['Date'] = df['InvoiceDate'].dt.date

# -----------------------------
# DAILY SALES
# -----------------------------

daily_sales = df.groupby('Date')['Revenue'].sum().reset_index()
daily_sales.columns = ['Date', 'Sales']
daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])

daily_sales = daily_sales.sort_values('Date')

# Fill missing dates (VERY IMPORTANT IMPROVEMENT)
full_range = pd.date_range(daily_sales['Date'].min(), daily_sales['Date'].max())
daily_sales = daily_sales.set_index('Date').reindex(full_range).fillna(0).rename_axis('Date').reset_index()

# -----------------------------
# SAVE
# -----------------------------

daily_sales.to_csv(
    os.path.join(BASE_PATH, "daily_sales.csv"),
    index=False
)

print("DONE: daily_sales.csv created")