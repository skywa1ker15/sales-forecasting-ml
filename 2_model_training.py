import pandas as pd
import numpy as np
import os
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_PATH = r"C:\Users\rushi\OneDrive\Desktop\Future Interns\Machine_Learning"

df = pd.read_csv(os.path.join(BASE_PATH, "daily_sales.csv"))
df['Date'] = pd.to_datetime(df['Date'])

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.weekday
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

df['Trend'] = (df['Date'] - df['Date'].min()).dt.days

# LAGS
df['Lag_1'] = df['Sales'].shift(1)
df['Lag_2'] = df['Sales'].shift(2)
df['Lag_7'] = df['Sales'].shift(7)
df['Lag_14'] = df['Sales'].shift(14)
df['Lag_30'] = df['Sales'].shift(30)

# ROLLING FEATURES
df['Roll_7_mean'] = df['Sales'].rolling(7).mean()
df['Roll_7_std'] = df['Sales'].rolling(7).std()

df['Roll_30_mean'] = df['Sales'].rolling(30).mean()

df = df.dropna()

# -----------------------------
# TRAIN / TEST SPLIT (TIME SAFE)
# -----------------------------

split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

features = [
    'Day','Month','Weekday','WeekOfYear','Trend',
    'Lag_1','Lag_2','Lag_7','Lag_14','Lag_30',
    'Roll_7_mean','Roll_7_std','Roll_30_mean'
]

X_train = train[features]
y_train = train['Sales']

X_test = test[features]
y_test = test['Sales']

# -----------------------------
# MODEL (XGBOOST = BIG UPGRADE)
# -----------------------------

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

# -----------------------------
# METRICS
# -----------------------------

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\nMODEL PERFORMANCE")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# -----------------------------
# SAVE OUTPUTS
# -----------------------------

results = pd.DataFrame({
    "Date": test['Date'],
    "Actual": y_test.values,
    "Predicted": preds
})

results.to_csv(os.path.join(BASE_PATH, "predictions.csv"), index=False)

metrics = pd.DataFrame({
    "Metric": ["MAE","RMSE","R2"],
    "Value": [mae, rmse, r2]
})

metrics.to_csv(os.path.join(BASE_PATH, "model_metrics.csv"), index=False)

joblib.dump(model, os.path.join(BASE_PATH, "sales_model.pkl"))

df.to_csv(os.path.join(BASE_PATH, "processed_sales.csv"), index=False)

print("DONE: model trained + saved")