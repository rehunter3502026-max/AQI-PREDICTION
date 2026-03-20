import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from google.colab import files
uploaded = files.upload()

df = pd.read_csv("noida_aqi_with_pollutants.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

df['AQI_lag1'] = df['AQI'].shift(1)
df['AQI_lag2'] = df['AQI'].shift(2)

df = df.dropna()

X = df[['PM2.5', 'NO2', 'CO', 'O3', 'AQI_lag1', 'AQI_lag2']]
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)



r2 = r2_score(y_test, y_pred)
r = np.sqrt(r2)

print("R2 Score:", r2)
print("Correlation Coefficient (r):", r)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

results = pd.DataFrame({
    "Actual AQI": y_test.values,
    "Predicted AQI": y_pred
})

print(results.head(10))

plt.figure()
plt.plot(y_test.values, label="Actual AQI")
plt.plot(y_pred, label="Predicted AQI")
plt.legend()
plt.title("Actual vs Predicted AQI")
plt.xlabel("Samples")
plt.ylabel("AQI")
plt.show()
last_date = df['Date'].iloc[-1]

last_aqi = df['AQI'].iloc[-1]
last_aqi2 = df['AQI'].iloc[-2]

for i in range(5):
    next_date = last_date + pd.Timedelta(days=i+1)

    features = [[
        df['PM2.5'].iloc[-1],
        df['NO2'].iloc[-1],
        df['CO'].iloc[-1],
        df['O3'].iloc[-1],
        last_aqi,
        last_aqi2
    ]]
    
    pred = model.predict(features)[0]

    print(next_date.date(), "→ Predicted AQI:", round(pred,2))

    last_aqi2 = last_aqi
    last_aqi = pred

plt.figure()
sns.heatmap(df[['AQI','PM2.5','NO2','CO','O3']].corr(), annot=True)
plt.show()

plt.figure()
plt.scatter(df['PM2.5'], df['AQI'])
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.show()

plt.figure()
plt.scatter(df['NO2'], df['AQI'])
plt.xlabel("NO2")
plt.ylabel("AQI")
plt.show()

plt.figure()
plt.scatter(df['CO'], df['AQI'])
plt.xlabel("CO")
plt.ylabel("AQI")
plt.show()

plt.figure()
plt.scatter(df['O3'], df['AQI'])
plt.xlabel("O3")
plt.ylabel("AQI")
plt.show()

plt.figure()
plt.hist(df['AQI'], bins=30)
plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.show()
errors = y_test - y_pred

plt.figure()
plt.scatter(y_pred, errors)
plt.xlabel("Predicted AQI")
plt.ylabel("Error")
plt.title("Residual Plot")
plt.show()

future_vals = []
last_aqi = df['AQI'].iloc[-1]
last_aqi2 = df['AQI'].iloc[-2]

for i in range(5):
    features = [[
        df['PM2.5'].iloc[-1],
        df['NO2'].iloc[-1],
        df['CO'].iloc[-1],
        df['O3'].iloc[-1],
        last_aqi,
        last_aqi2
    ]]
    pred = model.predict(features)[0]
    future_vals.append(pred)

    last_aqi2 = last_aqi
    last_aqi = pred

plt.figure()
plt.plot(range(1,6), future_vals)
plt.title("Next 5 Days AQI Forecast")
plt.xlabel("Days Ahead")
plt.ylabel("AQI")
plt.show()

v
