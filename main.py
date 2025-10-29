
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  Load Dataset
data = pd.read_csv("energydata_complete.csv")

#  View the first few rows
print("Dataset Loaded Successfully ")
print(data.head())

# Select useful features
features = ['T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T_out', 'RH_out', 'Press_mm_hg', 'Windspeed']
target = 'Appliances'

X = data[features]
y = data[target]

#  Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Train the Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

#  Make Predictions
y_pred = model.predict(X_test_scaled)

#  Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

#  Visualize Actual vs Predicted values
plt.figure(figsize=(8, 5))
plt.scatter(y_test[:100], y_pred[:100], color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Energy Consumption (Wh)")
plt.ylabel("Predicted Energy Consumption (Wh)")
plt.title("Actual vs Predicted Appliance Energy Consumption")
plt.show()
