# House-Prediction-by-linear-regression
This project implements a House Price Prediction system using Linear Regression, a fundamental machine learning algorithm. The goal is to predict house prices based on features such as area, number of bedrooms, and bathrooms.  

 1. Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

 2. Load Dataset
# NOTE: Replace 'house_data.csv' with your dataset file name
# Sample dataset columns assumed:
# Area, Bedrooms, Bathrooms, Price

data = pd.read_csv('house_data.csv')

print("\nFirst 5 rows of dataset:")
print(data.head())

 3. Dataset Information
print("\nDataset Info:")
print(data.info())

 4. Check Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

 Drop missing values (if any)
data.dropna(inplace=True)

 5. Feature Selection
X = data[['Area', 'Bedrooms', 'Bathrooms']]
y = data['Price']

print("\nSelected Features (X):")
print(X.head())

print("\nTarget Variable (y):")
print(y.head())

 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

 7. Model Creation & Training
model = LinearRegression()
model.fit(X_train, y_train)

 8. Model Parameters
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

print("Intercept:", model.intercept_)

9. Prediction
y_pred = model.predict(X_test)

10. Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

 11. Visualization: Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel('Actual House Price')
plt.ylabel('Predicted House Price')
plt.title('Actual vs Predicted House Prices')
plt.show()

 12. Predict New House Price (Custom Input)
# Example: Area = 2500 sq.ft, Bedrooms = 3, Bathrooms = 2
new_house = np.array([[2500, 3, 2]])
predicted_price = model.predict(new_house)

print("\nPredicted Price for New House:")
print(f"Area: 2500, Bedrooms: 3, Bathrooms: 2")
print("Predicted House Price:", predicted_price[0])

