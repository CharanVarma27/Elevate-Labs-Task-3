import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Step 1: Loading and Splitting the Data\n")
df = pd.read_csv('Housing.csv')

X = df.drop('price', axis=1) # Features
y = df['price'] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

print("\nStep 2: Training the Linear Regression Model\n")
model = LinearRegression()
model.fit(X_train, y_train)

print("\nStep 3: Evaluating Model Performance\n")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
print("\nStep 4: Visualizing Predictions vs. Actuals\n")
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, ci=None, scatter_kws={'alpha': 0.3})
plt.title('Actual vs. Predicted Housing Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

print("\nModel training and evaluation complete. The trained model is ready to make predictions.")
