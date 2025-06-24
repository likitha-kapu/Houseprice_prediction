import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import joblib

# Load the dataset
df = pd.read_csv("house_data.csv")
df.dropna(inplace=True)

X = df[['X1 transaction date', 'X2 house age',
        'X3 distance to the nearest MRT station',
        'X4 number of convenience stores']]

y = df['Y house price of unit area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_score = r2_score(y_test, lr_model.predict(X_test))

# Train XGBoost
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_score = r2_score(y_test, xgb_model.predict(X_test))

print(f"Linear Regression R2 Score: {lr_score:.2f}")
print(f"XGBoost R2 Score: {xgb_score:.2f}")

# Save better model
best_model = xgb_model if xgb_score > lr_score else lr_model
joblib.dump(best_model, "model.pkl")
print("Model saved.")
