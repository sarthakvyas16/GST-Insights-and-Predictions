import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 3, 5, 7, 11])  # Target values

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, 'linear_regression_model.pkl')
