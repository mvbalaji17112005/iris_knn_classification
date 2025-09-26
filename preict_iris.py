# predict_iris.py

import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load saved model + scaler
model = joblib.load("iris_knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load iris dataset (to get target names)
iris = load_iris()
target_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

def predict_iris(features):
    # features: [sepal length, sepal width, petal length, petal width]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return target_names[prediction]

# Example
sample = [5.1, 3.5, 1.4, 0.2]  # Iris-setosa
pred = predict_iris(sample)
print("Predicted Class:", pred)
