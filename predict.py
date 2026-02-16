import joblib
import numpy as np

def predict_price(input_features):
    model = joblib.load("model.pkl")
    prediction = model.predict([input_features])
    return prediction[0]
