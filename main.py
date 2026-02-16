from data_loader import load_data
from preprocess import preprocess_data
from train_model import train_models
from evaluate import evaluate
import joblib

# Step 1: Load Data
df = load_data()
print("Data Loaded Successfully")

# Step 2: Preprocess
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
print("Data Preprocessed")

# Step 3: Train Models
lr_model, rf_model = train_models(X_train, y_train)
print("Models Trained")

# Step 4: Evaluate
lr_rmse = evaluate(lr_model, X_test, y_test, "Linear Regression")
rf_rmse = evaluate(rf_model, X_test, y_test, "Random Forest")

# Step 5: Select Best Model
best_model = rf_model if rf_rmse < lr_rmse else lr_model
print("\nBest Model Selected:", type(best_model).__name__)

# Step 6: Save Model
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model Saved as model.pkl")

# Step 7: Test Prediction
sample_input = X_test[0]
prediction = best_model.predict([sample_input])
print("\nSample Predicted House Price:", prediction[0])
