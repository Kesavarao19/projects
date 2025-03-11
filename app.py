import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
import joblib

app = Flask(__name__)

# Define file paths
DATA_PATH = "dataset/weather_dataset.csv"
MODEL_PATH = "models/"
JSON_FILE = "json_data/predictions.json"

# Ensure necessary directories exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs("json_data", exist_ok=True)

# Step 1: Problem Formulation
# - Predicting Temperature and Weather Condition using Humidity and Wind Speed.

# Step 2: Data Collection and Preparation
def load_data():
    """Load dataset and preprocess it."""
    df = pd.read_csv(DATA_PATH)
    label_encoder = LabelEncoder()
    df["Weather Condition"] = label_encoder.fit_transform(df["Weather Condition"])
    joblib.dump(label_encoder, os.path.join(MODEL_PATH, "label_encoder.pkl"))
    return df, label_encoder

# Step 3: Exploratory Data Analysis (EDA)
def explore_data(df):
    """Basic EDA (checking null values and dataset info)."""
    print(df.info())
    print(df.describe())
    print(df["Weather Condition"].value_counts())

# Step 4 & 5: Feature Engineering and Model Training
def train_model():
    """Train models and save them."""
    df, label_encoder = load_data()
    
    X = df[["Humidity (%)", "Wind Speed (km/h)"]]
    y_temp = df["Temperature (°C)"]
    y_weather = df["Weather Condition"]

    temp_model = XGBRegressor()
    temp_model.fit(X, y_temp)

    weather_model = XGBClassifier(eval_metric='mlogloss')
    weather_model.fit(X, y_weather)

    joblib.dump(temp_model, os.path.join(MODEL_PATH, "temp_model.pkl"))
    joblib.dump(weather_model, os.path.join(MODEL_PATH, "weather_model.pkl"))

    return "Model trained successfully!"

# Load models if available; otherwise, train them
if os.path.exists(os.path.join(MODEL_PATH, "temp_model.pkl")):
    temp_model = joblib.load(os.path.join(MODEL_PATH, "temp_model.pkl"))
    weather_model = joblib.load(os.path.join(MODEL_PATH, "weather_model.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))
else:
    train_model()
    temp_model = joblib.load(os.path.join(MODEL_PATH, "temp_model.pkl"))
    weather_model = joblib.load(os.path.join(MODEL_PATH, "weather_model.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))

# Save prediction to JSON file
def save_prediction(prediction):
    with open(JSON_FILE, "w") as file:
        json.dump(prediction, file, indent=4)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_weather():
    try:
        data = request.get_json()
        print("Received Data:", data)  # Debugging

        # Validate input
        if not data or "humidity" not in data or "wind_speed" not in data:
            return jsonify({"error": "Invalid input. Provide 'humidity' and 'wind_speed'."}), 400

        humidity = float(data["humidity"])
        wind_speed = float(data["wind_speed"])

        input_data = np.array([[humidity, wind_speed]])
        
        # Make predictions
        predicted_temp = temp_model.predict(input_data)[0]
        predicted_weather = weather_model.predict(input_data)[0]
        predicted_weather_label = label_encoder.inverse_transform([predicted_weather])[0]

        response = {
            "Predicted Temperature (°C)": round(float(predicted_temp), 2),
            "Predicted Weather Condition": predicted_weather_label
        }

        # Save the prediction to a JSON file
        save_prediction(response)
        
        print("Prediction Response:", response)  # Debugging
        return jsonify(response)

    except Exception as e:
        print("Error:", str(e))  # Print error in terminal
        return jsonify({"error": str(e)}), 500  # Return JSON error response

# Fetch the latest saved prediction
@app.route("/latest_prediction", methods=["GET"])
def get_latest_prediction():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as file:
            return jsonify(json.load(file))
    return jsonify({"error": "No prediction found"}), 404

@app.route("/retrain", methods=["POST"])
def retrain():
    """Retrain models using updated dataset."""
    return train_model()

@app.route("/upload", methods=["POST"])
def upload():
    """Upload new dataset and retrain models."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file.save(DATA_PATH)

    return train_model()

if __name__ == "__main__":
    app.run(debug=True)
