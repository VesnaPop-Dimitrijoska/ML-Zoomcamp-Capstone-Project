from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load the trained model
MODEL_PATH = "best_model.pkl"
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

@app.route('/forecast', methods=['POST'])
def forecast():
    """
    Forecast cumulative traffic per all devices.
    The API expects a JSON payload with the number of steps to forecast.
    Example JSON format:
    {
        "forecast_length": 30
    }
    """
    # Parse input JSON
    data = request.get_json()

    # Get the forecast length from the payload
    forecast_length = int(data["forecast_length"])

    # Making forecast (predictions)
    forecasts = model.forecast(steps=forecast_length)
    result = {"forecast": forecasts.tolist()}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
