from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import os
import uuid

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
MODEL_PATH = "fraud_detection_model.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# These are the 30 features used when training the model
FEATURES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20", "V21", "V22",
    "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


@app.route("/")
def home():
    return jsonify({
        "message": "AI-Powered Financial Fraud Detection API",
        "status": "running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    # Check whether a CSV file was uploaded
    if "file" not in request.files:
        return jsonify({"error": "Please upload a CSV file."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file."}), 400

    try:
        # Read uploaded CSV
        df = pd.read_csv(file)

        # Check required columns
        missing_columns = [
            column for column in FEATURES
            if column not in df.columns
        ]

        if missing_columns:
            return jsonify({
                "error": "The uploaded CSV is missing required columns.",
                "missing_columns": missing_columns
            }), 400

        # Select only the features used by the model
        X = df[FEATURES].copy()

        # Convert values to numbers
        X = X.apply(pd.to_numeric, errors="coerce")

        # Check for invalid/missing values
        if X.isnull().any().any():
            return jsonify({
                "error": "The CSV contains missing or non-numeric values in the required feature columns."
            }), 400

        # Apply the same scaler used during training
        X_scaled = scaler.transform(X)

        # Make predictions
        predictions = model.predict(X_scaled)

        # Add predictions to the original data
        result_df = df.copy()
        result_df["Fraud_Prediction"] = predictions
        result_df["Prediction_Label"] = result_df["Fraud_Prediction"].map({
            0: "Normal",
            1: "Fraud"
        })

        # Summary
        total_transactions = len(result_df)
        fraud_count = int((predictions == 1).sum())
        normal_count = int((predictions == 0).sum())

        # Save result file
        output_filename = f"fraud_results_{uuid.uuid4().hex}.csv"

        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
        os.makedirs(temp_dir, exist_ok=True)

        output_path = os.path.join(temp_dir, output_filename)
        result_df.to_csv(output_path, index=False)

        return jsonify({
            "message": "Fraud detection completed successfully.",
            "total_transactions": total_transactions,
            "fraud_transactions": fraud_count,
            "normal_transactions": normal_count,
            "result_file": output_filename
        })
    except Exception as e:
        return jsonify({
            "error": f"Could not process the file: {str(e)}"
        }), 500


@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(_file_)),
        "tmp",
        filename
    )

    if not os.path.exists(file_path):
        return jsonify({"error": "Result file not found."}), 404

    return send_file(
        file_path,
        as_attachment=True,
        download_name="fraud_detection_results.csv"
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)