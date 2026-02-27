# -*- coding: utf-8 -*-
"""
Local Flask server for the QALY web app.
Serves the web/ static files and provides POST /api/calculate that runs the Python script and returns JSON.
"""
import os
import sys

# Use non-GUI backend so matplotlib does not create windows on a worker thread (macOS crashes otherwise)
import matplotlib
matplotlib.use("Agg")

from flask import Flask, request, jsonify, send_from_directory

# Project root (where server.py and ver_3 script live)
ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")
DATA_PATH = os.path.join(ROOT, "Cleaned_Dataset_QALY_Diet.csv")

# Map HTML form field names to Python initial_user_data keys
FORM_TO_PYTHON = {
    "gender": "Gender",
    "age_at_screening": "Age",
    "education_level": "Education Level",
    "unit_choice": "Unit Choice",
    "hdl_cholesterol": "HDL Cholesterol",
    "total_cholesterol": "Total Cholesterol",
    "insulin": "Insulin",
    "fasting_glucose": "Fasting Glucose",
    "bmi": "BMI",
    "waist_circumference_cm": "Waist Circumference (cm)",
    "high_blood_pressure_history": "High Blood Pressure History",
    "taking_bp_meds": "Taking BP Meds",
    "high_cholesterol_diagnosis": "High Cholesterol Diagnosis",
    "diabetes_diagnosis": "Diabetes Diagnosis",
    "taking_diabetic_pills": "Taking Diabetic Pills",
    "energy_kcal": "Energy Intake (kcal)",
    "dietary_fiber_gm": "Dietary Fiber (g)",
    "saturated_fats_gm": "Saturated Fat (g)",
    "vitamin_c": "Vitamin C (mg)",
    "alcohol_frequency": "Alcohol Frequency",
    "on_special_diet": "On Special Diet",
    "diet_type": "Diet Type",
    "interest_loss": "Interest Loss",
    "feeling_depressed": "Feeling Depressed",
    "feeling_tired": "Feeling Tired",
    "poor_appetite": "Poor Appetite",
    "problem_difficulty": "Problem Difficulty",
    "walk_bicycle": "Walk/Bicycle",
    "vigorous_activities": "Vigorous Activities",
    "days_vigorous_activities": "Days of Vigorous Activities",
    "moderate_activities": "Moderate Activities",
    "days_moderate_activities": "Days of Moderate Activities",
    "minutes_sedentary": "Minutes Sedentary",
}

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")


@app.after_request
def add_cors(resp):
    """Allow frontend from any origin (e.g. Live Server) to call the API."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(WEB_DIR, path)


@app.route("/api/calculate", methods=["POST", "OPTIONS"])
def calculate():
    if request.method == "OPTIONS":
        return "", 204
    """Accept JSON body with form fields; return utility, QALY, contributions, scenario_qaly_scores, warnings."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    body = request.get_json()
    if body is None:
        body = {}

    # Build initial_user_data for the Python script
    initial_user_data = {}
    for form_key, python_key in FORM_TO_PYTHON.items():
        if form_key in body:
            initial_user_data[python_key] = body[form_key]

    if not os.path.isfile(DATA_PATH):
        return jsonify({
            "error": f"Data file not found: {DATA_PATH}",
            "utility_score": None,
            "qaly": None,
            "remaining_years": None,
            "contributions": {},
            "scenario_qaly_scores": {},
            "warnings": [],
        }), 200

    # Import and run the calculation (script must be on path)
    sys.path.insert(0, ROOT)
    try:
        from qaly_calculator import qaly
    except ImportError as e:
        return jsonify({"error": f"Could not import calculator: {e}"}), 500

    try:
        result = qaly(initial_user_data=initial_user_data, data_path=DATA_PATH)
    except Exception as e:
        return jsonify({"error": str(e), "utility_score": None, "qaly": None, "remaining_years": None, "contributions": {}, "scenario_qaly_scores": {}, "warnings": []}), 200

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
