# -*- coding: utf-8 -*-
"""QALY/NHANES calculator — web and structure-improvement version.

This module is the main calculation engine for the local web app (server.py).
All structural changes and improvements are made here. The original Colab
script is preserved as ver_3_qaly_nhanes_calculation.py.
"""

import pandas as pd

import numpy as np
import warnings
import random
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Config: utility bounds, life expectancy, contribution weights (for tuning)
# -----------------------------------------------------------------------------
CONFIG = {
    "utility": {
        "min_score": 0.1,
        "max_score": 1.0,
        "benchmark_age": 80,
    },
    "weights_negative": {
        "hdl": 1.4,
        "total_cholesterol": 1.3,
        "glucose": 1.3,
        "bmi": 1.2,
        "waist": 1.2,
        "bp_untreated": 1.5,
        "bp_treated": 1.2,
        "diabetes_untreated": 1.4,
        "diabetes_treated": 1.1,
        "sedentary": 1.05,
    },
    "education_scaling_factor": 0.05,
    "scenario_seed": 42,
}


def sigmoid(x):
    """Sigmoid for normalizing contributions to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def age_grouping(age):
    """Map age to NHANES age group (19, 31, 51)."""
    if age <= 30:
        return 19
    if age <= 50:
        return 31
    return 51


def qaly(initial_user_data=None, data_path=None):
    """Run QALY calculation. If initial_user_data is provided, use it and return a result dict (no plots)."""

    # Default values (Colab @param style)
    gender = "Female"
    age_at_screening = 32
    education_level = "College graduate or above"
    unit_choice = "mmol"
    hdl_cholesterol = 1.81
    total_cholesterol = 4.91
    insulin = 3.3
    fasting_glucose = 2.8
    alcohol_frequency = 20
    high_blood_pressure_history = "No"
    taking_bp_meds = "No"
    high_cholesterol_diagnosis = "No"
    bmi = 31
    waist_circumference_cm = 65
    diabetes_diagnosis = "No"
    taking_diabetic_pills = "No"
    energy_kcal = 1800
    dietary_fiber_gm = 60
    saturated_fats_gm = 30
    vitamin_c = 80
    interest_loss = "Not at all"
    feeling_depressed = "Several days"
    feeling_tired = "More than half the days"
    poor_appetite = "Nearly every day"
    problem_difficulty = "Not difficult at all"
    walk_bicycle = "Yes"
    vigorous_activities = "Yes"
    days_vigorous_activities = 6
    moderate_activities = "Yes"
    days_moderate_activities = 4
    minutes_sedentary = 400
    on_special_diet = "Yes"
    diet_type = "High protein diet"

    # Override from API/form when provided
    if initial_user_data is not None:
        gender = initial_user_data.get("Gender", gender)
        age_at_screening = initial_user_data.get("Age", age_at_screening)
        education_level = initial_user_data.get("Education Level", education_level)
        unit_choice = initial_user_data.get("Unit Choice", unit_choice)
        hdl_cholesterol = initial_user_data.get("HDL Cholesterol", hdl_cholesterol)
        total_cholesterol = initial_user_data.get("Total Cholesterol", total_cholesterol)
        insulin = initial_user_data.get("Insulin", insulin)
        fasting_glucose = initial_user_data.get("Fasting Glucose", fasting_glucose)
        alcohol_frequency = initial_user_data.get("Alcohol Frequency", alcohol_frequency)
        high_blood_pressure_history = initial_user_data.get("High Blood Pressure History", high_blood_pressure_history)
        taking_bp_meds = initial_user_data.get("Taking BP Meds", taking_bp_meds)
        high_cholesterol_diagnosis = initial_user_data.get("High Cholesterol Diagnosis", high_cholesterol_diagnosis)
        bmi = initial_user_data.get("BMI", bmi)
        waist_circumference_cm = initial_user_data.get("Waist Circumference (cm)", waist_circumference_cm)
        diabetes_diagnosis = initial_user_data.get("Diabetes Diagnosis", diabetes_diagnosis)
        taking_diabetic_pills = initial_user_data.get("Taking Diabetic Pills", taking_diabetic_pills)
        energy_kcal = initial_user_data.get("Energy Intake (kcal)", energy_kcal)
        dietary_fiber_gm = initial_user_data.get("Dietary Fiber (g)", dietary_fiber_gm)
        saturated_fats_gm = initial_user_data.get("Saturated Fat (g)", saturated_fats_gm)
        vitamin_c = initial_user_data.get("Vitamin C (mg)", vitamin_c)
        interest_loss = initial_user_data.get("Interest Loss", interest_loss)
        feeling_depressed = initial_user_data.get("Feeling Depressed", feeling_depressed)
        feeling_tired = initial_user_data.get("Feeling Tired", feeling_tired)
        poor_appetite = initial_user_data.get("Poor Appetite", poor_appetite)
        problem_difficulty = initial_user_data.get("Problem Difficulty", problem_difficulty)
        walk_bicycle = initial_user_data.get("Walk/Bicycle", walk_bicycle)
        vigorous_activities = initial_user_data.get("Vigorous Activities", vigorous_activities)
        days_vigorous_activities = initial_user_data.get("Days of Vigorous Activities", days_vigorous_activities)
        moderate_activities = initial_user_data.get("Moderate Activities", moderate_activities)
        days_moderate_activities = initial_user_data.get("Days of Moderate Activities", days_moderate_activities)
        minutes_sedentary = initial_user_data.get("Minutes Sedentary", minutes_sedentary)
        on_special_diet = initial_user_data.get("On Special Diet", on_special_diet)
        diet_type = initial_user_data.get("Diet Type", diet_type)
        # Coerce numeric types in case they came as strings
        if isinstance(age_at_screening, (str, type(None))) and age_at_screening is not None:
            try: age_at_screening = int(float(age_at_screening))
            except (ValueError, TypeError): age_at_screening = 32
        if isinstance(bmi, (str, type(None))) and bmi is not None:
            try: bmi = float(bmi)
            except (ValueError, TypeError): bmi = 31
        if isinstance(waist_circumference_cm, (str, type(None))) and waist_circumference_cm is not None:
            try: waist_circumference_cm = float(waist_circumference_cm)
            except (ValueError, TypeError): waist_circumference_cm = 65
        for key in ("HDL Cholesterol", "Total Cholesterol", "Insulin", "Fasting Glucose", "Energy Intake (kcal)", "Dietary Fiber (g)", "Saturated Fat (g)", "Vitamin C (mg)", "Alcohol Frequency", "Days of Vigorous Activities", "Days of Moderate Activities", "Minutes Sedentary"):
            v = initial_user_data.get(key)
            if v is not None and isinstance(v, str):
                try: initial_user_data[key] = float(v)
                except (ValueError, TypeError): pass
        hdl_cholesterol = initial_user_data.get("HDL Cholesterol", hdl_cholesterol)
        total_cholesterol = initial_user_data.get("Total Cholesterol", total_cholesterol)
        insulin = initial_user_data.get("Insulin", insulin)
        fasting_glucose = initial_user_data.get("Fasting Glucose", fasting_glucose)
        energy_kcal = initial_user_data.get("Energy Intake (kcal)", energy_kcal)
        dietary_fiber_gm = initial_user_data.get("Dietary Fiber (g)", dietary_fiber_gm)
        saturated_fats_gm = initial_user_data.get("Saturated Fat (g)", saturated_fats_gm)
        vitamin_c = initial_user_data.get("Vitamin C (mg)", vitamin_c)
        alcohol_frequency = initial_user_data.get("Alcohol Frequency", alcohol_frequency)
        days_vigorous_activities = initial_user_data.get("Days of Vigorous Activities", days_vigorous_activities)
        days_moderate_activities = initial_user_data.get("Days of Moderate Activities", days_moderate_activities)
        minutes_sedentary = initial_user_data.get("Minutes Sedentary", minutes_sedentary)

    api_mode = initial_user_data is not None

    # Collecting all initial user data directly from widgets, following your provided snippet



    initial_user_data = {
        "Gender": gender,
        "Age": age_at_screening,
        "Education Level": education_level,
        "Unit Choice": unit_choice,
        "HDL Cholesterol": hdl_cholesterol,
        "Total Cholesterol": total_cholesterol,
        "Insulin": insulin,
        "Fasting Glucose": fasting_glucose,
        "Alcohol Frequency": alcohol_frequency,
        "High Blood Pressure History": high_blood_pressure_history,
        "Taking BP Meds": taking_bp_meds,
        "High Cholesterol Diagnosis": high_cholesterol_diagnosis,
        "BMI": bmi,
        "Waist Circumference (cm)": waist_circumference_cm,
        "Diabetes Diagnosis": diabetes_diagnosis,
        "Taking Diabetic Pills": taking_diabetic_pills,
        "Energy Intake (kcal)": energy_kcal,
        "Dietary Fiber (g)": dietary_fiber_gm,
        "Saturated Fat (g)": saturated_fats_gm,
        "Vitamin C (mg)": vitamin_c,
        "Interest Loss": interest_loss,
        "Feeling Depressed": feeling_depressed,
        "Feeling Tired": feeling_tired,
        "Poor Appetite": poor_appetite,
        "Problem Difficulty": problem_difficulty,
        "Walk/Bicycle": walk_bicycle,
        "Vigorous Activities": vigorous_activities,
        "Days of Vigorous Activities": days_vigorous_activities,
        "Moderate Activities": moderate_activities,
        "Days of Moderate Activities": days_moderate_activities,
        "Minutes Sedentary": minutes_sedentary,
        "On Special Diet": on_special_diet,
        "Diet Type": diet_type
    }

    # Output the collected data for user verification
    initial_user_data

    # Define conversion factors for mg to mmol
    mg_to_mmol_factors = {
        "HDL Cholesterol": 0.0259,
        "Total Cholesterol": 0.0259,
        "Insulin": 0.00694,
        "Fasting Glucose": 0.0555
    }

    # Assume the unit_choice is provided by the user widget (mg or mmol)
    unit_choice = "mmol"  # This would be dynamic based on user input

    # Define realistic ranges or adjustments for each metric
    numeric_boundaries = {
        "HDL Cholesterol": {"healthy": (50, 70), "unhealthy": (20, 40)},
        "Total Cholesterol": {"healthy": (150, 200), "unhealthy": (200, 250)},
        "Insulin": {"healthy": (2, 25), "unhealthy": (50, 100)},
        "Fasting Glucose": {"healthy": (70, 99), "unhealthy": (126, 200)},
        "BMI": {"healthy": (18.5, 24.9), "unhealthy": (30, 40)},
        "Alcohol Frequency": {"healthy": (0, 4), "unhealthy": (5, 30)},
        "Waist Circumference (cm)": {"healthy": (80, 100), "unhealthy": (110, 150)},
        "Energy Intake (kcal)": {"healthy": (1800, 2500), "unhealthy": ((1000, 1500), (3000, 5000))},
        "Dietary Fiber (g)": {"healthy": (22, 34), "unhealthy": (10, 21)},
        "Saturated Fat (g)": {"healthy": (5, 25), "unhealthy": (26, 50)},
        "Vitamin C (mg)": {"healthy": (75, 90), "unhealthy": ((30, 74),(91, 150))},
        "Days of Vigorous Activities": {"healthy": (1, 7), "unhealthy": (0, 0)},
        "Minutes Sedentary": {"healthy": (60, 240), "unhealthy": (360, 540)},
        "Days of Moderate Activities": {"healthy": (1, 7), "unhealthy": (0, 0)},
    }

    categorical_adjustments = {
        "High Blood Pressure History": {"healthy": "No", "unhealthy": "Yes"},
        "High Cholesterol Diagnosis": {"healthy": "No", "unhealthy": "Yes"},
        "Diabetes Diagnosis": {"healthy": "No", "unhealthy": "Yes"},
        "Taking BP Meds": {"healthy": "No", "unhealthy": "Yes"},
        "Taking Diabetic Pills": {"healthy": "No", "unhealthy": "Yes"},
        "Interest Loss": {"healthy": ("Not at all"), "unhealthy": ("Several days", "More than half the days", "Nearly every day")},
        "Feeling Depressed": {"healthy": ("Not at all"), "unhealthy": ("Several days", "More than half the days", "Nearly every day")},
        "Feeling Tired": {"healthy": ("Not at all"), "unhealthy": ("Several days", "More than half the days", "Nearly every day")},
        "Poor Appetite": {"healthy": ("Not at all"), "unhealthy": ("Several days", "More than half the days", "Nearly every day")},
        "Problem Difficulty": {"healthy": ("Not difficult at all"), "unhealthy": ("Somewhat difficult", "Very difficult", "Extremely difficult")},
        "Walk/Bicycle": {"healthy": "Yes", "unhealthy": "No"},
        "Vigorous Activities": {"healthy": "Yes", "unhealthy": "No"},
        "Moderate Activities": {"healthy": "Yes", "unhealthy": "No"},
    }

    # Create scenarios based on initial user data (reproducible with fixed seed)
    random.seed(CONFIG["scenario_seed"])
    scenarios = {
        "best_case": initial_user_data.copy(),
        "worst_case": initial_user_data.copy(),
        "middle_case_1": initial_user_data.copy(),
        "middle_case_2": initial_user_data.copy(),
    }

    # Adjust each metric for each scenario
    for key, value in initial_user_data.items():
        if key in numeric_boundaries:  # For numerical metrics
            healthy_range = numeric_boundaries[key]["healthy"]
            unhealthy_range = numeric_boundaries[key]["unhealthy"]

            # Adjust the ranges based on the unit choice
            if unit_choice == "mmol":
                if key in mg_to_mmol_factors:
                    factor = mg_to_mmol_factors[key]
                    healthy_range = tuple([x * factor for x in healthy_range])
                    unhealthy_range = tuple([x * factor for x in unhealthy_range])

            # Handle potential nested ranges for unhealthy values
            if isinstance(unhealthy_range[0], tuple):
                selected_unhealthy_range = random.choice(unhealthy_range)
            else:
                selected_unhealthy_range = unhealthy_range

            # Adjust values for each scenario
            if isinstance(value, int):  # Integer metrics (e.g., days)
                scenarios["best_case"][key] = random.randint(int(healthy_range[0]), int(healthy_range[1]))
                scenarios["worst_case"][key] = random.randint(int(selected_unhealthy_range[0]), int(selected_unhealthy_range[1]))
                scenarios["middle_case_1"][key] = round((healthy_range[0] + selected_unhealthy_range[0]) / 2)
                scenarios["middle_case_2"][key] = round((healthy_range[1] + selected_unhealthy_range[1]) / 2)
            else:  # Float metrics
                scenarios["best_case"][key] = random.uniform(*healthy_range)
                scenarios["worst_case"][key] = random.uniform(*selected_unhealthy_range)
                scenarios["middle_case_1"][key] = (healthy_range[0] + selected_unhealthy_range[0]) / 2
                scenarios["middle_case_2"][key] = (healthy_range[1] + selected_unhealthy_range[1]) / 2

        elif key in categorical_adjustments:  # For categorical metrics
            scenarios["best_case"][key] = categorical_adjustments[key]["healthy"]
            if isinstance(categorical_adjustments[key]["unhealthy"], tuple):
                scenarios["worst_case"][key] = random.choice(categorical_adjustments[key]["unhealthy"])
            else:
                scenarios["worst_case"][key] = categorical_adjustments[key]["unhealthy"]
            scenarios["middle_case_1"][key] = value  # Keep original for middle scenario
            scenarios["middle_case_2"][key] = value  # Keep original for middle scenario

    # Output the generated scenarios
    print("Generated Scenarios:")
    for scenario_name, scenario_data in scenarios.items():
        print(f"\n{scenario_name.upper()}:")
        for key, value in scenario_data.items():
            print(f"{key}: {value}")

    file_path = data_path if data_path is not None else '/content/drive/MyDrive/Protocol Health/QALY/NHANES data/Merged data/Cleaned_Dataset_QALY_Diet.csv'

    # Specify columns to ignore
    columns_to_ignore = ["DR1TALCO", "DRQSDT11", "DRQSDT12"]

    # Normalize missing-value strings (used in read_csv and again for any that appear after load)
    na_strings = ["no info", "No info", "NO INFO", "-", "NA", "NaN", "nan", ""]
    nhanes_data = pd.read_csv(
        file_path,
        encoding="utf-8",
        na_values=na_strings,
        keep_default_na=True,
    ).drop(columns=columns_to_ignore, errors="ignore")
    nhanes_data.replace(na_strings, pd.NA, inplace=True)

    # Define the columns that should be numeric (include PAQ655, PAQ670, PAD680 for benchmarks/consistency)
    numeric_columns = [
        "LBDHDD", "LBDHDDSI", "LBXTC", "LBDTCSI", "LBXIN", "LBDINSI",
        "LBXGLU", "LBDGLUSI", "ALQ170", "BMXBMI", "DMDEDUC2", "DR1TKCAL",
        "DR1TFIBE", "DR1TSFAT", "BMXWAIST", "DR1TVC",
        "PAQ655", "PAQ670", "PAD680",
    ]

    # Convert only the specified columns to numeric, forcing errors to NaN (skip if column missing)
    for col in numeric_columns:
        if col in nhanes_data.columns:
            nhanes_data[col] = pd.to_numeric(nhanes_data[col], errors="coerce")

    # Now, re-run the descriptive statistics on all columns
    stats_summary = nhanes_data.describe(include='all').transpose()
    stats_summary['missing_values'] = nhanes_data.isnull().sum()

    # Display the updated statistics summary
    stats_summary

    # Create a copy of the dataset to work on
    nhanes_data_copy = nhanes_data.copy()

    # Define the numerical metrics that need benchmarks
    benchmark_metrics = [
        "LBDHDD", "LBDHDDSI", "LBXTC", "LBDTCSI", "LBXIN", "LBDINSI",
        "LBXGLU", "LBDGLUSI", "ALQ170", "BMXBMI", "RIDAGEYR", "DR1TKCAL",
        "DR1TFIBE", "DR1TSFAT", "PAQ655", "PAQ670", "PAD680",
        "BMXWAIST", "DR1TVC"
    ]

    categorical_mappings = {
        "BPQ020": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know", None: "Missing"},
        "BPQ050A": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know", None: "Missing"},
        "BPQ080": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know", None: "Missing"},
        "DIQ010": {1: "Yes", 2: "No", 3: "Borderline", 7: "Refused", 9: "Don't know", None: "Missing"},
        "DIQ070": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know", None: "Missing"},
        "DPQ010": {0: "Not at all", 1: "Several days", 2: "More than half the days",
                   3: "Nearly every day", 7: "Refused", 9: "Don't know", None: "Missing"},
        "DPQ020": {0: "Not at all", 1: "Several days", 2: "More than half the days",
                   3: "Nearly every day", 7: "Refused", 9: "Don't know", None: "Missing"},
        "DPQ040": {0: "Not at all", 1: "Several days", 2: "More than half the days",
                   3: "Nearly every day", 7: "Refused", 9: "Don't know", None: "Missing"},
        "DPQ050": {0: "Not at all", 1: "Several days", 2: "More than half the days",
                   3: "Nearly every day", 7: "Refused", 9: "Don't know", None: "Missing"},
        "DPQ100": {0: "Not at all difficult", 1: "Somewhat difficult", 2: "Very difficult",
                   3: "Extremely difficult", 7: "Refused", 9: "Don't know", None: "Missing"},
        "PAQ635": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know", None: "Missing"},
        "PAQ650": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know", None: "Missing"},
        "PAQ655": {1: "1 Day", 2: "2 Days", 3: "3 Days", 4: "4 Days", 5: "5 Days", 6: "6 Days", 7: "7 Days",
                   77: "Refused", 99: "Don't know", None: "Missing"},
        "PAQ665": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know", None: "Missing"},
        "PAQ670": {1: "1 Day", 2: "2 Days", 3: "3 Days", 4: "4 Days", 5: "5 Days", 6: "6 Days", 7: "7 Days",
                   77: "Refused", 99: "Don't know", None: "Missing"},
        "PAD680": None,  # Numerical ranges, no mapping needed
        "DRQSDIET": {1: "Yes", 2: "No", 9: "Don't know", None: "Missing"},
        "DRQSDT1": {1: "Yes", 2: "No", 9: "Don't know", None: "Missing"},  # diet type columns use same Yes/No coding
        "DMDEDUC2":{1: "Less than 9th grade	", 2: "9-11th grade (Includes 12th grade with no diploma)", 3: "High school graduate/GED or equivalent	", 4: "Some college or AA degree	", 5:"College graduate or above	", 7: "Refused", 9: "Don't know", None: "Missing"},
    }

    # Reset categorical columns explicitly
    for col in categorical_mappings.keys():
        if col in nhanes_data.columns:
            nhanes_data_copy[col] = nhanes_data[col]

    # Convert numerical columns
    for metric in benchmark_metrics:
        nhanes_data_copy[metric] = pd.to_numeric(nhanes_data_copy[metric], errors='coerce')

    benchmarks = {}
    for metric in benchmark_metrics:
        benchmarks[metric] = (
            float(nhanes_data_copy[metric].mean()),  # Convert mean to float
            float(nhanes_data_copy[metric].std())    # Convert std to float
        )

    # Process categorical benchmarks
    categorical_benchmarks = {}
    for col, mapping in categorical_mappings.items():
        if col in nhanes_data_copy.columns:
            # Convert to numeric if necessary
            try:
                nhanes_data_copy[col] = pd.to_numeric(nhanes_data_copy[col], errors="coerce")
            except Exception as e:
                print(f"Error converting {col} to numeric: {e}")

            # Apply mapping
            nhanes_data_copy[col] = nhanes_data_copy[col].replace(mapping).fillna("Missing")

            # Exclude "Missing" values for benchmarks
            filtered_col = nhanes_data_copy[col][nhanes_data_copy[col] != "Missing"]
            if filtered_col.empty:
                categorical_benchmarks[col] = {
                    "top": "Missing",
                    "top_percentage": 100.0,
                    "second_top": None,
                    "second_percentage": None,
                    "all_categories": {}  # No valid categories
                }
            else:
                value_counts = filtered_col.value_counts(normalize=True)
                # Store all categories and their percentages
                all_categories = {value: round(percentage * 100, 2) for value, percentage in value_counts.items()}

                categorical_benchmarks[col] = {
                    "top": value_counts.idxmax(),
                    "top_percentage": round(value_counts.max() * 100, 2),
                    "all_categories": all_categories  # Store all categories and percentages
                }

    # Mapping for unit-specific metrics
    metric_si_mapping = {
        "LBDHDD": "LBDHDDSI",
        "LBXTC": "LBDTCSI",
        "LBXIN": "LBDINSI",
        "LBXGLU": "LBDGLUSI"}

    # Vectorized application with a lambda function
    nhanes_data['metric_value'] = nhanes_data.apply(
        lambda row: row.get(metric) if unit_choice == "mmol" else row.get(metric_si_mapping.get(metric, f"{metric}SI")),
        axis=1
    )

    nhanes_data['metric_benchmark'] = nhanes_data.apply(
        lambda row: benchmarks.get(metric_si_mapping.get(metric, f"{metric}SI") if unit_choice != "mmol" else metric, (None, None)),
        axis=1
    )

    # Handle missing values separately
    nhanes_data['metric_value'] = nhanes_data['metric_value'].apply(
        lambda x: None if x in ["-", "Don't know"] or pd.isna(x) else x
    )

    # Dynamic thresholds for HDL based on NHANES benchmarks
    hdl_mean_mg, hdl_std_mg = benchmarks["LBDHDD"]  # Mean and SD for HDL (mg/dL)
    hdl_mean_mmol, hdl_std_mmol = benchmarks["LBDHDDSI"]  # Mean and SD for HDL (mmol/L)

    # Define dynamic thresholds using NHANES data
    baseline_hdl = {
        'mg': {
            'optimal': hdl_mean_mg,  # Mean HDL in mg/dL
            'penalty_men': hdl_mean_mg - 1 * hdl_std_mg,  # One SD below the mean
            'penalty_women': hdl_mean_mg - 0.8 * hdl_std_mg,  # Adjusted for gender
            'excessive': hdl_mean_mg + 2 * hdl_std_mg  # Two SDs above the mean
        },
        'mmol': {
            'optimal': hdl_mean_mmol,  # Mean HDL in mmol/L
            'penalty_men': hdl_mean_mmol - 1 * hdl_std_mmol,  # One SD below the mean
            'penalty_women': hdl_mean_mmol - 0.8 * hdl_std_mmol,  # Adjusted for gender
            'excessive': hdl_mean_mmol + 2 * hdl_std_mmol  # Two SDs above the mean
        }
    }

    # Sigmoid parameters
    k_low = 0.2
    k_high = 0.05

    # Check unit choice
    if unit_choice == "Don't know":
        # Neutral score for unspecified unit
        hdl_contribution = 0
    else:
        # Determine thresholds based on unit choice
        thresholds = baseline_hdl[unit_choice]

        # Gender-specific penalty thresholds
        if gender == "Male":
            penalty_threshold = thresholds['penalty_men']
        elif gender == "Female":
            penalty_threshold = thresholds['penalty_women']
        else:
            penalty_threshold = thresholds['penalty_men']  # Default to male thresholds if gender unknown

        optimal = thresholds['optimal']
        excessive = thresholds['excessive']

        # Unified HDL value
        hdl = hdl_cholesterol

        # Calculate utility contribution
        if hdl is None or hdl == "Don't know":
            hdl_score = 0  # Neutral if HDL value is missing
        elif hdl < penalty_threshold:
            # Severe penalty for very low HDL
            hdl_score = -(1 - (1 / (1 + np.exp(-k_low * (hdl - (2 * penalty_threshold))))))
        elif hdl <= optimal:
            # Bonus for values moving toward optimal
            hdl_score = 1 - (1 / (1 + np.exp(-k_low * (hdl - optimal))))
        elif hdl > excessive:
            # Gradual penalty for excessive HDL
            hdl_score = -(1 - (1 / (1 + np.exp(-k_high * (hdl - excessive)))))
        else:
            # Bonus for optimal range
            hdl_score = 1 - (1 / (1 + np.exp(-k_low * (hdl - optimal))))

        # Scale the contribution to the utility score
        hdl_contribution = round(hdl_score * 0.25, 4)

    # Scenario HDL Contributions
    scenario_hdl_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        hdl = scenario_data["HDL Cholesterol"]  # Get HDL value for the current scenario
        if unit_choice == "Don't know":
            hdl_score = 0  # Neutral if HDL value is missing or unit unknown
        elif hdl is None or hdl == "Don't know":
            hdl_score = 0  # Neutral if HDL value is missing
        elif hdl < penalty_threshold:
            hdl_score = -(1 - (1 / (1 + np.exp(-k_low * (hdl - (2 * penalty_threshold))))))
        elif hdl <= optimal:
            hdl_score = 1 - (1 / (1 + np.exp(-k_low * (hdl - optimal))))
        elif hdl > excessive:
            hdl_score = -(1 - (1 / (1 + np.exp(-k_high * (hdl - excessive)))))
        else:
            hdl_score = 1 - (1 / (1 + np.exp(-k_low * (hdl - optimal))))

        # Scale the contribution to the utility score
        hdl_contribution_scenario = round(hdl_score * 0.25, 4)
        scenario_hdl_contributions[scenario_name] = hdl_contribution_scenario

    # Baseline thresholds for Total Cholesterol (clinical cutpoints)
    baseline_tc = {
        'mg': {
            'acceptable': 199,
            'borderline_low': 200,
            'borderline_high': 239,
            'high': 240
        },
        'mmol': {
            'acceptable': 5.14,
            'borderline_low': 5.17,
            'borderline_high': 6.18,
            'high': 6.2
        }
    }

    # Sigmoid parameters
    k_moderate = 0.1  # Steepness for borderline high
    k_sharp = 0.2     # Steepness for very high cholesterol

    # Adjusted thresholds using NHANES benchmarks
    if unit_choice == "mg":
        nhanes_mean, nhanes_std = benchmarks.get("LBXTC", (None, None))
    elif unit_choice == "mmol":
        nhanes_mean, nhanes_std = benchmarks.get("LBDTCSI", (None, None))
    else:
        tc_contribution = 0  # Neutral score
        nhanes_mean, nhanes_std = None, None

    if nhanes_mean is not None and nhanes_std is not None:
        acceptable = nhanes_mean - nhanes_std
        borderline_low = nhanes_mean - 0.5 * nhanes_std
        borderline_high = nhanes_mean + 0.5 * nhanes_std
        high = nhanes_mean + nhanes_std
    else:
        thresholds = baseline_tc[unit_choice]
        acceptable = thresholds['acceptable']
        borderline_low = thresholds['borderline_low']
        borderline_high = thresholds['borderline_high']
        high = thresholds['high']

    sigmoid_center = high  # Center of sigmoid for high cholesterol

    if unit_choice != "Don't know":
        # Get the total cholesterol value
        tc = total_cholesterol

        # Calculate utility contribution
        if tc is None or tc == "Don't know":
            tc_score = 0  # Neutral if Total Cholesterol value is missing
        elif tc < acceptable:
            # Bonus for acceptable values
            tc_score = 1 - (1 / (1 + np.exp(-k_moderate * (tc - acceptable))))
        elif borderline_low <= tc <= borderline_high:
            # Moderate penalty for borderline high cholesterol
            tc_score = -(1 - (1 / (1 + np.exp(-k_moderate * (tc - sigmoid_center)))))
        elif tc >= high:
            # Sharp penalty for very high cholesterol
            tc_score = -(1 - (1 / (1 + np.exp(-k_sharp * (tc - sigmoid_center)))))
        else:
            tc_score = 0  # Neutral for undefined ranges

        # Scale the contribution to the utility score
        tc_contribution = round(tc_score * 0.4, 4)  # Adjust weight as needed
    else:
        tc_contribution = 0  # Neutral for 'Don't know'

    # Scenario TC Contributions
    scenario_tc_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Get Total Cholesterol value for the current scenario
        tc_scenario = scenario_data["Total Cholesterol"]

        # Correct Thresholds Calculation
        if nhanes_mean is not None and nhanes_std is not None:
            # Use NHANES benchmarks combined with baseline thresholds
            thresholds = baseline_tc[unit_choice]
            acceptable = max(thresholds['acceptable'], nhanes_mean - nhanes_std)
            borderline_high = min(thresholds['borderline_high'], nhanes_mean + 0.5 * nhanes_std)
            high = max(thresholds['high'], nhanes_mean + nhanes_std)
        else:
            # Use baseline thresholds if NHANES benchmarks are unavailable
            thresholds = baseline_tc[unit_choice]
            acceptable = thresholds['acceptable']
            borderline_high = thresholds['borderline_high']
            high = thresholds['high']

        sigmoid_center = high  # Center of sigmoid for high cholesterol

        # Calculate utility contribution using the main calculation logic
        alpha = 0.01  # Scaling factor for linear adjustment (tune as needed)

        if tc_scenario is None or tc_scenario == "Don't know":
            scenario_tc_score = 0  # Neutral if Total Cholesterol value is missing
        elif tc_scenario < acceptable:
            # Bonus for acceptable values
            scenario_tc_score = 1 - (1 / (1 + np.exp(-k_moderate * (acceptable - tc_scenario))))
        elif borderline_low <= tc_scenario <= borderline_high:
            # Moderate penalty for borderline cholesterol
            scenario_tc_score = -(1 - (1 / (1 + np.exp(-k_moderate * (tc_scenario - borderline_high)))))
        elif borderline_high < tc_scenario < high:
            # Mild penalty for values between borderline_high and high
            scenario_tc_score = -(1 - (1 / (1 + np.exp(-k_moderate * (tc_scenario - high))))) * (1 + alpha * (tc_scenario / high))
        elif tc_scenario >= high:
            # Sharp penalty for very high cholesterol
            scenario_tc_score = -(1 - (1 / (1 + np.exp(-k_sharp * (tc_scenario - high))))) * (1 + alpha * (tc_scenario / high))
        else:
            scenario_tc_score = 0  # Neutral for undefined ranges

        # Scale the contribution to the utility score
        scenario_tc_contribution = round(scenario_tc_score * 0.4, 4)  # Adjust weight as needed
        if scenario_tc_contribution == -0.0:
            scenario_tc_contribution = 0.0  # Avoid negative zero

      # Store contribution
        scenario_tc_contributions[scenario_name] = scenario_tc_contribution

    # Fetch NHANES benchmarks for Fasting Glucose
    nhanes_mean, nhanes_std = benchmarks.get("LBXGLU" if unit_choice == "mg" else "LBDGLUSI", (None, None))

    # Sigmoid parameters
    k_moderate = 0.2  # Steepness for borderline high
    k_sharp = 0.4     # Steepness for very high glucose

    if nhanes_mean is not None and nhanes_std is not None:
        # Calculate NHANES-based thresholds
        optimal_threshold = nhanes_mean
        borderline_low = nhanes_mean
        borderline_high = nhanes_mean + nhanes_std
        high_threshold = nhanes_mean + 2 * nhanes_std

        # Get the glucose value
        glucose = fasting_glucose

        # Calculate utility contribution
        if glucose is None or glucose == "Don't know":
            glucose_score = 0  # Neutral if glucose value is missing
        elif glucose < optimal_threshold:
            # Bonus for healthy range
            glucose_score = 1 - (1 / (1 + np.exp(-k_moderate * (glucose - optimal_threshold))))
        elif borderline_low <= glucose <= borderline_high:
            # Gradual penalty for borderline high (prediabetic)
            glucose_score = -(1 - (1 / (1 + np.exp(-k_moderate * (glucose - borderline_high)))))
        elif glucose >= high_threshold:
            # Sharp penalty for high (diabetic)
            glucose_score = -(1 - (1 / (1 + np.exp(-k_sharp * (glucose - high_threshold)))))
        else:
            glucose_score = 0  # Neutral for undefined ranges

        # Scale the contribution to the utility score
        glucose_contribution = round(glucose_score * 0.15, 4)  # Adjust weight as needed
    else:
        glucose_contribution = 0  # Neutral score if benchmarks are missing

    # Scenario-Based Fasting Glucose Contributions
    scenario_glucose_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Fetch NHANES benchmarks for Fasting Glucose
        nhanes_mean, nhanes_std = benchmarks.get("LBXGLU" if unit_choice == "mg" else "LBDGLUSI", (None, None))

        # Sigmoid parameters
        k_moderate = 0.2  # Steepness for borderline high
        k_sharp = 0.4     # Steepness for very high glucose

        if nhanes_mean is not None and nhanes_std is not None:
            # Calculate NHANES-based thresholds
            optimal_threshold = nhanes_mean
            borderline_low = nhanes_mean
            borderline_high = nhanes_mean + nhanes_std
            high_threshold = nhanes_mean + 2 * nhanes_std

            # Get the glucose value for the current scenario
            scenario_glucose = scenario_data.get("Fasting Glucose")

            # Calculate utility contribution
            if scenario_glucose is None or scenario_glucose == "Don't know":
                scenario_glucose_score = 0  # Neutral if glucose value is missing
            elif scenario_glucose < optimal_threshold:
                # Bonus for healthy range
                scenario_glucose_score = 1 - (1 / (1 + np.exp(-k_moderate * (scenario_glucose - optimal_threshold))))
            elif borderline_high <= scenario_glucose < high_threshold:
                # Moderate penalty for borderline high (prediabetic)
                scenario_glucose_score = -(1 - (1 / (1 + np.exp(-k_moderate * (scenario_glucose - borderline_high))))) - 0.02 * (scenario_glucose - borderline_high)
            elif scenario_glucose >= high_threshold:
                # Sharp penalty for high (diabetic)
                scenario_glucose_score = -(1 - (1 / (1 + np.exp(-k_sharp * (scenario_glucose - high_threshold))))) * (1 + alpha * (scenario_glucose / high_threshold))
            else:
                scenario_glucose_score = 0  # Neutral for undefined ranges


            # Scale the contribution to the utility score
            scenario_glucose_contribution = round(scenario_glucose_score * 0.15, 4)  # Adjust weight as needed
        else:
            scenario_glucose_contribution = 0  # Neutral score if benchmarks are missing

        # Store contribution for the current scenario
        scenario_glucose_contributions[scenario_name] = scenario_glucose_contribution


    # Baseline thresholds for BMI (medical classification)
    baseline_bmi = {
        'low': 18.5,          # Below this is underweight
        'optimal_low': 18.5,  # Lower bound of optimal range
        'optimal_high': 24.9, # Upper bound of optimal range
        'overweight_low': 25, # Lower bound of overweight
        'overweight_high': 29.9, # Upper bound of overweight
        'obesity_i': 30,      # Obesity Class I
        'obesity_ii': 35,     # Obesity Class II
        'obesity_iii': 40     # Obesity Class III
    }

    # Extract NHANES benchmarks for BMI (mean and standard deviation)
    nhanes_mean, nhanes_std = benchmarks.get("BMXBMI", (None, None))

    # Dynamically adjust thresholds based on NHANES data
    if nhanes_mean is not None and nhanes_std is not None:
        # Use NHANES data to adjust the steepness of the penalties/rewards
        k_gradual = 0.2 + (nhanes_std / nhanes_mean)  # Adjust steepness based on NHANES std
        k_steep = 0.4 + (nhanes_std / nhanes_mean)    # Steepness for penalties based on NHANES std
    else:
        # If NHANES data is not available (fallback), use fixed penalties
        k_gradual = 0.2
        k_steep = 0.4

    # Calculate utility contribution for BMI based on medical thresholds and NHANES fine-tuning
    if bmi is None or bmi == "Don't know":
        bmi_score = 0  # Neutral if BMI is missing
    elif bmi < baseline_bmi['low']:
        # Severe penalty for underweight
        bmi_score = -(1 - (1 / (1 + np.exp(-k_steep * (bmi - baseline_bmi['optimal_low'])))))
    elif baseline_bmi['optimal_low'] <= bmi <= baseline_bmi['optimal_high']:
        # Bonus for optimal range
        bmi_score = 1 - (1 / (1 + np.exp(-k_gradual * (bmi - baseline_bmi['optimal_high']))))
    elif baseline_bmi['overweight_low'] <= bmi <= baseline_bmi['overweight_high']:
        # Gradual penalty for overweight
        bmi_score = -(1 - (1 / (1 + np.exp(-k_gradual * (bmi - baseline_bmi['optimal_high'])))))
    elif bmi >= baseline_bmi['obesity_i']:
        # Steeper penalty for obesity
        bmi_score = -(1 - (1 / (1 + np.exp(-k_steep * (bmi - baseline_bmi['obesity_i'])))))
    else:
        bmi_score = 0  # Neutral for undefined ranges

    # Scale the contribution to the utility score
    bmi_contribution = round(bmi_score * 0.18, 4)  # Adjust weight as needed

    # Scenario-Based BMI Contributions
    scenario_bmi_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Extract NHANES benchmarks for BMI (mean and standard deviation)
        nhanes_mean, nhanes_std = benchmarks.get("BMXBMI", (None, None))

        # Dynamically adjust thresholds based on NHANES data
        if nhanes_mean is not None and nhanes_std is not None:
            k_gradual = 0.2 + (nhanes_std / nhanes_mean)  # Adjust steepness based on NHANES std
            k_steep = 0.4 + (nhanes_std / nhanes_mean)    # Steepness for penalties based on NHANES std
        else:
            k_gradual = 0.2  # Fallback steepness
            k_steep = 0.4

        # Get BMI value for the current scenario
        scenario_bmi = scenario_data.get("BMI")

        # Calculate contribution based on BMI thresholds
        if scenario_bmi is None or scenario_bmi == "Don't know":
            scenario_bmi_score = 0  # Neutral if BMI is missing
        elif scenario_bmi < baseline_bmi['low']:
            # Severe penalty for underweight
            scenario_bmi_score = -(1 - (1 / (1 + np.exp(-k_steep * (scenario_bmi - baseline_bmi['optimal_low'])))))
        elif baseline_bmi['optimal_low'] <= scenario_bmi <= baseline_bmi['optimal_high']:
            # Bonus for optimal range
            scenario_bmi_score = 1 - (1 / (1 + np.exp(-k_gradual * (scenario_bmi - baseline_bmi['optimal_high']))))
        elif baseline_bmi['overweight_low'] <= scenario_bmi <= baseline_bmi['overweight_high']:
            # Gradual penalty for overweight
            scenario_bmi_score = -(1 - (1 / (1 + np.exp(-k_gradual * (scenario_bmi - baseline_bmi['optimal_high'])))))
        elif scenario_bmi >= baseline_bmi['obesity_i']:
            # Steeper penalty for obesity
            scenario_bmi_score = -(1 - (1 / (1 + np.exp(-k_steep * (scenario_bmi - baseline_bmi['obesity_i'])))))
        else:
            scenario_bmi_score = 0  # Neutral for undefined ranges

        # Scale the contribution to the utility score
        scenario_bmi_contribution = round(scenario_bmi_score * 0.18, 4)  # Adjust weight as needed

        # Store the contribution for the scenario
        scenario_bmi_contributions[scenario_name] = scenario_bmi_contribution

    waist = waist_circumference_cm

    # Retrieve NHANES benchmarks (population-level, not gender-differentiated)
    nhanes_mean, nhanes_std = benchmarks.get("BMXWAIST", (None, None))


    # Calculate gender-specific thresholds based on NHANES data or baseline values
    if gender == "Male":
        nhanes_mean, nhanes_std = benchmarks.get("BMXWAIST", (None, None))
        thresholds = {
            'low_risk': max(nhanes_mean - nhanes_std, 94),  # Use higher of NHANES or baseline low risk
            'high_risk': max(nhanes_mean, 102),  # Use higher of NHANES or baseline high risk
            'cap': 120  # Cap remains the same
        }
    elif gender == "Female":
        nhanes_mean, nhanes_std = benchmarks.get("BMXWAIST", (None, None))
        thresholds = {
            'low_risk': max(nhanes_mean - nhanes_std, 80),  # Use higher of NHANES or baseline low risk
            'high_risk': max(nhanes_mean, 88),  # Use higher of NHANES or baseline high risk
            'cap': 120  # Cap remains the same
        }
    else:
        waist_contribution = 0  # Neutral score for missing gender
        thresholds = None

    # Scaling Logic
    if thresholds:
        if waist is None or waist == "Don't know":
            waist_score = 0  # Neutral if Waist Circumference value is missing
        elif waist <= thresholds['low_risk']:
            # Bonus for values below the low-risk threshold
            waist_score = 0.1
        elif thresholds['low_risk'] < waist <= thresholds['high_risk']:
            # Gradual penalty for values in the borderline range
            waist_score = -0.2 * ((waist - thresholds['low_risk']) / (thresholds['high_risk'] - thresholds['low_risk']))
        elif waist > thresholds['high_risk']:
            # Steeper penalty for values above the high-risk threshold
            # Capped penalty for extreme waist circumference
            capped_waist = min(waist, thresholds['cap'])
            waist_score = -0.2 * ((capped_waist - thresholds['high_risk']) / (thresholds['cap'] - thresholds['high_risk']))
        else:
            waist_score = 0  # Neutral for undefined ranges

        # Scale the contribution to the utility score
        waist_contribution = round(waist_score * 0.12, 4)  # Adjust weight as needed
    else:
        waist_contribution = 0  # Neutral if thresholds are undefined

    # Scenario-Based Waist Circumference Contributions
    scenario_waist_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Retrieve NHANES benchmarks (population-level, not gender-differentiated)
        nhanes_mean, nhanes_std = benchmarks.get("BMXWAIST", (None, None))

        # Determine gender-specific thresholds
        gender = scenario_data.get("Gender")
        if gender == "Male":
            thresholds = {
                'low_risk': max(nhanes_mean - nhanes_std, 94) if nhanes_mean else 94,
                'high_risk': max(nhanes_mean, 102) if nhanes_mean else 102,
                'cap': 120
            }
        elif gender == "Female":
            thresholds = {
                'low_risk': max(nhanes_mean - nhanes_std, 80) if nhanes_mean else 80,
                'high_risk': max(nhanes_mean, 88) if nhanes_mean else 88,
                'cap': 120
            }
        else:
            scenario_waist_contributions[scenario_name] = 0  # Neutral score for missing gender
            continue

        # Get Waist Circumference value for the current scenario
        scenario_waist = scenario_data.get("Waist Circumference (cm)")

        # Scaling Logic
        if scenario_waist is None or scenario_waist == "Don't know":
            scenario_waist_score = 0  # Neutral if Waist Circumference value is missing
        elif scenario_waist <= thresholds['low_risk']:
            # Bonus for values below the low-risk threshold
            scenario_waist_score = 0.1
        elif thresholds['low_risk'] < scenario_waist <= thresholds['high_risk']:
            # Gradual penalty for values in the borderline range
            scenario_waist_score = -0.2 * ((scenario_waist - thresholds['low_risk']) / (thresholds['high_risk'] - thresholds['low_risk']))
        elif scenario_waist > thresholds['high_risk']:
            # Steeper penalty for values above the high-risk threshold
            capped_waist = min(scenario_waist, thresholds['cap'])
            scenario_waist_score = -0.2 * ((capped_waist - thresholds['high_risk']) / (thresholds['cap'] - thresholds['high_risk']))
        else:
            scenario_waist_score = 0  # Neutral for undefined ranges

        # Scale the contribution to the utility score
        scenario_waist_contribution = round(scenario_waist_score * 0.12, 4)  # Adjust weight as needed

        # Store the contribution for the scenario
        scenario_waist_contributions[scenario_name] = scenario_waist_contribution

    # Baseline thresholds for Fiber Intake (g/day)
    baseline_fiber = {
        'Male': {
            '19-30': 34,
            '31-50': 31,
            '51+': 28
        },
        'Female': {
            '19-30': 28,
            '31-50': 25,
            '51+': 22
        }
    }

    # Sigmoid parameters
    k_gradual = 0.3  # Steepness for gradual penalties below baseline
    k_minimal = 0.1  # Steepness for minimal penalties above optimal

    # Retrieve NHANES benchmarks for fiber intake
    nhanes_mean, nhanes_std = benchmarks.get("DR1TFIBE", (None, None))

    # Determine age group and baseline based on gender and age
    if gender == "Male":
        if 19 <= age_at_screening <= 30:
            fiber_baseline = max(nhanes_mean, baseline_fiber['Male']['19-30']) if nhanes_mean is not None else baseline_fiber['Male']['19-30']
        elif 31 <= age_at_screening <= 50:
            fiber_baseline = max(nhanes_mean, baseline_fiber['Male']['31-50']) if nhanes_mean is not None else baseline_fiber['Male']['31-50']
        elif age_at_screening > 50:
            fiber_baseline = max(nhanes_mean, baseline_fiber['Male']['51+']) if nhanes_mean is not None else baseline_fiber['Male']['51+']
        else:
            fiber_baseline = None
    elif gender == "Female":
        if 19 <= age_at_screening <= 30:
            fiber_baseline = max(nhanes_mean, baseline_fiber['Female']['19-30']) if nhanes_mean is not None else baseline_fiber['Female']['19-30']
        elif 31 <= age_at_screening <= 50:
            fiber_baseline = max(nhanes_mean, baseline_fiber['Female']['31-50']) if nhanes_mean is not None else baseline_fiber['Female']['31-50']
        elif age_at_screening > 50:
            fiber_baseline = max(nhanes_mean, baseline_fiber['Female']['51+']) if nhanes_mean is not None else baseline_fiber['Female']['51+']
        else:
            fiber_baseline = None
    else:
        fiber_contribution = 0  # Neutral score for missing gender
        fiber_baseline = None

    # Fiber intake logic
    if fiber_baseline is not None:
        # Get the fiber intake value
        fiber = dietary_fiber_gm

        # Sigmoidal Logic for scoring
        if fiber is None or fiber == "Don't know":
            fiber_score = 0  # Neutral if Fiber Intake value is missing
        elif fiber < 20:
            # Gradual penalties for intake below 20 g/day
            fiber_score = -(1 - (1 / (1 + np.exp(-k_gradual * (fiber - 20)))))
        elif 20 <= fiber <= fiber_baseline:
            # Bonus for values in the healthy range
            fiber_score = 1 - (1 / (1 + np.exp(-k_gradual * (fiber - fiber_baseline))))
        elif fiber > fiber_baseline and fiber <= 50:
            # Minimal penalty for moderate excess intake
            fiber_score = 0.05  # Small bonus for exceeding baseline moderately
        elif fiber > 50:
            # Gradual penalties for extreme intake (>50 g/day)
            fiber_score = -(1 - (1 / (1 + np.exp(-k_minimal * (fiber - 50)))))
        else:
            fiber_score = 0  # Neutral for undefined ranges

        # Scale the contribution to the utility score
        fiber_contribution = round(fiber_score * 0.1, 4)  # Adjust weight as needed
    else:
        fiber_contribution = 0  # Neutral if baseline is undefined

    # Scenario-Based Fiber Intake Contributions
    scenario_fiber_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Retrieve NHANES benchmarks for fiber intake
        nhanes_mean, nhanes_std = benchmarks.get("DR1TFIBE", (None, None))

        # Determine gender and age group for baseline thresholds
        gender = scenario_data.get("Gender")
        age_at_screening = scenario_data.get("Age")
        fiber_baseline = None

        if gender == "Male":
            if 19 <= age_at_screening <= 30:
                fiber_baseline = max(nhanes_mean, baseline_fiber['Male']['19-30']) if nhanes_mean is not None else baseline_fiber['Male']['19-30']
            elif 31 <= age_at_screening <= 50:
                fiber_baseline = max(nhanes_mean, baseline_fiber['Male']['31-50']) if nhanes_mean is not None else baseline_fiber['Male']['31-50']
            elif age_at_screening > 50:
                fiber_baseline = max(nhanes_mean, baseline_fiber['Male']['51+']) if nhanes_mean is not None else baseline_fiber['Male']['51+']
        elif gender == "Female":
            if 19 <= age_at_screening <= 30:
                fiber_baseline = max(nhanes_mean, baseline_fiber['Female']['19-30']) if nhanes_mean is not None else baseline_fiber['Female']['19-30']
            elif 31 <= age_at_screening <= 50:
                fiber_baseline = max(nhanes_mean, baseline_fiber['Female']['31-50']) if nhanes_mean is not None else baseline_fiber['Female']['31-50']
            elif age_at_screening > 50:
                fiber_baseline = max(nhanes_mean, baseline_fiber['Female']['51+']) if nhanes_mean is not None else baseline_fiber['Female']['51+']
        else:
            scenario_fiber_contributions[scenario_name] = 0  # Neutral score for missing gender
            continue

        # Fiber intake value for the current scenario
        fiber = scenario_data.get("Dietary Fiber (g)")

        # Scoring Logic
        if fiber_baseline is not None:
            if fiber is None or fiber == "Don't know":
                scenario_fiber_score = 0  # Neutral if Fiber Intake value is missing
            elif fiber < 20:
                # Gradual penalties for intake below 20 g/day
                scenario_fiber_score = -(1 - (1 / (1 + np.exp(-k_gradual * (fiber - 20)))))
            elif 20 <= fiber <= fiber_baseline:
                # Bonus for values in the healthy range
                scenario_fiber_score = 1 - (1 / (1 + np.exp(-k_gradual * (fiber - fiber_baseline))))
            elif fiber > fiber_baseline and fiber <= 50:
                # Higher bonus for exceeding baseline moderately
                scenario_fiber_score = 0.1 + 0.02 * (fiber - fiber_baseline) / (50 - fiber_baseline)
            elif fiber > 50:
                # Gradual penalties for extreme intake (>50 g/day)
                scenario_fiber_score = -(1 - (1 / (1 + np.exp(-k_minimal * (fiber - 50)))))
            else:
                scenario_fiber_score = 0  # Neutral for undefined ranges

            # Scale the contribution to the utility score
            scenario_fiber_contribution = round(scenario_fiber_score * 0.1, 4)  # Adjust weight as needed
        else:
            scenario_fiber_contribution = 0  # Neutral if baseline is undefined

        # Store contribution for the scenario
        scenario_fiber_contributions[scenario_name] = scenario_fiber_contribution

    # Baseline thresholds for Vitamin C Intake (mg/day)
    baseline_vitamin_c = {
        'Male': 90,
        'Female': 75
    }


    # Retrieve NHANES benchmarks (mean and std)
    nhanes_mean, nhanes_std = benchmarks.get("DR1TVC", (None, None))

    # Initialize NHANES thresholds
    nhanes_thresholds = None

    if nhanes_mean is not None and nhanes_std is not None:
        nhanes_thresholds = {
            'low_risk': max(nhanes_mean - nhanes_std, 50),  # Minimum safe intake is 50 mg/day
            'baseline_male': max(nhanes_mean, 90),  # Higher of NHANES or baseline male
            'baseline_female': max(nhanes_mean, 75),  # Higher of NHANES or baseline female
            'cap': nhanes_mean + 2 * nhanes_std  # Cap at 2 SDs above mean
        }

    # If NHANES data is unavailable, fall back to baseline thresholds
    baseline_vitamin_c = {
        'Male': 90,
        'Female': 75
    }

    # Determine thresholds based on gender
    if gender == "Male":
        vitamin_c_baseline = (
            nhanes_thresholds['baseline_male'] if nhanes_thresholds else baseline_vitamin_c['Male']
        )
        low_risk = (
            nhanes_thresholds['low_risk'] if nhanes_thresholds else 50
        )
        cap_bonus = (
            nhanes_thresholds['cap'] if nhanes_thresholds else 200
        )
    elif gender == "Female":
        vitamin_c_baseline = (
            nhanes_thresholds['baseline_female'] if nhanes_thresholds else baseline_vitamin_c['Female']
        )
        low_risk = (
            nhanes_thresholds['low_risk'] if nhanes_thresholds else 50
        )
        cap_bonus = (
            nhanes_thresholds['cap'] if nhanes_thresholds else 200
        )
    else:
        vitamin_c_contribution = 0  # Neutral score for missing gender
        vitamin_c_baseline = None

    # Vitamin C intake logic
    if vitamin_c_baseline is not None:

        # Scaling Logic
        if vitamin_c is None or vitamin_c == "Don't know":
            vitamin_c_score = 0  # Neutral if Vitamin C intake is missing
        elif vitamin_c < low_risk:
            # Penalty for values below low risk
            vitamin_c_score = -0.05 * (low_risk - vitamin_c) / (vitamin_c_baseline - low_risk)
        elif low_risk <= vitamin_c <= vitamin_c_baseline:
            # Bonus for values within the baseline range
            vitamin_c_score = 0.05 * (vitamin_c - low_risk) / (vitamin_c_baseline - low_risk)
        elif vitamin_c_baseline < vitamin_c <= cap_bonus:
            # Gradual bonus for moderate excess intake
            vitamin_c_score = 0.05 * (vitamin_c - vitamin_c_baseline) / (cap_bonus - vitamin_c_baseline)
        elif vitamin_c > cap_bonus:
            # Cap excessive bonuses
            vitamin_c_score = 0.05  # Fixed maximum bonus
        else:
            vitamin_c_score = 0  # Neutral for undefined ranges

        # Scale the contribution to the utility score
        vitamin_c_contribution = round(vitamin_c_score * 0.1, 4)  # Adjust weight as needed
    else:
        vitamin_c_contribution = 0  # Neutral if baseline is undefined

    # Scenario-Based Vitamin C Contributions
    scenario_vitamin_c_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Retrieve NHANES benchmarks for Vitamin C intake
        nhanes_mean, nhanes_std = benchmarks.get("DR1TVC", (None, None))

        # Initialize NHANES thresholds
        nhanes_thresholds = None
        if nhanes_mean is not None and nhanes_std is not None:
            nhanes_thresholds = {
                'low_risk': max(nhanes_mean - nhanes_std, 50),  # Minimum safe intake is 50 mg/day
                'baseline_male': max(nhanes_mean, 90),  # Higher of NHANES or baseline male
                'baseline_female': max(nhanes_mean, 75),  # Higher of NHANES or baseline female
                'cap': nhanes_mean + 2 * nhanes_std  # Cap at 2 SDs above mean
            }

        # Determine gender-specific thresholds
        gender = scenario_data.get("Gender")
        if gender == "Male":
            vitamin_c_baseline = nhanes_thresholds['baseline_male'] if nhanes_thresholds else baseline_vitamin_c['Male']
            low_risk = nhanes_thresholds['low_risk'] if nhanes_thresholds else 50
            cap_bonus = nhanes_thresholds['cap'] if nhanes_thresholds else 200
        elif gender == "Female":
            vitamin_c_baseline = nhanes_thresholds['baseline_female'] if nhanes_thresholds else baseline_vitamin_c['Female']
            low_risk = nhanes_thresholds['low_risk'] if nhanes_thresholds else 50
            cap_bonus = nhanes_thresholds['cap'] if nhanes_thresholds else 200
        else:
            scenario_vitamin_c_contributions[scenario_name] = 0  # Neutral score for missing gender
            continue

        # Get Vitamin C intake for the scenario
        vitamin_c = scenario_data.get("Vitamin C (mg)")

        # Scoring Logic
        if vitamin_c_baseline is not None:
            if vitamin_c is None or vitamin_c == "Don't know":
                scenario_vitamin_c_score = 0  # Neutral if Vitamin C intake is missing
            elif vitamin_c < low_risk:
                # Penalty for values below low risk
                scenario_vitamin_c_score = -0.5 * (low_risk - vitamin_c) / (vitamin_c_baseline - low_risk)
            elif low_risk <= vitamin_c <= vitamin_c_baseline:
                # Bonus for values within the baseline range
                scenario_vitamin_c_score = 0.4 * (vitamin_c - low_risk) / (vitamin_c_baseline - low_risk)
            elif vitamin_c_baseline < vitamin_c <= cap_bonus:
                # Gradual bonus for moderate excess intake
                scenario_vitamin_c_score = 0.4 + 0.5 * (vitamin_c - vitamin_c_baseline) / (cap_bonus - vitamin_c_baseline)
            elif vitamin_c > cap_bonus:
                # Cap excessive bonuses
                scenario_vitamin_c_score = 0.5  # Fixed maximum bonus
            else:
                scenario_vitamin_c_score = 0  # Neutral for undefined ranges

            # Scale the contribution to the utility score
            scenario_vitamin_c_contribution = round(scenario_vitamin_c_score * 0.3, 4)  # Adjust weight for Vitamin C
        else:
            scenario_vitamin_c_contribution = 0  # Neutral if baseline is undefined

        # Store the contribution for the scenario
        scenario_vitamin_c_contributions[scenario_name] = scenario_vitamin_c_contribution

    # Baseline thresholds for Calorie Intake (kcal/day)
    baseline_calories = {
        'Male': {
            'Sedentary': {19: 2400, 31: 2200, 51: 2000},
            'Moderate': {19: 2600, 31: 2400, 51: 2200},
            'Active': {19: 3000, 31: 2800, 51: 2400}
        },
        'Female': {
            'Sedentary': {19: 1800, 31: 1800, 51: 1600},
            'Moderate': {19: 2000, 31: 2000, 51: 1800},
            'Active': {19: 2400, 31: 2200, 51: 2000}
        }
    }

    # Get NHANES benchmarks for calorie intake
    nhanes_mean, nhanes_std = benchmarks.get("DR1TKCAL", (None, None))

    # Calculate thresholds combining NHANES and baseline values
    def calculate_threshold(age, gender, activity):
        # Retrieve baseline calorie intake
        baseline_value = baseline_calories[gender][activity][age]

        if nhanes_mean is not None and nhanes_std is not None:
            # NHANES thresholds
            nhanes_low = nhanes_mean - nhanes_std
            nhanes_high = nhanes_mean + nhanes_std

            # Combine NHANES and baseline thresholds
            combined_low = max(nhanes_low, baseline_value * 0.8)
            combined_high = min(nhanes_high, baseline_value * 1.2)
            return combined_low, combined_high
        else:
            # Fallback to baseline thresholds with a 20% margin
            return baseline_value * 0.8, baseline_value * 1.2

    # Determine activity level from widget inputs
    if vigorous_activities == "Yes" and days_vigorous_activities >= 3:
        activity_level = "Active"
    elif moderate_activities == "Yes" and days_moderate_activities >= 4:
        activity_level = "Moderate"
    elif walk_bicycle == "Yes" or minutes_sedentary > 300:  # Sedentary or minimal activity
        activity_level = "Sedentary"
    else:
        activity_level = "Don't know"  # Fallback

    # Determine calorie intake thresholds based on age, gender, and activity level
    if gender in baseline_calories and activity_level != "Don't know":
        if age_at_screening <= 30:
            age_group = 19
        elif 31 <= age_at_screening <= 50:
            age_group = 31
        else:
            age_group = 51

        # Calculate thresholds
        low, high = calculate_threshold(age_group, gender, activity_level)
    elif gender == "Prefer not to say" and activity_level != "Don't know":
        # Average thresholds for "Prefer not to say"
        if age_at_screening <= 30:
            age_group = 19
        elif 31 <= age_at_screening <= 50:
            age_group = 31
        else:
            age_group = 51

        low_male, high_male = calculate_threshold(age_group, 'Male', activity_level)
        low_female, high_female = calculate_threshold(age_group, 'Female', activity_level)
        low = (low_male + low_female) / 2
        high = (high_male + high_female) / 2
    else:
        calorie_contribution = 0  # Neutral score
        low, high = None, None

    # Scaling logic for calorie intake
    if low is not None and high is not None:
        # Get calorie intake value
        calories = energy_kcal  # Value from the widget

        if calories is None or calories == "Don't know":
            calorie_score = 0  # Neutral if calorie intake is missing
        elif low <= calories <= high:
            calorie_score = 0.1  # Bonus for being within healthy range
        elif calories < low:
            # Penalty for low intake
            deviation = (low - calories) / low
            calorie_score = -0.1 * deviation
        elif calories > high:
            # Penalty for high intake
            deviation = (calories - high) / high
            calorie_score = -0.1 * deviation

        # Cap extreme penalties
        calorie_score = max(min(calorie_score, 0.1), -0.2)

        # Scale the contribution to the utility score
        calorie_contribution = round(calorie_score * 0.25, 4)
    else:
        calorie_contribution = 0  # Neutral if thresholds are undefined

    # Scenario-Based Calorie Contributions
    scenario_calorie_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Retrieve NHANES benchmarks for calorie intake
        nhanes_mean, nhanes_std = benchmarks.get("DR1TKCAL", (None, None))

        # Determine gender, age, and activity level
        gender = scenario_data.get("Gender")
        age_at_screening = scenario_data.get("Age")
        vigorous_activities = scenario_data.get("Vigorous Activities")
        days_vigorous_activities = scenario_data.get("Days of Vigorous Activities")
        moderate_activities = scenario_data.get("Moderate Activities")
        days_moderate_activities = scenario_data.get("Days of Moderate Activities")
        walk_bicycle = scenario_data.get("Walk/Bicycle")
        minutes_sedentary = scenario_data.get("Minutes Sedentary")
        energy_kcal = scenario_data.get("Energy Intake (kcal)")

        # Determine activity level
        if vigorous_activities == "Yes" and days_vigorous_activities >= 3:
            activity_level = "Active"
        elif moderate_activities == "Yes" and days_moderate_activities >= 4:
            activity_level = "Moderate"
        elif walk_bicycle == "Yes" or minutes_sedentary > 300:
            activity_level = "Sedentary"
        else:
            activity_level = "Don't know"

        # Determine calorie thresholds based on age, gender, and activity level
        if gender in baseline_calories and activity_level != "Don't know":
            if age_at_screening <= 30:
                age_group = 19
            elif 31 <= age_at_screening <= 50:
                age_group = 31
            else:
                age_group = 51

            # Calculate thresholds
            low, high = calculate_threshold(age_group, gender, activity_level)
        elif gender == "Prefer not to say" and activity_level != "Don't know":
            # Average thresholds for "Prefer not to say"
            if age_at_screening <= 30:
                age_group = 19
            elif 31 <= age_at_screening <= 50:
                age_group = 31
            else:
                age_group = 51

            low_male, high_male = calculate_threshold(age_group, 'Male', activity_level)
            low_female, high_female = calculate_threshold(age_group, 'Female', activity_level)
            low = (low_male + low_female) / 2
            high = (high_male + high_female) / 2
        else:
            scenario_calorie_contributions[scenario_name] = 0  # Neutral score
            continue

        # Scoring Logic for Calorie Intake
        if low is not None and high is not None:
            if energy_kcal is None or energy_kcal == "Don't know":
                calorie_contribution = 0  # Neutral if calorie intake is missing
            elif low <= energy_kcal <= high:
                calorie_contribution = 0.1 * 0.25  # Bonus for being within the healthy range
            elif energy_kcal < low:
                # Higher penalty for low intake
                deviation = (low - energy_kcal) / low
                calorie_contribution = -0.15 * deviation * 0.25
            elif energy_kcal > high:
                # Higher penalty for high intake
                deviation = (energy_kcal - high) / high
                calorie_contribution = -0.15 * deviation * 0.25

            # Cap extreme penalties
            calorie_contribution = round(max(min(calorie_contribution, 0.025), -0.075), 4)
        else:
            calorie_contribution = 0  # Neutral if thresholds are undefined

        # Store the contribution for the scenario
        scenario_calorie_contributions[scenario_name] = calorie_contribution

    # Retrieve NHANES benchmarks for BPQ020 and BPQ050A
    bpq020_benchmarks = categorical_benchmarks.get("BPQ020", {})
    bpq050a_benchmarks = categorical_benchmarks.get("BPQ050A", {})

    # Get NHANES distribution percentages for BPQ020
    bpq020_yes_pct = bpq020_benchmarks.get("Yes", 0)
    bpq020_no_pct = bpq020_benchmarks.get("No", 0)

    # Get NHANES distribution percentages for BPQ050A
    bpq050a_yes_pct = bpq050a_benchmarks.get("Yes", 0)
    bpq050a_no_pct = bpq050a_benchmarks.get("No", 0)

    # Initialize contributions
    bpq020_contribution = 0
    bpq050a_contribution = 0

    # Get responses from the widget
    bpq020 = high_blood_pressure_history  # "Yes", "No", or "Don't know"
    bpq050a = taking_bp_meds  # "Yes", "No", or "Don't know"

    # Adjust BPQ020 Contribution based on NHANES context
    if bpq020 == "Yes":
        bpq020_contribution = -0.3 * (1 - bpq020_yes_pct / 100)  # Penalty for "Yes"
        if bpq050a == "No":
            bpq020_contribution *= 1.5  # Stronger penalty if the person has untreated high BP
    elif bpq020 == "No":
        bpq020_contribution = 0.1 * (1 - bpq020_no_pct / 100)  # Bonus for "No"
    elif bpq020 == "Don't know":
        bpq020_contribution = 0  # Neutral score

    # Adjust BPQ050A Contribution based on BPQ020 response (No double penalty)
    if bpq050a == "Yes":
        if bpq020 == "Yes":
            bpq050a_contribution = 0.2 * (1 - bpq050a_yes_pct / 100)  # Bonus for taking meds with high BP
        elif bpq020 == "No":
            bpq050a_contribution = 0  # No bonus for no BP history
        else:
            bpq050a_contribution = 0  # Neutral if "Don't know"
    elif bpq050a == "No":
        if bpq020 == "Yes":
            # Increase the penalty if person has high BP and not taking medication
            bpq050a_contribution = -0.4 * (1 - bpq050a_no_pct / 100)  # Stronger penalty for not taking meds with high BP
        elif bpq020 == "No":
            bpq050a_contribution = 0  # Neutral if no HBP history
        else:
            bpq050a_contribution = 0  # Neutral if "Don't know"
    elif bpq050a == "Don't know":
        bpq050a_contribution = 0  # Neutral for "Don't know"

    # Apply sigmoid normalization only if BPQ050A is not "No" and BPQ020 is "Yes"
    if bpq050a != "No" and bpq020 == "Yes":
        bpq050a_contribution = sigmoid(bpq050a_contribution)

    # Apply a scaling factor to BPQ050A contribution
    scaling_factor = 0.4  # Scale factor to keep contribution within reasonable bounds
    bpq050a_contribution *= scaling_factor

    # Scenario-Based Blood Pressure Contributions
    scenario_bp_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Retrieve NHANES benchmarks for BPQ020 and BPQ050A
        bpq020_benchmarks = categorical_benchmarks.get("BPQ020", {})
        bpq050a_benchmarks = categorical_benchmarks.get("BPQ050A", {})

        # Get NHANES distribution percentages for BPQ020
        bpq020_yes_pct = bpq020_benchmarks.get("Yes", 0)
        bpq020_no_pct = bpq020_benchmarks.get("No", 0)

        # Get NHANES distribution percentages for BPQ050A
        bpq050a_yes_pct = bpq050a_benchmarks.get("Yes", 0)
        bpq050a_no_pct = bpq050a_benchmarks.get("No", 0)

        # Initialize contributions
        bpq020_contribution = 0
        bpq050a_contribution = 0

        # Retrieve responses for the scenario
        bpq020 = scenario_data.get("High Blood Pressure History")  # "Yes", "No", or "Don't know"
        bpq050a = scenario_data.get("Taking BP Meds")  # "Yes", "No", or "Don't know"

        # Logic for BPQ020
        if bpq020 == "Yes":
            bpq020_contribution = -0.3 * (1 - bpq020_yes_pct / 100)
            if bpq050a == "No":
                bpq020_contribution *= 1.5
        elif bpq020 == "No":
            bpq020_contribution = 0.1 * (1 - bpq020_no_pct / 100)
        elif bpq020 == "Don't know":
            bpq020_contribution = 0

        # Logic for BPQ050A
        if bpq050a == "Yes":
            if bpq020 == "Yes":
                bpq050a_contribution = 0.2 * (1 - bpq050a_yes_pct / 100)
            elif bpq020 == "No":
                bpq050a_contribution = 0
            else:
                bpq050a_contribution = 0
        elif bpq050a == "No":
            if bpq020 == "Yes":
                bpq050a_contribution = -0.4 * (1 - bpq050a_no_pct / 100)
            elif bpq020 == "No":
                bpq050a_contribution = 0
            else:
                bpq050a_contribution = 0
        elif bpq050a == "Don't know":
            bpq050a_contribution = 0

        # Apply sigmoid normalization for BPQ050A contribution if needed
        if bpq050a != "No" and bpq020 == "Yes":
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            bpq050a_contribution = sigmoid(bpq050a_contribution)

        # Apply a scaling factor to BPQ050A contribution
        scaling_factor = 0.4
        bpq050a_contribution *= scaling_factor

        # Combine contributions
        total_bp_contribution = round(bpq020_contribution + bpq050a_contribution, 4)

        # Store the total contribution for the scenario
        scenario_bp_contributions[scenario_name] = total_bp_contribution

    # Initialize contribution
    bpq080_contribution = 0

    # Get response from widget
    bpq080 = high_cholesterol_diagnosis  # "Yes", "No", or "Don't know"

    # Retrieve NHANES benchmark for BPQ080 (high cholesterol diagnosis)
    bpq080_benchmarks = categorical_benchmarks.get("BPQ080", {})

    # Get NHANES distribution percentages from 'all_categories' (These are actual proportions from NHANES)
    bpq080_categories = bpq080_benchmarks.get("all_categories", {})
    bpq080_yes_pct = bpq080_categories.get("Yes", 0)  # Percentage of "Yes" responses
    bpq080_no_pct = bpq080_categories.get("No", 0)    # Percentage of "No" responses

    # Specify overall prevalence for "Yes" and "No" responses
    overall_prevalence_yes = bpq080_yes_pct / 100  # Convert to decimal form
    overall_prevalence_no = bpq080_no_pct / 100  # Convert to decimal form

    # Base penalty for 'Yes' (high cholesterol diagnosis)
    base_penalty = -0.1

    # Apply penalty for high cholesterol diagnosis ('Yes')
    if bpq080 == "Yes":
        # Calculate the deviation between the user's response and NHANES data
        deviation = overall_prevalence_yes - (bpq080_yes_pct / 100)
        # Adjust the penalty based on the deviation
        bpq080_contribution = base_penalty * (1 + deviation)  # Adjust the penalty based on deviation

        # Apply small penalty for "Yes" due to it being a minority category (minority health condition)
        minor_category_penalty = -0.05  # A small additional penalty for "Yes"
        bpq080_contribution += minor_category_penalty  # Apply the additional penalty

    # For "No" (no high cholesterol diagnosis), apply a bonus based on "No" prevalence
    elif bpq080 == "No":
        # Calculate the deviation for "No" (user response of no high cholesterol)
        deviation = (bpq080_no_pct / 100) - overall_prevalence_no  # Use the NHANES "No" percentage
        # Apply a bonus adjusted by the deviation from NHANES
        bpq080_contribution = 0.1 + (deviation * 0.5)  # Apply bonus adjusted by deviation from NHANES

    # Neutral for "Don't know"
    elif bpq080 == "Don't know":
        bpq080_contribution = 0  # Neutral for "Don't know"

    # Scenario-Based High Cholesterol Diagnosis Contributions
    scenario_bpq080_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Retrieve response for high cholesterol diagnosis
        bpq080 = scenario_data.get("High Cholesterol Diagnosis")  # "Yes", "No", or "Don't know"

        # Retrieve NHANES benchmark for BPQ080
        bpq080_benchmarks = categorical_benchmarks.get("BPQ080", {})
        bpq080_categories = bpq080_benchmarks.get("all_categories", {})
        bpq080_yes_pct = bpq080_categories.get("Yes", 0)
        bpq080_no_pct = bpq080_categories.get("No", 0)

        # Calculate overall prevalence
        overall_prevalence_yes = bpq080_yes_pct / 100
        overall_prevalence_no = bpq080_no_pct / 100

        # Initialize contribution
        bpq080_contribution = 0

        # Scoring logic
        if bpq080 == "Yes":
            deviation = overall_prevalence_yes - (bpq080_yes_pct / 100)
            base_penalty = -0.1
            minor_category_penalty = -0.05
            bpq080_contribution = base_penalty * (1 + deviation) + minor_category_penalty
        elif bpq080 == "No":
            deviation = (bpq080_no_pct / 100) - overall_prevalence_no
            bpq080_contribution = 0.1 + (deviation * 0.5)
        elif bpq080 == "Don't know":
            bpq080_contribution = 0  # Neutral for "Don't know"

        # Scale the contribution to fit the utility score range
        scenario_bpq080_contribution = round(bpq080_contribution, 4)

        # Store the contribution for the scenario
        scenario_bpq080_contributions[scenario_name] = scenario_bpq080_contribution

    # Initialize contributions
    diq010_contribution = 0
    diq070_contribution = 0

    # Get responses from widget
    diq010 = diabetes_diagnosis  # "Yes", "No", or "Don't know"
    diq070 = taking_diabetic_pills  # "Yes", "No", or "Don't know"

    # Retrieve NHANES benchmark for diabetes diagnosis (DIQ010) and medication (DIQ070)
    diq010_benchmarks = categorical_benchmarks.get("DIQ010", {})
    diq070_benchmarks = categorical_benchmarks.get("DIQ070", {})

    # Get NHANES distribution percentages for DIQ010 (diabetes diagnosis)
    diq010_categories = diq010_benchmarks.get("all_categories", {})
    diq010_yes_pct = diq010_categories.get("Yes", 0)
    diq010_no_pct = diq010_categories.get("No", 0)

    # Get NHANES distribution percentages for DIQ070 (taking diabetic pills)
    diq070_categories = diq070_benchmarks.get("all_categories", {})
    diq070_yes_pct = diq070_categories.get("Yes", 0)
    diq070_no_pct = diq070_categories.get("No", 0)

    # Specify overall prevalences for DIQ010 ("Yes" and "No" responses)
    overall_prevalence_diq010_yes = diq010_yes_pct / 100  # Convert to decimal form


    # Base penalty for "Yes" (diabetes diagnosis)
    base_penalty_diq010 = -0.15

    # For DIQ010 (diabetes diagnosis), adjust contribution based on NHANES data
    if diq010 == "Yes":
        # Calculate the deviation between the user's response and NHANES data for diabetes diagnosis
        deviation = overall_prevalence_diq010_yes - (diq010_yes_pct / 100)
        diq010_contribution = base_penalty_diq010 * (1 + deviation)  # Adjust the penalty based on deviation
    elif diq010 == "No":
        diq010_contribution = 0.1  # Bonus for no diabetes diagnosis
    elif diq010 == "Don't know":
        diq010_contribution = 0  # Neutral for "Don't know"

    # For DIQ070 (taking diabetic pills), adjust contribution based on NHANES data
    if diq070 == "Yes":
        if diq010 == "Yes":
            diq070_contribution = 0.05 * (1 - diq070_yes_pct / 100)  # Bonus for taking pills if diagnosed
        elif diq010 == "No":
            diq070_contribution = 0  # Neutral if no diabetes diagnosis
        else:
            diq070_contribution = 0  # Neutral for "Don't know"
    elif diq070 == "No":
        if diq010 == "Yes":
            diq070_contribution = -0.1 * (1 - diq070_no_pct / 100)  # Penalty for not taking medication despite diagnosis
        elif diq010 == "No":
            diq070_contribution = 0  # Neutral if no diabetes diagnosis
        else:
            diq070_contribution = 0  # Neutral for "Don't know"
    elif diq070 == "Don't know":
        diq070_contribution = 0  # Neutral for "Don't know"

    # Scenario-Based Diabetes Contributions
    scenario_diabetes_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Retrieve NHANES benchmarks
        diq010_benchmarks = categorical_benchmarks.get("DIQ010", {})
        diq070_benchmarks = categorical_benchmarks.get("DIQ070", {})

        # Get NHANES distribution percentages for DIQ010 (diabetes diagnosis)
        diq010_categories = diq010_benchmarks.get("all_categories", {})
        diq010_yes_pct = diq010_categories.get("Yes", 0)
        diq010_no_pct = diq010_categories.get("No", 0)

        # Get NHANES distribution percentages for DIQ070 (taking diabetic pills)
        diq070_categories = diq070_benchmarks.get("all_categories", {})
        diq070_yes_pct = diq070_categories.get("Yes", 0)
        diq070_no_pct = diq070_categories.get("No", 0)

        # Retrieve scenario-specific responses
        diq010 = scenario_data.get("Diabetes Diagnosis")  # "Yes", "No", or "Don't know"
        diq070 = scenario_data.get("Taking Diabetic Pills")  # "Yes", "No", or "Don't know"

        # Initialize contributions
        diq010_contribution = 0
        diq070_contribution = 0

        # Adjust DIQ010 Contribution
        if diq010 == "Yes":
            deviation = (diq010_yes_pct / 100)  # Prevalence in NHANES
            diq010_contribution = -0.15 * (1 + deviation)  # Penalty for diabetes diagnosis
        elif diq010 == "No":
            diq010_contribution = 0.1  # Bonus for no diabetes diagnosis
        elif diq010 == "Don't know":
            diq010_contribution = 0  # Neutral

        # Adjust DIQ070 Contribution
        if diq070 == "Yes":
            if diq010 == "Yes":
                diq070_contribution = 0.05 * (1 - diq070_yes_pct / 100)  # Bonus for compliance
            elif diq010 == "No":
                diq070_contribution = 0  # Neutral if no diagnosis
        elif diq070 == "No":
            if diq010 == "Yes":
                diq070_contribution = -0.1 * (1 - diq070_no_pct / 100)  # Penalty for non-compliance
            elif diq010 == "No":
                diq070_contribution = 0  # Neutral if no diagnosis
        elif diq070 == "Don't know":
            diq070_contribution = 0  # Neutral

        # Total contribution for the scenario
        total_diabetes_contribution = round(diq010_contribution + diq070_contribution, 4)

        # Store contributions
        scenario_diabetes_contributions[scenario_name] = total_diabetes_contribution

    # Initialize contributions
    dpq010_contribution = 0
    dpq020_contribution = 0
    dpq040_contribution = 0

    # Get responses from the widget
    dpq010 = interest_loss  # "Not at all", "Several days", "More than half the days", "Nearly every day"
    dpq020 = feeling_depressed
    dpq040 = feeling_tired

    # Retrieve NHANES benchmark for depression metrics (from `all_categories` for each depression symptom)
    dpq010_benchmarks = categorical_benchmarks.get("DPQ010", {})
    dpq020_benchmarks = categorical_benchmarks.get("DPQ020", {})
    dpq040_benchmarks = categorical_benchmarks.get("DPQ040", {})

    # Get NHANES distribution percentages for each depression metric
    dpq010_categories = dpq010_benchmarks.get("all_categories", {})
    dpq020_categories = dpq020_benchmarks.get("all_categories", {})
    dpq040_categories = dpq040_benchmarks.get("all_categories", {})

    dpq010_not_at_all_pct = dpq010_categories.get("Not at all", 0)
    dpq010_several_days_pct = dpq010_categories.get("Several days", 0)
    dpq010_more_than_half_pct = dpq010_categories.get("More than half the days", 0)
    dpq010_nearly_every_day_pct = dpq010_categories.get("Nearly every day", 0)

    dpq020_not_at_all_pct = dpq020_categories.get("Not at all", 0)
    dpq020_several_days_pct = dpq020_categories.get("Several days", 0)
    dpq020_more_than_half_pct = dpq020_categories.get("More than half the days", 0)
    dpq020_nearly_every_day_pct = dpq020_categories.get("Nearly every day", 0)

    dpq040_not_at_all_pct = dpq040_categories.get("Not at all", 0)
    dpq040_several_days_pct = dpq040_categories.get("Several days", 0)
    dpq040_more_than_half_pct = dpq040_categories.get("More than half the days", 0)
    dpq040_nearly_every_day_pct = dpq040_categories.get("Nearly every day", 0)

    # Define contribution logic based on user response and NHANES distribution
    def calculate_dpq_contribution(response, not_at_all_pct, several_days_pct, more_than_half_pct, nearly_every_day_pct):
        if response == "Not at all":
            return 0.03  # Bonus for no symptoms, adjust based on prevalence if needed
        elif response == "Several days":
            return -0.01 * several_days_pct  # Mild penalty for occasional symptoms
        elif response == "More than half the days":
            return -0.02 * more_than_half_pct  # Moderate penalty for frequent symptoms
        elif response == "Nearly every day":
            return -0.03 * nearly_every_day_pct  # Severe penalty for persistent symptoms
        elif response == "Don't know":
            return 0  # Neutral for uncertainty
        else:
            return 0  # Default to neutral if response is missing or invalid

    # Calculate contributions for each depression metric (adjusted based on NHANES prevalence)
    dpq010_contribution = calculate_dpq_contribution(dpq010, dpq010_not_at_all_pct, dpq010_several_days_pct, dpq010_more_than_half_pct, dpq010_nearly_every_day_pct)
    dpq020_contribution = calculate_dpq_contribution(dpq020, dpq020_not_at_all_pct, dpq020_several_days_pct, dpq020_more_than_half_pct, dpq020_nearly_every_day_pct)
    dpq040_contribution = calculate_dpq_contribution(dpq040, dpq040_not_at_all_pct, dpq040_several_days_pct, dpq040_more_than_half_pct, dpq040_nearly_every_day_pct)

    # Scenario-Based Depression Contributions
    scenario_depression_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Retrieve responses for depression markers
        dpq010 = scenario_data.get("Interest Loss")
        dpq020 = scenario_data.get("Feeling Depressed")
        dpq040 = scenario_data.get("Feeling Tired")

        # Retrieve NHANES benchmarks for each depression marker
        dpq010_benchmarks = categorical_benchmarks.get("DPQ010", {}).get("all_categories", {})
        dpq020_benchmarks = categorical_benchmarks.get("DPQ020", {}).get("all_categories", {})
        dpq040_benchmarks = categorical_benchmarks.get("DPQ040", {}).get("all_categories", {})

        # Get percentages for DPQ010
        dpq010_not_at_all_pct = dpq010_benchmarks.get("Not at all", 0)
        dpq010_several_days_pct = dpq010_benchmarks.get("Several days", 0)
        dpq010_more_than_half_pct = dpq010_benchmarks.get("More than half the days", 0)
        dpq010_nearly_every_day_pct = dpq010_benchmarks.get("Nearly every day", 0)

        # Get percentages for DPQ020
        dpq020_not_at_all_pct = dpq020_benchmarks.get("Not at all", 0)
        dpq020_several_days_pct = dpq020_benchmarks.get("Several days", 0)
        dpq020_more_than_half_pct = dpq020_benchmarks.get("More than half the days", 0)
        dpq020_nearly_every_day_pct = dpq020_benchmarks.get("Nearly every day", 0)

        # Get percentages for DPQ040
        dpq040_not_at_all_pct = dpq040_benchmarks.get("Not at all", 0)
        dpq040_several_days_pct = dpq040_benchmarks.get("Several days", 0)
        dpq040_more_than_half_pct = dpq040_benchmarks.get("More than half the days", 0)
        dpq040_nearly_every_day_pct = dpq040_benchmarks.get("Nearly every day", 0)

        # Contribution calculation function
        def calculate_dpq_contribution(response, not_at_all_pct, several_days_pct, more_than_half_pct, nearly_every_day_pct):
            if response == "Not at all":
                return 0.05  # Increased bonus for "Not at all"
            elif response == "Several days":
                return -0.01 * several_days_pct
            elif response == "More than half the days":
                return -0.02 * more_than_half_pct
            elif response == "Nearly every day":
                return -0.03 * nearly_every_day_pct
            elif response == "Don't know":
                return 0
            else:
                return 0

        # Calculate contributions for each depression marker
        dpq010_contribution = calculate_dpq_contribution(dpq010, dpq010_not_at_all_pct, dpq010_several_days_pct, dpq010_more_than_half_pct, dpq010_nearly_every_day_pct)
        dpq020_contribution = calculate_dpq_contribution(dpq020, dpq020_not_at_all_pct, dpq020_several_days_pct, dpq020_more_than_half_pct, dpq020_nearly_every_day_pct)
        dpq040_contribution = calculate_dpq_contribution(dpq040, dpq040_not_at_all_pct, dpq040_several_days_pct, dpq040_more_than_half_pct, dpq040_nearly_every_day_pct)

        # Total contribution for the scenario
        total_depression_contribution = round(dpq010_contribution + dpq020_contribution + dpq040_contribution, 4)

        # Store the contribution for the scenario
        scenario_depression_contributions[scenario_name] = total_depression_contribution

    dmdeduc2 = education_level
    # Initialize contributions
    nhanes_contribution = 0
    baseline_contribution = 0

    # Clean up the input value (strip any spaces or tabs)
    dmdeduc2_cleaned = dmdeduc2.strip()  # This ensures we remove any unwanted characters like tabs or spaces

    # Get the NHANES categorical benchmark for education level (from NHANES or benchmark dictionary)
    education_benchmark = categorical_benchmarks.get('DMDEDUC2', {})

    # Get the education level categories and their corresponding percentages from NHANES data
    education_categories = education_benchmark.get('all_categories', {})

    # Strip trailing tabs from the keys in education_categories for comparison
    education_categories_cleaned = {category.strip(): percentage for category, percentage in education_categories.items()}

    # Check the percentage for this education level from NHANES, after cleaning
    education_proportion = education_categories_cleaned.get(dmdeduc2_cleaned, 0)

    # Calculate NHANES contribution based on the proportion
    # Using predefined penalties and bonuses
    if dmdeduc2_cleaned == "Less than 9th grade":
        nhanes_contribution = -0.1 * education_proportion  # Stronger penalty for lowest education level
    elif dmdeduc2_cleaned == "9-11th grade (Includes 12th grade with no diploma)":
        nhanes_contribution = -0.05 * education_proportion  # Larger penalty for middle education levels
    elif dmdeduc2_cleaned == "High school graduate/GED or equivalent":
        nhanes_contribution = 0.25 * education_proportion  # Moderate penalty for high school graduate
    elif dmdeduc2_cleaned == "Some college or AA degree":
        nhanes_contribution = 0.05 * education_proportion  # Small bonus for some college
    elif dmdeduc2_cleaned == "College graduate or above":
        nhanes_contribution = 0.1 * education_proportion  # Larger bonus for college degree
    elif dmdeduc2_cleaned == "Don't know":
        nhanes_contribution = 0  # Neutral for unknown

    # Now calculate baseline contribution (based on predefined logic)
    if dmdeduc2_cleaned == "Less than 9th grade":
        baseline_contribution = -0.1  # Stronger penalty for lowest education level
    elif dmdeduc2_cleaned == "9-11th grade (Includes 12th grade with no diploma)":
        baseline_contribution = -0.05  # Larger penalty for middle education levels
    elif dmdeduc2_cleaned == "High school graduate/GED or equivalent":
        baseline_contribution = 0.25  # Moderate penalty for high school graduate
    elif dmdeduc2_cleaned == "Some college or AA degree":
        baseline_contribution = 0.05  # Small bonus for some college
    elif dmdeduc2_cleaned == "College graduate or above":
        baseline_contribution = 0.1  # Larger bonus for college degree
    elif dmdeduc2_cleaned == "Don't know":
        baseline_contribution = 0  # Neutral for unknown


    # Fine-tune the education contribution to avoid excessive weight in QALY calculation
    scaling_factor = CONFIG["education_scaling_factor"]
    scaled_nhanes_contribution = nhanes_contribution * scaling_factor
    scaled_baseline_contribution = baseline_contribution  # No sigmoid normalization for baseline

    # Normalize the NHANES contribution using sigmoid
    scaled_nhanes_contribution = sigmoid(scaled_nhanes_contribution)

    # Combine the contributions in a balanced way
    final_contribution = scaled_nhanes_contribution + scaled_baseline_contribution

    # Ensure the final contribution doesn't exceed 1.0 (if needed)
    dmdeduc2_contribution = min(final_contribution * 0.2, 1.0)

    # Define the column for poor appetite (DPQ050) and age
    dpq050_column = 'DPQ050'  # Assuming "DPQ050" column for poor appetite
    age_column = 'RIDAGEYR'  # Assuming "RIDAGEYR" for age in years

    # Group by age and calculate prevalence for poor appetite (dpq050)
    age_groups = [19, 31, 51]  # Define age groups (19, 31, 51 as mentioned earlier)

    # Filter out rows where dpq050 is "Missing" or NaN
    valid_rows = nhanes_data[nhanes_data[dpq050_column] != "Missing"]

    # Apply the grouping logic (uses module-level age_grouping)
    valid_rows['age_group'] = valid_rows[age_column].apply(age_grouping)

    # Calculate prevalence for each age group
    prevalence_data = {}
    for age_group in age_groups:
        group_data = valid_rows[valid_rows['age_group'] == age_group]
        # Calculate the proportion of each category in DPQ050 for the given age group
        group_prevalence = group_data[dpq050_column].value_counts(normalize=True)
        prevalence_data[age_group] = group_prevalence

    dpq050 = poor_appetite

    # Convert categorical responses into a numeric scale
    response_mapping = {
        "Not at all": 0,
        "Several days": 1,
        "More than half the days": 2,
        "Nearly every day": 3,
        "Don't know": None  # Neutral for uncertainty
    }

    # Sigmoid parameters
    sigmoid_center = 1.5  # Center at 1.5
    sigmoid_steepness = 4  # Steepness of the curve

    # Get numeric scale value for the response
    dpq050_numeric = response_mapping.get(dpq050)

    # Get the age group for the user
    if age_at_screening <= 30:
        age_group = 19
    elif 31 <= age_at_screening <= 50:
        age_group = 31
    else:
        age_group = 51

    # Get NHANES prevalence for the given age group (from the calculated prevalence data)
    prevalence = prevalence_data.get(age_group, {}).get(dpq050_numeric, 0.0)  # Default to 0 if no data

    # Sigmoid function
    sigmoid_score = 0  # Default neutral score
    if dpq050_numeric is not None:
        sigmoid_score = 1 - (1 / (1 + np.exp(-sigmoid_steepness * (dpq050_numeric - sigmoid_center))))

    # Adjust the sigmoid score using NHANES prevalence (correlated, not replaced)
    adjusted_score = sigmoid_score * prevalence  # Combine sigmoid and prevalence

    # Final contribution scaling
    dpq050_contribution = round((adjusted_score - 0.5) * 0.1, 4)  # Scale to fit the contribution range

    # Scenario-Based Poor Appetite Contributions
    scenario_poor_appetite_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Retrieve user response for poor appetite
        dpq050 = scenario_data.get("Poor Appetite")

       # Convert categorical responses into a numeric scale
        response_mapping = {
            "Not at all": 0,
            "Several days": 1,
            "More than half the days": 2,
            "Nearly every day": 3,
            "Don't know": None  # Neutral for uncertainty
        }
        dpq050_numeric = response_mapping.get(dpq050)

        # Determine age group for the scenario
        age_at_screening = scenario_data.get("Age", 0)
        if age_at_screening <= 30:
            age_group = 19
        elif 31 <= age_at_screening <= 50:
            age_group = 31
        else:
            age_group = 51

        # Retrieve NHANES prevalence for the age group and response
        prevalence = prevalence_data.get(age_group, {}).get(dpq050_numeric, 0.0)

       # Sigmoid parameters
        sigmoid_center = 1.5  # Center of the sigmoid
        sigmoid_steepness = 4  # Steepness of the curve

        # Sigmoid function
        sigmoid_score = 0  # Default neutral score
        if dpq050_numeric is not None:
            sigmoid_score = 1 - (1 / (1 + np.exp(-sigmoid_steepness * (dpq050_numeric - sigmoid_center))))

        # Adjust the sigmoid score using NHANES prevalence
        adjusted_score = sigmoid_score * prevalence  # Combine sigmoid and prevalence

        # Final contribution scaling
        dpq050_contribution = round((adjusted_score - 0.5) * 0.1, 4)  # Scale to fit the contribution range

        # Store the contribution for the scenario
        scenario_poor_appetite_contributions[scenario_name] = dpq050_contribution

    # Get sedentary time from widget
    sedentary_time = minutes_sedentary  # Minutes per day of sedentary activity

    # Sigmoid parameters (convert center to minutes)
    sigmoid_center = 360  # 6 hours/day in minutes
    sigmoid_steepness = 4  # Increased steepness for more significant transition

    # Retrieve NHANES benchmarks for sedentary time (from benchmarks)
    nhanes_mean, nhanes_std = benchmarks.get("PAD680", (None, None))

    # Sigmoid function
    sigmoid_score = 0  # Default neutral score
    if sedentary_time is not None and nhanes_mean is not None and nhanes_std is not None:
        # Normalize the sedentary time using NHANES mean and std
        normalized_sedentary_time = (sedentary_time - nhanes_mean) / nhanes_std

        # Apply the sigmoid function with normalized value
        sigmoid_score = 1 - (1 / (1 + np.exp(-sigmoid_steepness * normalized_sedentary_time)))

        # Adjust score based on sedentary time
        if sedentary_time < sigmoid_center:
            # Bonus for less sedentary activity (below center), with larger bonus scaling
            pad680_contribution = round((sigmoid_score - 0.5) * 0.5, 4)  # Increased bonus scaling factor
        else:
            # Penalty for more sedentary activity (above center)
            pad680_contribution = round((sigmoid_score - 0.5) * -0.3, 4)  # Increased penalty scaling factor
    else:
        pad680_contribution = 0  # Neutral if sedentary time or NHANES data is missing

    # Scenario-Based Sedentary Time Contributions
    scenario_sedentary_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Get sedentary time for the current scenario
        sedentary_time = scenario_data.get("Minutes Sedentary")

        # Retrieve NHANES benchmarks for sedentary time
        nhanes_mean, nhanes_std = benchmarks.get("PAD680", (None, None))

        # Sigmoid function parameters
        sigmoid_center = 360  # 6 hours/day in minutes
        sigmoid_steepness = 4  # Steepness of the curve

        # Default contribution
        sedentary_contribution = 0  # Neutral score if data is missing

        if sedentary_time is not None and nhanes_mean is not None and nhanes_std is not None:
            # Normalize sedentary time relative to NHANES mean
            normalized_sedentary_time = (sedentary_time - sigmoid_center) / nhanes_std

            # Apply sigmoid function
            sigmoid_score = 1 / (1 + np.exp(-sigmoid_steepness * normalized_sedentary_time))

            # Calculate contribution
            if sedentary_time < sigmoid_center:
                # Bonus for less sedentary activity (below center)
                sedentary_contribution = round((0.5 - sigmoid_score) * 0.4, 4)  # Adjusted bonus scaling
            elif sedentary_time > sigmoid_center:
                # Higher penalty for more sedentary activity (above center)
                sedentary_contribution = round((sigmoid_score - 0.5) * -0.35, 4)  # Increased penalty scaling
        else:
            sigmoid_score = 0  # Default sigmoid score for missing data

        # Store the contribution for the scenario
        scenario_sedentary_contributions[scenario_name] = sedentary_contribution

    # Get saturated fat intake and total calorie intake from widget
    saturated_fat = saturated_fats_gm  # Total saturated fatty acids (gm)
    total_calories = energy_kcal  # Total calorie intake (kcal)

    # Conversion factor: 1 gram of fat = 9 kcal
    fat_to_kcal = 9

    # Calculate 10% of total calories as the baseline for saturated fat
    if total_calories is not None:
        baseline_saturated_fat = (0.1 * total_calories) / fat_to_kcal  # In grams
    else:
        baseline_saturated_fat = None  # Neutral if calorie intake is missing

    # Get NHANES data (mean and std for saturated fat consumption)
    nhanes_mean, nhanes_std = benchmarks.get("DR1TSFAT", (None, None))

    # Adjust thresholds using NHANES data if available
    if nhanes_mean is not None and nhanes_std is not None:
        # Combine NHANES and baseline values
        nhanes_low_threshold = nhanes_mean - nhanes_std  # Lower threshold
        nhanes_high_threshold = nhanes_mean + nhanes_std  # Upper threshold
    else:
        # Fallback to baseline if NHANES data is not available
        nhanes_low_threshold = baseline_saturated_fat * 0.7  # 30% below baseline as lower threshold
        nhanes_high_threshold = baseline_saturated_fat * 1.3  # 30% above baseline as upper threshold

    # Scaling parameters
    scaling_factor = -0.15  # Penalty for exceeding high threshold
    bonus_factor = 0.1      # Bonus for being in the optimal range

    # Initialize contribution
    dr1tsfat_contribution = 0

    # Calculate contribution based on the relationship between saturated fat and baseline/NHANES thresholds
    if saturated_fat is not None and baseline_saturated_fat is not None:
        if saturated_fat <= nhanes_low_threshold:
            # No penalty for low saturated fat intake
            dr1tsfat_contribution = 0  # Neutral for very low intake
        elif nhanes_low_threshold < saturated_fat <= nhanes_high_threshold:
            # Bonus for being in the optimal range (within NHANES range or baseline)
            dr1tsfat_contribution = round(bonus_factor, 4)
        else:
            # Penalty for exceeding high threshold
            dr1tsfat_contribution = round(scaling_factor * (saturated_fat - nhanes_high_threshold) / nhanes_high_threshold, 4)

    # Scenario-Based Saturated Fat Contributions
    scenario_saturated_fat_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Retrieve data from the scenario
        saturated_fat_scenario = scenario_data.get("Saturated Fat (g)")
        total_calories_scenario = scenario_data.get("Energy Intake (kcal)")

        # Conversion factor: 1 gram of fat = 9 kcal
        fat_to_kcal_scenario = 9

        # Calculate 10% of total calories as the baseline for saturated fat
        if total_calories_scenario is not None:
            baseline_saturated_fat_scenario = (0.1 * total_calories_scenario) / fat_to_kcal_scenario  # In grams
        else:
            baseline_saturated_fat_scenario = None  # Neutral if calorie intake is missing

        # Get NHANES data (mean and std for saturated fat consumption)
        nhanes_mean_scenario, nhanes_std_scenario = benchmarks.get("DR1TSFAT", (None, None))

        # Adjust thresholds using NHANES data if available
        if nhanes_mean_scenario is not None and nhanes_std_scenario is not None:
            # Combine NHANES and baseline values
            nhanes_low_threshold_scenario = max(nhanes_mean_scenario - nhanes_std_scenario, 0)  # Lower threshold
            nhanes_high_threshold_scenario = nhanes_mean_scenario + nhanes_std_scenario  # Upper threshold
        else:
            # Fallback to baseline if NHANES data is not available
            nhanes_low_threshold_scenario = (
                baseline_saturated_fat_scenario * 0.7 if baseline_saturated_fat_scenario is not None else None
            )
            nhanes_high_threshold_scenario = (
                baseline_saturated_fat_scenario * 1.3 if baseline_saturated_fat_scenario is not None else None
            )

        # Scaling parameters
        scaling_factor_scenario = -0.15  # Penalty for exceeding baseline
        bonus_factor_scenario = 0.1     # Bonus for being in the optimal range

        # Initialize contribution
        saturated_fat_contribution_scenario = 0

        # Calculate contribution based on the relationship between saturated fat and thresholds
        if (
            saturated_fat_scenario is not None
            and baseline_saturated_fat_scenario is not None
            and nhanes_high_threshold_scenario is not None
        ):
            if saturated_fat_scenario <= nhanes_high_threshold_scenario:
                # Bonus for being in the optimal or below high threshold
                saturated_fat_contribution_scenario = round(bonus_factor_scenario, 4)
            else:
                # Penalty for exceeding the high threshold
                saturated_fat_contribution_scenario = round(
                    scaling_factor_scenario * (saturated_fat_scenario - nhanes_high_threshold_scenario) / nhanes_high_threshold_scenario, 4
                )

        # Store the contribution for the scenario
        scenario_saturated_fat_contributions[scenario_name] = saturated_fat_contribution_scenario

    # Get response from widget
    dpq100 = problem_difficulty  # "Not difficult at all", "Somewhat difficult", "Very difficult", "Extremely difficult", or "Don't know"

    # Define the column for depression-related difficulty (DPQ100) and age
    dpq100_column = 'DPQ100'  # Assuming "DPQ100" column for depression-related difficulty
    age_column = 'RIDAGEYR'  # Assuming "RIDAGEYR" for age in years

    # Filter out rows where dpq100 is "Missing" or NaN
    valid_rows = nhanes_data[nhanes_data[dpq100_column] != "Missing"]

    # Apply the grouping logic
    valid_rows['age_group'] = valid_rows[age_column].apply(age_grouping)

    # Calculate prevalence for each age group
    prevalence_data = {}
    for age_group in age_groups:
        group_data = valid_rows[valid_rows['age_group'] == age_group]
        # Calculate the proportion of each category in DPQ100 for the given age group
        group_prevalence = group_data[dpq100_column].value_counts(normalize=True)
        prevalence_data[age_group] = group_prevalence


    # Convert categorical responses into a numeric scale
    response_mapping = {
        "Not difficult at all": 0,        # Optimal response
        "Somewhat difficult": 1,          # Mild penalty
        "Very difficult": 2,              # Moderate penalty
        "Extremely difficult": 3,         # Severe penalty
        "Don't know": None                # Neutral for uncertainty
    }

    # Sigmoid parameters
    sigmoid_center = 1.5  # Center at 1.5
    sigmoid_steepness = 4  # Steepness of the curve

    # Get numeric scale value for the response
    dpq100_numeric = response_mapping.get(dpq100)

    # Get the age group for the user
    if age_at_screening <= 30:
        age_group = 19
    elif 31 <= age_at_screening <= 50:
        age_group = 31
    else:
        age_group = 51

    # Get NHANES prevalence for the given age group (from the calculated prevalence data)
    prevalence = prevalence_data.get(age_group, {}).get(dpq100_numeric, 0.0)  # Default to 0 if no data

    # Sigmoid function
    sigmoid_score = 0  # Default neutral score
    if dpq100_numeric is not None:
        sigmoid_score = 1 - (1 / (1 + np.exp(-sigmoid_steepness * (dpq100_numeric - sigmoid_center))))

    # Adjust the sigmoid score using NHANES prevalence (correlated, not replaced)
    adjusted_score = sigmoid_score * prevalence  # Combine sigmoid and prevalence

    # Final contribution scaling
    dpq100_contribution = round((adjusted_score - 0.5) * 0.1, 4)  # Scale to fit the contribution range

    # Scenario-Based DPQ100 Contributions
    scenario_dpq100_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Retrieve data from the scenario
        dpq100_response = scenario_data.get("Problem Difficulty")
        age_at_screening = scenario_data.get("Age")

        # Define the column for depression-related difficulty (DPQ100) and age
        dpq100_column = 'DPQ100'  # Assuming "DPQ100" column for depression-related difficulty
        age_column = 'RIDAGEYR'  # Assuming "RIDAGEYR" for age in years


        # Filter out rows where dpq100 is "Missing" or NaN
        valid_rows = nhanes_data[nhanes_data[dpq100_column] != "Missing"]

        # Create age group based on the 'RIDAGEYR' column
        def scenario_age_grouping(age):
            if age <= 30:
                return 19
            elif 31 <= age <= 50:
                return 31
            else:
                return 51

        # Apply the grouping logic
        valid_rows['scenario_age_group'] = valid_rows[age_column].apply(scenario_age_grouping)

        # Calculate prevalence for each age group
        scenario_prevalence_data = {}
        for age_group in age_groups:
            group_data = valid_rows[valid_rows['scenario_age_group'] == age_group]
            # Calculate the proportion of each category in DPQ100 for the given age group
            group_prevalence = group_data[dpq100_column].value_counts(normalize=True)
            scenario_prevalence_data[age_group] = group_prevalence

        # Convert categorical responses into a numeric scale
        scenario_response_mapping = {
            "Not difficult at all": 0,        # Optimal response
            "Somewhat difficult": 1,          # Mild penalty
            "Very difficult": 2,              # Moderate penalty
            "Extremely difficult": 3,         # Severe penalty
            "Don't know": None                # Neutral for uncertainty
        }

        # Sigmoid parameters
        scenario_sigmoid_center = 1.5  # Center at 1.5
        scenario_sigmoid_steepness = 4  # Steepness of the curve

        # Get numeric scale value for the response
        dpq100_numeric_value = scenario_response_mapping.get(dpq100_response)

        # Get the age group for the scenario
        if age_at_screening <= 30:
            scenario_age_group = 19
        elif 31 <= age_at_screening <= 50:
            scenario_age_group = 31
        else:
            scenario_age_group = 51

        # Get NHANES prevalence for the given age group (from the calculated prevalence data)
        scenario_prevalence = scenario_prevalence_data.get(scenario_age_group, {}).get(dpq100_numeric_value, 0.0)  # Default to 0 if no data

        # Sigmoid function
        scenario_sigmoid_score = 0  # Default neutral score
        if dpq100_numeric_value is not None:
            scenario_sigmoid_score = 1 - (1 / (1 + np.exp(-scenario_sigmoid_steepness * (dpq100_numeric_value - scenario_sigmoid_center))))

        # Adjust the sigmoid score using NHANES prevalence (correlated, not replaced)
        scenario_adjusted_score = scenario_sigmoid_score * scenario_prevalence  # Combine sigmoid and prevalence

        # Final contribution scaling
        scenario_contribution = round((scenario_adjusted_score - 0.5) * 0.1, 4)  # Scale to fit the contribution range

        # Store the contribution for the scenario
        scenario_dpq100_contributions[scenario_name] = scenario_contribution


    # Define the physical activity columns for analysis
    physical_activity_columns = ['PAQ635', 'PAQ650', 'PAQ655', 'PAQ665', 'PAQ670']

    # Filter out rows where the activity columns are missing or NaN
    valid_rows = nhanes_data_copy.dropna(subset=physical_activity_columns)

    # Apply the age grouping logic
    valid_rows['age_group'] = valid_rows['RIDAGEYR'].apply(age_grouping)

    # Calculate prevalence of physical activity responses for each age group
    prevalence_data = {}

    # Iterate over each age group to calculate prevalence for each activity
    for age_group in age_groups:
        group_data = valid_rows[valid_rows['age_group'] == age_group]

        # Calculate the distribution of responses for each activity
        paq635_distribution = group_data['PAQ635'].value_counts(normalize=True).to_dict()
        paq650_distribution = group_data['PAQ650'].value_counts(normalize=True).to_dict()
        paq655_distribution = group_data['PAQ655'].value_counts(normalize=True).to_dict()
        paq665_distribution = group_data['PAQ665'].value_counts(normalize=True).to_dict()
        paq670_distribution = group_data['PAQ670'].value_counts(normalize=True).to_dict()

        # Store the distribution in the prevalence_data dictionary
        prevalence_data[age_group] = {
            'PAQ635': paq635_distribution,
            'PAQ650': paq650_distribution,
            'PAQ655': paq655_distribution,
            'PAQ665': paq665_distribution,
            'PAQ670': paq670_distribution
        }

    # Define target times in hours per week
    moderate_target_hours = 2.5  # 150 minutes = 2.5 hours
    vigorous_target_hours = 1.25  # 75 minutes = 1.25 hours

    # Scaling parameters
    bonus_factor = 0.1  # Proportional bonus for meeting or exceeding target
    penalty_factor = -0.1  # Proportional penalty for falling short

    # Initialize contributions
    paq635_contribution = 0
    paq650_contribution = 0
    paq655_contribution = 0
    paq665_contribution = 0
    paq670_contribution = 0

    if age_at_screening <= 30:
        age_group = 19
    elif 31 <= age_at_screening <= 50:
        age_group = 31
    else:
        age_group = 51

    # Fetch the distribution (percentages) for physical activity responses from the prevalence data
    prevalence_data_for_age_group = prevalence_data.get(age_group, {})

    # PAQ635: Walking or bicycling (Yes/No)
    if walk_bicycle == "Yes":
        paq635_contribution = 0.05  # Bonus for walking or bicycling
    elif walk_bicycle == "No":
        paq635_contribution = -0.05  # Penalty for not walking or bicycling

    # PAQ650 & PAQ655: Vigorous activities (Yes/No) and days per week
    if vigorous_activities == "Yes" and days_vigorous_activities is not None:
        vigorous_time = days_vigorous_activities * 1  # Assuming 1 hour per session
        if vigorous_time >= vigorous_target_hours:
            # Bonus for meeting or exceeding vigorous activity target
            paq650_contribution = round(bonus_factor * (vigorous_time - vigorous_target_hours) / vigorous_target_hours, 4)
        else:
            # Penalty for falling short of vigorous activity target
            paq650_contribution = round(penalty_factor * (vigorous_target_hours - vigorous_time) / vigorous_target_hours, 4)

        # Adjust the contribution using NHANES prevalence data for vigorous activity
        prevalence_vigorous = prevalence_data_for_age_group.get('PAQ650', {}).get('Yes', 0.0)  # Fetching prevalence from NHANES data
        paq655_contribution = paq650_contribution * prevalence_vigorous  # Adjusted contribution based on prevalence

    # PAQ665 & PAQ670: Moderate activities (Yes/No) and days per week
    if moderate_activities == "Yes" and days_moderate_activities is not None:
        moderate_time = days_moderate_activities * 1  # Assuming 1 hour per session
        if moderate_time >= moderate_target_hours:
            # Bonus for meeting or exceeding moderate activity target
            paq665_contribution = round(bonus_factor * (moderate_time - moderate_target_hours) / moderate_target_hours, 4)
        else:
            # Penalty for falling short of moderate activity target
            paq665_contribution = round(penalty_factor * (moderate_target_hours - moderate_time) / moderate_target_hours, 4)

        # Adjust the contribution using NHANES prevalence data for moderate activity
        prevalence_moderate = prevalence_data_for_age_group.get('PAQ665', {}).get('Yes', 0.0)  # Fetching prevalence from NHANES data
        paq670_contribution = paq665_contribution * prevalence_moderate  # Adjusted contribution based on prevalence

    # Define the relative weight of the physical activity category in the QALY model
    physical_activity_weight = 0.2

    # Scenario-Based Physical Activity Contributions with Debugging and Adjustments
    scenario_physical_activity_contributions = {}

    for scenario_name, scenario_data in scenarios.items():
        # Extract scenario-specific inputs
        walk_bicycle = scenario_data.get("Walk/Bicycle")
        vigorous_activities = scenario_data.get("Vigorous Activities")
        days_vigorous_activities = scenario_data.get("Days of Vigorous Activities")
        moderate_activities = scenario_data.get("Moderate Activities")
        days_moderate_activities = scenario_data.get("Days of Moderate Activities")
        age_at_screening = scenario_data.get("Age")

        # Determine age group
        age_group = 19 if age_at_screening <= 30 else 31 if age_at_screening <= 50 else 51
        prevalence_data_for_age_group = prevalence_data.get(age_group, {})

        # Initialize base contributions
        paq635_contribution = 0.05 if walk_bicycle == "Yes" else -0.05
        paq650_contribution = 0
        paq655_contribution = 0
        paq665_contribution = 0
        paq670_contribution = 0

        # Calculate contributions for vigorous activities
        if vigorous_activities == "Yes" and days_vigorous_activities is not None:
            vigorous_time = days_vigorous_activities * 1
            if vigorous_time >= vigorous_target_hours:
                paq650_contribution = round(bonus_factor * (vigorous_time - vigorous_target_hours) / vigorous_target_hours, 4)
            else:
                paq650_contribution = round(penalty_factor * (vigorous_target_hours - vigorous_time) / vigorous_target_hours, 4)
            prevalence_vigorous = prevalence_data_for_age_group.get("PAQ650", {}).get("Yes", 0.0)
            paq655_contribution = paq650_contribution * prevalence_vigorous

        # Calculate contributions for moderate activities
        if moderate_activities == "Yes" and days_moderate_activities is not None:
            moderate_time = days_moderate_activities * 1
            if moderate_time >= moderate_target_hours:
                paq665_contribution = round(bonus_factor * (moderate_time - moderate_target_hours) / moderate_target_hours, 4)
            else:
                paq665_contribution = round(penalty_factor * (moderate_target_hours - moderate_time) / moderate_target_hours, 4)
            prevalence_moderate = prevalence_data_for_age_group.get("PAQ665", {}).get("Yes", 0.0)
            paq670_contribution = paq665_contribution * prevalence_moderate

        # Combine contributions
        contributions = {
            "PAQ635": paq635_contribution,
            "PAQ650": paq650_contribution,
            "PAQ655": paq655_contribution,
            "PAQ665": paq665_contribution,
            "PAQ670": paq670_contribution,
        }

        # Ensure contributions are weighted without distortion
        weighted_contributions = {
            key: round(value * physical_activity_weight, 4) for key, value in contributions.items()
        }

        # Store weighted contributions
        scenario_physical_activity_contributions[scenario_name] = weighted_contributions

    # Get insulin value and unit choice from widget
    insulin = insulin  # Fasting insulin value (user input)
    unit_choice = unit_choice  # "mmol" or "mg" (user input)

    # Baseline thresholds for insulin in different units
    baseline_insulin = {
        'mmol': {
            'normal_low': 2.6 * 6.945,  # Normal low in pmol/L
            'normal_high': 24.9 * 6.945,  # Normal high in pmol/L
            'hyperinsulinemia': 50 * 6.945  # Hyperinsulinemia in pmol/L
        },
        'mg': {
            'normal_low': 2.6,  # Normal low in µU/mL
            'normal_high': 24.9,  # Normal high in µU/mL
            'hyperinsulinemia': 50  # Hyperinsulinemia in µU/mL
        }
    }

    # Scaling parameters
    scaling_factor = -0.5  # Proportional penalty for deviations
    bonus_factor = 0.5     # Bonus for values in the normal range

    # Initialize insulin contribution
    insulin_contribution = 0


    # Filter for insulin levels and age groups from NHANES data
    valid_rows = nhanes_data_copy.dropna(subset=['LBXIN', 'LBDINSI', 'RIDAGEYR'])

    valid_rows['age_group'] = valid_rows['RIDAGEYR'].apply(age_grouping)

    # Calculate insulin distribution for each age group (prevalence)
    insulin_distribution = {}
    for age_group in age_groups:
        group_data = valid_rows[valid_rows['age_group'] == age_group]

        # Get insulin levels based on unit choice
        if unit_choice == 'mmol':
            insulin_column = 'LBDINSI'  # pmol/L
        elif unit_choice == 'mg':
            insulin_column = 'LBXIN'   # µU/mL
        else:
            insulin_column = None

        if insulin_column:
            insulin_distribution[age_group] = group_data[insulin_column].value_counts(normalize=True).to_dict()

    # Determine thresholds based on unit choice
    if unit_choice == "mmol":
        thresholds = baseline_insulin['mmol']
    elif unit_choice == "mg":
        thresholds = baseline_insulin['mg']
    else:
        insulin_contribution = 0  # Neutral score
        thresholds = None

    # Calculate insulin contribution based on the thresholds
    if thresholds and insulin is not None:
        if thresholds['normal_low'] <= insulin <= thresholds['normal_high']:
            # Bonus for being in the normal range
            insulin_contribution = round(bonus_factor, 4)
        elif insulin > thresholds['normal_high']:
            # Penalty for hyperinsulinemia
            insulin_contribution = round(scaling_factor * (insulin - thresholds['normal_high']) / thresholds['normal_high'], 4)
        elif insulin < thresholds['normal_low']:
            # Penalty for below-normal insulin levels
            insulin_contribution = round(scaling_factor * (thresholds['normal_low'] - insulin) / thresholds['normal_low'], 4)

        # Adjust contribution based on NHANES prevalence data for insulin in the user's age group
        age_group = age_grouping(age_at_screening)  # Assuming age_at_screening comes from the widget
        prevalence_data_for_age_group = insulin_distribution.get(age_group, {})

        # Instead of looking for the exact insulin value, we'll find the closest matching range
        closest_range = None
        for insulin_value, prevalence in prevalence_data_for_age_group.items():
            if insulin_value >= thresholds['normal_low'] and insulin_value <= thresholds['normal_high']:
                closest_range = insulin_value
                break

        if closest_range is None:
            # If no match, take the closest lower or upper bound as appropriate
            closest_range = min(prevalence_data_for_age_group, key=lambda x: abs(x - insulin))

        insulin_prevalence = prevalence_data_for_age_group.get(closest_range, 0.0)  # Get prevalence for the closest insulin value

        # Adjust insulin contribution based on prevalence
        adjusted_contribution = insulin_contribution * insulin_prevalence
        insulin_contribution = adjusted_contribution  # Final adjusted contribution

    # Scenario-Based Insulin Contributions
    scenario_insulin_contributions = {}

    # Scaling parameters
    scaling_factor = -0.25  # Proportional penalty for deviations
    bonus_factor = 0.5     # Bonus for values in the normal range

    # Iterate over each scenario
    for scenario_name, scenario_data in scenarios.items():
        # Extract scenario-specific inputs
        insulin = scenario_data.get("Insulin")  # Fasting insulin value
        unit_choice = scenario_data.get("Unit Choice")  # "mmol" or "mg"
        age_at_screening = scenario_data.get("Age")  # Age of the individual

        # Determine thresholds based on unit choice
        if unit_choice == "mmol":
            thresholds = baseline_insulin['mmol']
        elif unit_choice == "mg":
            thresholds = baseline_insulin['mg']
        else:
            scenario_insulin_contributions[scenario_name] = 0  # Neutral score
            continue

        # Initialize insulin contribution for the scenario
        insulin_contribution_scenario = 0

        # Calculate insulin contribution based on the thresholds for the scenario
        if thresholds and insulin is not None:
            # Apply bonus for being in the normal range
            if thresholds['normal_low'] <= insulin <= thresholds['normal_high']:
                insulin_contribution_scenario = bonus_factor  # Positive bonus for normal range

            # Apply penalty for hyperinsulinemia (above normal_high)
            elif insulin > thresholds['normal_high']:
                insulin_contribution_scenario = scaling_factor * (insulin - thresholds['normal_high']) / thresholds['normal_high']

            # Apply penalty for hypoinsulinemia (below normal_low)
            elif insulin < thresholds['normal_low']:
                insulin_contribution_scenario = scaling_factor * (thresholds['normal_low'] - insulin) / thresholds['normal_low']

        # Store the scenario insulin contribution
        scenario_insulin_contributions[scenario_name] = insulin_contribution_scenario

    # Get alcohol consumption from the widget
    alcohol_frequency = alcohol_frequency  # Number of times 4-5 drinks were consumed in the past 30 days
    gender = gender  # "Male", "Female", or "Prefer not to say"

    # Define risk thresholds for drinking limits (in terms of drinks per week)
    if gender == "Male":
        moderate_drinking_limit = 14  # 14 drinks/week for men (moderate)
    elif gender == "Female":
        moderate_drinking_limit = 7   # 7 drinks/week for women (moderate)
    else:
        moderate_drinking_limit = 7  # Default to female threshold if gender is unknown

    # Conversion for 30-day period (approximately 4 weeks in a month)
    monthly_limit = moderate_drinking_limit * 4  # Approximate 4 weeks in a month

    # Scaling parameters
    scaling_factor = -0.15  # Penalty for exceeding the monthly limit

    # Initialize contribution
    alcohol_contribution = 0

    # Convert alcohol frequency (in days) into total drinks (4.5 drinks on average per occasion)
    total_drinks = alcohol_frequency * 4.5  # Average 4.5 drinks on each occasion

    # Check if the total number of drinks exceeds the monthly limit
    if total_drinks > monthly_limit:
        # Penalty for exceeding the monthly limit in drinks
        alcohol_contribution = round(scaling_factor * (total_drinks - monthly_limit) / monthly_limit, 4)
    else:
        # No penalty if alcohol consumption is within the limit
        alcohol_contribution = 0

    # Scenario-Based Alcohol Consumption Contributions
    scenario_alcohol_contributions = {}

    # Scaling parameters
    scaling_factor = -0.15  # Penalty for exceeding the monthly limit

    # Iterate over each scenario
    for scenario_name, scenario_data in scenarios.items():
        # Extract scenario-specific inputs
        alcohol_frequency = scenario_data.get("Alcohol Frequency")  # Number of times 4-5 drinks were consumed in the past 30 days
        gender = scenario_data.get("Gender")  # "Male", "Female", or "Prefer not to say"

        # Define risk thresholds for drinking limits (in terms of drinks per week)
        if gender == "Male":
            moderate_drinking_limit = 14  # 14 drinks/week for men (moderate)
        elif gender == "Female":
            moderate_drinking_limit = 7   # 7 drinks/week for women (moderate)
        else:
            moderate_drinking_limit = 7  # Default to female threshold if gender is unknown

        # Conversion for 30-day period (approximately 4 weeks in a month)
        monthly_limit = moderate_drinking_limit * 4  # Approximate 4 weeks in a month

        # Convert alcohol frequency (in days) into total drinks (4.5 drinks on average per occasion)
        total_drinks = alcohol_frequency * 4.5  # Average 4.5 drinks on each occasion

        # Initialize contribution for the scenario
        alcohol_contribution_scenario = 0

        # Check if the total number of drinks exceeds the monthly limit
        if total_drinks > monthly_limit:
            # Penalty for exceeding the monthly limit in drinks
            alcohol_contribution_scenario = round(scaling_factor * (total_drinks - monthly_limit) / monthly_limit, 4)
        else:
            # No penalty if alcohol consumption is within the limit
            alcohol_contribution_scenario = 0

        # Store the scenario alcohol contribution
        scenario_alcohol_contributions[scenario_name] = alcohol_contribution_scenario

    # Get diet responses from the widget
    special_diet = on_special_diet  # "Yes", "No", or "Don't know"
    diet_type = diet_type  # Specific diet type if "Yes"

    # Baseline contributions for each diet type
    baseline_diet_contributions = {
        "Weight loss/Low calorie diet": 0.05,
        "Sugar free/Low sugar diet": 0.05,
        "Low fiber diet": -0.05,
        "Low fat/Low cholesterol diet": 0.03,
        "Low salt/Low sodium diet": 0.05,
        "High fiber diet": 0.05,
        "Diabetic diet": 0.0,
        "Weight gain/Muscle building diet": 0.03,
        "Low carbohydrate diet": 0.03,
        "High protein diet": 0.03,
        "Other special diet": 0.0  # Default for unlisted diets
    }

    # Initialize baseline contribution for special diet
    diet_contribution = 0

    # Filter for valid rows based on NHANES data (age, gender, and special diet)
    valid_rows = nhanes_data_copy.dropna(subset=['RIDAGEYR', 'RIAGENDR', 'DRQSDIET'])
    valid_rows['age_group'] = valid_rows['RIDAGEYR'].apply(age_grouping)

    # Calculate diet prevalence for each age group and gender
    diet_prevalence = {}
    for age_group in [19, 31, 51]:
        group_data = valid_rows[valid_rows['age_group'] == age_group]
        diet_distribution = group_data['DRQSDIET'].value_counts(normalize=True).to_dict()
        diet_prevalence[age_group] = diet_distribution

    # Calculate baseline contribution for the selected diet type
    if special_diet in ["No", "Don't know"]:
        diet_contribution = 0  # No contribution if no special diet or unknown
    else:
        # Get baseline contribution for the selected diet type
        diet_contribution = baseline_diet_contributions.get(diet_type, 0)

        # Apply adjustments (both positive and negative) for specific diets
        if diet_type == "Weight loss/Low calorie diet":
            if bmi is not None:  # Ensure BMI is valid
                if bmi > 30:  # Obesity
                    diet_contribution += 0.03  # Bonus for obesity
                elif bmi < 18.5:  # Underweight
                    diet_contribution -= 0.1  # Penalty for being underweight
        elif diet_type == "Sugar free/Low sugar diet":
            if diabetes_diagnosis == "Yes":
                diet_contribution += 0.05  # Bonus for diabetic patients
            if total_calories and total_calories < 1200:  # Low energy intake
                diet_contribution -= 0.02  # Penalty for low energy intake
        elif diet_type == "Low fiber diet":
            if bmi and bmi < 18.5 and dietary_fiber_gm and dietary_fiber_gm < 10:  # Severe GI issues proxy
                diet_contribution -= 0.05  # Penalty for potential risk
        elif diet_type == "High fiber diet":
            if high_blood_pressure_history == "Yes" or high_cholesterol_diagnosis == "Yes":
                diet_contribution += 0.03  # Cardiovascular benefits
        elif diet_type == "Low fat/Low cholesterol diet":
            if total_cholesterol and total_cholesterol > 200:  # High LDL cholesterol
                diet_contribution += 0.05
            if saturated_fats_gm and total_calories:  # Ensure both values are valid
                if saturated_fats_gm < 0.2 * total_calories:  # Fat intake < 20%
                    diet_contribution -= 0.02
        elif diet_type == "Low salt/Low sodium diet":
            if high_blood_pressure_history == "Yes":  # Hypertension
                diet_contribution += 0.05
        elif diet_type == "Diabetic diet":
            if diabetes_diagnosis == "Yes":
                diet_contribution += 0.05
        elif diet_type == "Weight gain/Muscle building diet":
            if bmi and bmi < 18.5:  # Underweight
                diet_contribution += 0.05
            if bmi and bmi > 25:  # Overweight
                diet_contribution -= 0.03
        elif diet_type == "Low carbohydrate diet":
            if diabetes_diagnosis == "Yes":
                diet_contribution += 0.05
            if saturated_fats_gm and total_calories:  # Ensure both values are valid
                if saturated_fats_gm > 0.1 * total_calories:  # Saturated fat > 10%
                    diet_contribution -= 0.02
        elif diet_type == "High protein diet":
            if vigorous_activities == "Yes" or moderate_activities == "Yes":  # Active lifestyle
                diet_contribution += 0.03
            if insulin and insulin > 24.9:  # Kidney strain for CKD patients
                diet_contribution -= 0.03

        # Apply NHANES data for diet prevalence adjustments in age and gender
        age_group = age_grouping(age_at_screening)  # Get user's age group from the widget
        prevalence_data_for_age_group = diet_prevalence.get(age_group, {})

        # Find prevalence for the selected diet type
        diet_prevalence_value = prevalence_data_for_age_group.get(diet_type, 0.0)

        # Apply the prevalence adjustment if prevalence > 0
        if diet_prevalence_value > 0:
            # Adjust the contribution based on the prevalence of the selected diet type
            diet_contribution *= diet_prevalence_value
        else:
            # If no prevalence data, keep earlier contribution
            pass

    # Apply categorical mappings to nhanes_data so row-wise contribution apply sees "Yes"/"No" (not 1/2)
    for col in ["BPQ020", "BPQ050A", "BPQ080", "DIQ010", "DIQ070", "DRQSDT1"]:
        if col in nhanes_data.columns and col in categorical_mappings and categorical_mappings[col] is not None:
            nhanes_data[col] = pd.to_numeric(nhanes_data[col], errors="coerce").replace(categorical_mappings[col]).fillna("Missing")

    # Define the column names for contributions (insulin_contribution listed once only)
    contribution_columns = [
        'hdl_contribution', 'tc_contribution', 'glucose_contribution', 'insulin_contribution',
        'bmi_contribution', 'waist_contribution', 'fiber_contribution', 'calorie_contribution',
        'bpq020_contribution', 'bpq050a_contribution', 'bpq080_contribution', 'diq010_contribution',
        'diq070_contribution', 'dpq010_contribution', 'dpq020_contribution', 'dpq040_contribution',
        'dpq050_contribution', 'dpq100_contribution', 'paq635_contribution', 'paq650_contribution',
        'paq665_contribution', 'alcohol_contribution', 'diet_contribution'
    ]

    # Specify columns to ignore (like SEQN, Age, Gender)
    columns_to_ignore = ["SEQN", "RIDAGEYR", "RIAGENDR", "DR1TALCO", "DRQSDT11", "DRQSDT12"]

    # Filter out the columns to ignore
    filtered_contribution_columns = [col for col in contribution_columns if col not in columns_to_ignore]


    # Apply the contribution calculations directly
    for col in filtered_contribution_columns:
        nhanes_data[col] = nhanes_data.apply(lambda row:
            (
                # HDL Contribution
                row['LBDHDDSI'] * 0.01 if col == 'hdl_contribution' else
                # Total Cholesterol Contribution
                row['LBDTCSI'] * -0.01 if col == 'tc_contribution' else
                # Fasting Glucose Contribution
                row['LBDGLUSI'] * -0.01 if col == 'glucose_contribution' else
                # Insulin Contribution
                row['LBDINSI'] * 0.0001 if col == 'insulin_contribution' else
                # BMI Contribution
                row['BMXBMI'] * 0.002 if col == 'bmi_contribution' else
                # Waist Circumference Contribution
                row['BMXWAIST'] * 0.001 if col == 'waist_contribution' else
                # Dietary Fiber Contribution
                row['DR1TFIBE'] * 0.01 if col == 'fiber_contribution' else
                # Calorie Intake Contribution
                row['DR1TKCAL'] * -0.0001 if col == 'calorie_contribution' else
                # BPQ020 Contribution (history of high blood pressure)
                0.1 if col == 'bpq020_contribution' and pd.notna(row['BPQ020']) and row['BPQ020'] == 'Yes' else 0 if col == 'bpq020_contribution' else
                # BPQ050A Contribution (on prescribed medicine for HBP)
                0.05 if col == 'bpq050a_contribution' and pd.notna(row['BPQ050A']) and row['BPQ050A'] == 'Yes' else 0 if col == 'bpq050a_contribution' else
                # BPQ080 Contribution (history of high cholesterol diagnosis)
                0.175 if col == 'bpq080_contribution' and pd.notna(row['BPQ080']) and row['BPQ080'] == 'Yes' else 0 if col == 'bpq080_contribution' else
                # DIQ010 Contribution (history of diabetes diagnosis)
                0.1 if col == 'diq010_contribution' and pd.notna(row['DIQ010']) and row['DIQ010'] == 'Yes' else 0 if col == 'diq010_contribution' else
                # DIQ070 Contribution (taking diabetic pills)
                0 if col == 'diq070_contribution' and pd.notna(row['DIQ070']) and row['DIQ070'] == 'No' else 0 if col == 'diq070_contribution' else
                # DPQ010 Contribution (interest in doing things)
                pd.to_numeric(row['DPQ010'], errors='coerce') * -0.01 if col == 'dpq010_contribution' else
                # DPQ020 Contribution (feeling down, depressed, or hopeless)
                pd.to_numeric(row['DPQ020'], errors='coerce') * 0.05 if col == 'dpq020_contribution' else
                # DPQ040 Contribution (feeling tired or low energy)
                pd.to_numeric(row['DPQ040'], errors='coerce') * 0 if col == 'dpq040_contribution' else
                # DPQ050 Contribution (poor appetite or overeating)
                pd.to_numeric(row['DPQ050'], errors='coerce') * 0.05 if col == 'dpq050_contribution' else
                # DPQ100 Contribution (difficulty caused by depression symptoms)
                pd.to_numeric(row['DPQ100'], errors='coerce') * -0.05 if col == 'dpq100_contribution' else
                # PAQ635 Contribution (walking or bicycling)
                pd.to_numeric(row['PAQ635'], errors='coerce') * 0.02 if col == 'paq635_contribution' else
                # PAQ650 Contribution (vigorous recreational activities)
                pd.to_numeric(row['PAQ650'], errors='coerce') * 0.03 if col == 'paq650_contribution' else
                # PAQ665 Contribution (moderate recreational activities)
                pd.to_numeric(row['PAQ665'], errors='coerce') * 0.04 if col == 'paq665_contribution' else
                # Insulin Contribution
                pd.to_numeric(row['LBDINSI'], errors='coerce') * 0.0001 if col == 'insulin_contribution' else
                # Alcohol Contribution (past 30 days # of 4-5 drink occasions)
                pd.to_numeric(row['ALQ170'], errors='coerce') * 0.005 if col == 'alcohol_contribution' else
                # Special Diet Contribution (Weight loss/Low calorie diet)
                0.08 if col == 'diet_contribution' and pd.notna(row['DRQSDT1']) and row['DRQSDT1'] == 'Yes' else 0  # Change for other diets if needed
            ), axis=1)

    # Convert contribution columns to numeric, replacing non-numeric values with NaN and then filling with 0
    for col in filtered_contribution_columns:
        # Convert values like '1.0' and '2.0' to float numbers
        nhanes_data[col] = nhanes_data[col].apply(lambda x: float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x)

        # Now apply pd.to_numeric to convert any remaining non-numeric values
        nhanes_data[col] = pd.to_numeric(nhanes_data[col], errors='coerce').fillna(0)

    # Now you can proceed to calculate the utility score
    nhanes_data['utility_score'] = nhanes_data[filtered_contribution_columns].sum(axis=1)

    # Initialize the utility score to 1.0 (ideal health state)
    utility_score = 1.0
    print(f"Initial Utility Score: {utility_score}")

    # Initialize warning list
    warnings = []
    min_utility = CONFIG["utility"]["min_score"]
    max_utility = CONFIG["utility"]["max_score"]
    w = CONFIG["weights_negative"]

    # HDL Contribution: Should reward high HDL (good cholesterol)
    if hdl_contribution < 0:
        utility_score += hdl_contribution * w["hdl"]
    print(f"After HDL Contribution: {utility_score} (HDL: {hdl_contribution})")

    # Total Cholesterol Contribution: Penalize high total cholesterol
    if tc_contribution < 0:
        utility_score += tc_contribution * w["total_cholesterol"]
    print(f"After Total Cholesterol Contribution: {utility_score} (TC: {tc_contribution})")

    # Fasting Glucose Contribution: Penalize high fasting glucose levels
    if glucose_contribution < 0:
        utility_score += glucose_contribution * w["glucose"]
    print(f"After Fasting Glucose Contribution: {utility_score} (Glucose: {glucose_contribution})")

    # BMI Contribution: Penalize high BMI (obesity)
    if bmi_contribution < 0:
        utility_score += bmi_contribution * w["bmi"]
    print(f"After BMI Contribution: {utility_score} (BMI: {bmi_contribution})")

    # Waist Circumference: Penalize high waist circumference
    if waist_contribution < 0:
        utility_score += waist_contribution * w["waist"]
    print(f"After Waist Circumference Contribution: {utility_score} (Waist: {waist_contribution})")

    # High Blood Pressure (BPQ020) Contribution: Increase penalty for untreated high BP
    if bpq020_contribution < 0 and bpq050a_contribution < 0:
        utility_score += bpq020_contribution * w["bp_untreated"]
        warnings.append("Warning: You have high blood pressure but are not taking medication. Consider consulting a healthcare professional.")
    elif bpq020_contribution < 0 and bpq050a_contribution > 0:
        utility_score += bpq020_contribution * w["bp_treated"]
        warnings.append("Good job: You are managing your high blood pressure with medication. Keep up the good work.")
    print(f"After High Blood Pressure Contribution: {utility_score} (BPQ020: {bpq020_contribution}, BPQ050A: {bpq050a_contribution})")

    # High Cholesterol History (BPQ080): No change unless specific conditions are identified
    utility_score += bpq080_contribution
    print(f"After High Cholesterol History Contribution: {utility_score} (BPQ080: {bpq080_contribution})")

    # Diabetes History (DIQ010) and Medication (DIQ070): Penalize if untreated
    if diq010_contribution < 0 and diq070_contribution < 0:
        utility_score += diq010_contribution * w["diabetes_untreated"]
        warnings.append("Warning: You have diabetes but are not taking medication. Please consult a healthcare provider.")
    elif diq010_contribution < 0 and diq070_contribution > 0:
        utility_score += diq010_contribution * w["diabetes_treated"]
        warnings.append("Good job: You are managing your diabetes with medication. Keep up the good work.")
    elif diq010_contribution > 0:
        warnings.append("Good news: No diabetes detected. Keep monitoring your health.")
    print(f"After Diabetes Contribution: {utility_score} (DIQ010: {diq010_contribution}, DIQ070: {diq070_contribution})")

    # Depression Symptoms (DPQ010, DPQ020): Penalize for depression
    utility_score += dpq010_contribution
    print(f"After Depression Contribution (DPQ010): {utility_score} (DPQ010: {dpq010_contribution})")

    utility_score += dpq020_contribution
    print(f"After Depression Contribution (DPQ020): {utility_score} (DPQ020: {dpq020_contribution})")

    # Sedentary Time (PAD680): Penalize more for high sedentary time
    if pad680_contribution < 0:
        utility_score += pad680_contribution * w["sedentary"]
        warnings.append("Warning: You have too much sedentary time. Try to increase physical activity.")
    print(f"After Sedentary Time Contribution: {utility_score} (PAD680: {pad680_contribution})")

    # Fiber Intake, Calorie Intake, Alcohol, and Diet: Reward positive contributions and penalize negatives
    utility_score += fiber_contribution
    print(f"After Fiber Contribution: {utility_score} (Fiber: {fiber_contribution})")

    utility_score += calorie_contribution
    print(f"After Calorie Contribution: {utility_score} (Calorie Intake: {calorie_contribution})")

    utility_score += alcohol_contribution
    print(f"After Alcohol Contribution: {utility_score} (Alcohol: {alcohol_contribution})")

    # Physical Activity Contributions (PAQ635, PAQ650, etc.): Reward activity
    utility_score += paq635_contribution
    print(f"After Walking/Bicycling Contribution: {utility_score} (PAQ635: {paq635_contribution})")

    utility_score += paq650_contribution
    print(f"After Vigorous Activity Contribution: {utility_score} (PAQ650: {paq650_contribution})")

    utility_score += paq665_contribution
    print(f"After Moderate Activity Contribution: {utility_score} (PAQ665: {paq665_contribution})")

    # Special Diet: Reward if on a beneficial diet (e.g., low-calorie diet)
    utility_score += diet_contribution
    print(f"After Diet Contribution: {utility_score} (Diet: {diet_contribution})")

    # Additional Contributions (missing in prior snippet)
    utility_score += vitamin_c_contribution
    print(f"After Vitamin C Contribution: {utility_score} (Vitamin C: {vitamin_c_contribution})")

    utility_score += dpq040_contribution  # Contribution for feeling tired or low energy
    print(f"After Tired/Low Energy Contribution: {utility_score} (DPQ040: {dpq040_contribution})")

    utility_score += dmdeduc2_contribution  # Contribution for education level
    print(f"After Education Level Contribution: {utility_score} (Deduc2: {dmdeduc2_contribution})")

    utility_score += dpq050_contribution  # Contribution for poor appetite or overeating
    print(f"After Poor Appetite Contribution: {utility_score} (DPQ050: {dpq050_contribution})")

    utility_score += dr1tsfat_contribution  # Contribution for total saturated fatty acids
    print(f"After Saturated Fat Contribution: {utility_score} (Total Saturated Fats: {dr1tsfat_contribution})")

    utility_score += dpq100_contribution  # Contribution for depression symptoms difficulty
    print(f"After Depression Difficulty Contribution: {utility_score} (DPQ100: {dpq100_contribution})")

    utility_score += paq655_contribution  # Contribution for vigorous activity days
    print(f"After Vigorous Activity Days Contribution: {utility_score} (PAQ655: {paq655_contribution})")

    utility_score += paq670_contribution  # Contribution for moderate activity days
    print(f"After Moderate Activity Days Contribution: {utility_score} (PAQ670: {paq670_contribution})")

    if unit_choice != "Don't know":
        utility_score += insulin_contribution  # Contribution for fasting insulin (if unit is known)
        print(f"After Insulin Contribution: {utility_score} (Insulin: {insulin_contribution})")

    # Apply caps from config
    utility_score = max(min_utility, min(max_utility, utility_score))

    # Final Utility Score
    print(f"Final Utility Score after adjustments: {utility_score}")

    benchmark_age = CONFIG["utility"]["benchmark_age"]
    remaining_years = max(0, benchmark_age - age_at_screening)

    # Calculate QALY over remaining years
    qaly = round(utility_score * remaining_years, 2)

    # Output the results
    print(f"Utility Score: {round(utility_score, 4)}")
    print(f"Estimated remaining years of life: {remaining_years}")
    print(f"Quality-Adjusted Life Years (QALY): {qaly}")

    # Save for API return (before any overwrites below)
    final_utility_score = utility_score
    final_qaly = qaly
    main_warnings = list(warnings)

    # Provide personalized feedback based on the utility score
    if utility_score >= 0.8:
        print("Your health is in great shape! Keep maintaining your current habits.")
    elif 0.5 <= utility_score < 0.8:
        print("Your health is average. Consider making improvements in lifestyle.")
    else:
        print("Your health needs attention. Consult a healthcare professional.")

    # Print warnings at the end of the output
    if warnings:
        for warning in warnings:
            print(warning)

    # Initialize the base utility score
    utility_score = 1.0
    cumulative_scores = [utility_score]

    # Dynamic contribution calculations using the provided functions or variables
    contributions = {
        "HDL": hdl_contribution * w["hdl"] if hdl_contribution < 0 else hdl_contribution,
        "Total Cholesterol": tc_contribution * w["total_cholesterol"] if tc_contribution < 0 else tc_contribution,
        "Fasting Glucose": glucose_contribution * w["glucose"] if glucose_contribution < 0 else glucose_contribution,
        "BMI": bmi_contribution * w["bmi"] if bmi_contribution < 0 else bmi_contribution,
        "Waist Circumference": waist_contribution * w["waist"] if waist_contribution < 0 else waist_contribution,
        "High Blood Pressure": bpq020_contribution * w["bp_untreated"] if bpq020_contribution < 0 else bpq020_contribution,
        "High Cholesterol History": bpq080_contribution,
        "Diabetes History": diq010_contribution * w["diabetes_untreated"] if diq010_contribution < 0 else diq010_contribution,
        "Depression (DPQ010)": dpq010_contribution,
        "Depression (DPQ020)": dpq020_contribution * 1.2 if dpq020_contribution < 0 else dpq020_contribution,
        "Sedentary Time": pad680_contribution * w["sedentary"] if pad680_contribution < 0 else pad680_contribution,
        "Fiber": fiber_contribution,
        "Calories": calorie_contribution,
        "Alcohol": alcohol_contribution,
        "Walking/Bicycling": paq635_contribution,
        "Vigorous Activity": paq650_contribution,
        "Moderate Activity": paq665_contribution,
        "Special Diet": diet_contribution,
        "Vitamin C": vitamin_c_contribution,
        "Low Energy": dpq040_contribution * 1.2 if dpq040_contribution < 0 else dpq040_contribution,
        "Education Level": dmdeduc2_contribution,
        "Poor Appetite": dpq050_contribution * 1.1 if dpq050_contribution < 0 else dpq050_contribution,
        "Saturated Fat": dr1tsfat_contribution,
        "Depression Difficulty": dpq100_contribution,
        "Vigorous Activity Days": paq655_contribution,
        "Moderate Activity Days": paq670_contribution,
        "Fasting Insulin": insulin_contribution,
    }



    # Convert contributions into arrays for plotting (one bar per label, explicitly aligned)
    parameters = list(contributions.keys())
    values = list(contributions.values())
    n_bars = len(parameters)
    y_pos = np.arange(n_bars)

    # Bar Chart for Contribution Breakdown: explicit y-positions and yticks so every bar has one label
    fig_height = max(8, n_bars * 0.32)
    plt.figure(figsize=(12, fig_height))
    colors = ['green' if v > 0 else 'red' for v in values]
    plt.barh(y_pos, values, color=colors, height=0.8)
    plt.yticks(y_pos, parameters, fontsize=9)
    plt.xlabel('Contribution Value')
    plt.title('Utility Score Contribution Breakdown')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.ylim(y_pos[0] - 0.5, y_pos[-1] + 0.5)
    plt.tight_layout()
    if not api_mode:
        plt.show()

    # Cumulative Utility Score Calculation
    # Apply contributions dynamically
    #print(f"Initial Utility Score: {utility_score:.3f}")
    for key, value in contributions.items():
        # Apply weighted logic only to negative contributions
        adjusted_value = value * 1.4 if value < 0 else value
        utility_score += adjusted_value
        utility_score = max(0.1, utility_score)  # Cap at 0.1 to prevent negative scores
        cumulative_scores.append(utility_score)
        #print(f"After {key} Contribution: {utility_score:.3f} ({key}: {adjusted_value:.4f})")

    # Plot cumulative utility score progression
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(cumulative_scores)), cumulative_scores, marker='o', linestyle='-', color='blue')
    plt.xticks(range(len(cumulative_scores)), ["Initial"] + list(contributions.keys()), rotation=90)
    plt.ylabel('Utility Score')
    plt.title('Cumulative Utility Score Progression')
    plt.grid()
    plt.tight_layout()
    if not api_mode:
        plt.show()

    #Tornado Chart for Parameter Sensitivity
    sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    sorted_parameters = [p[0] for p in sorted_contributions]
    sorted_values = [p[1] for p in sorted_contributions]
    plt.figure(figsize=(10, 8))
    plt.barh(sorted_parameters, sorted_values, color=['green' if v > 0 else 'red' for v in sorted_values])
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Contribution Value')
    plt.title('Parameter Sensitivity Analysis (Tornado Chart)')
    plt.tight_layout()
    if not api_mode:
        plt.show()

    # Initialize the utility score to 1.0 (ideal health state)
    scenario_utility_score = 1.0
    print(f"Initial Utility Score (Scenario): {scenario_utility_score}")

    # Define the caps
    # Initialize warning list (caps already set above: min_utility, max_utility)
    warnings = []

    scenario_qaly_scores = {}

    # Define the scenarios
    scenarios = ['best_case', 'worst_case', 'middle_case_1', 'middle_case_2']

    scenario_w = CONFIG["weights_negative"]
    for scenario_name in scenarios:
        scenario_utility_score = 1.0
        print(f"\n\n{'-'*50}")
        print(f"Utility Calculation for {scenario_name} Scenario:")
        print(f"{'-'*50}\n")

        if scenario_hdl_contributions[scenario_name] < 0:
            scenario_utility_score += scenario_hdl_contributions[scenario_name] * scenario_w["hdl"]
        print(f"After HDL Contribution (Scenario): {scenario_utility_score} (HDL: {scenario_hdl_contributions[scenario_name]})")

        if scenario_tc_contributions[scenario_name] < 0:
            scenario_utility_score += scenario_tc_contributions[scenario_name] * scenario_w["total_cholesterol"]
        print(f"After Total Cholesterol Contribution (Scenario): {scenario_utility_score} (TC: {scenario_tc_contributions[scenario_name]})")

        if scenario_glucose_contributions[scenario_name] < 0:
            scenario_utility_score += scenario_glucose_contributions[scenario_name] * scenario_w["glucose"]
        print(f"After Fasting Glucose Contribution (Scenario): {scenario_utility_score} (Glucose: {scenario_glucose_contributions[scenario_name]})")

        if scenario_bmi_contributions[scenario_name] < 0:
            scenario_utility_score += scenario_bmi_contributions[scenario_name] * scenario_w["bmi"]
        print(f"After BMI Contribution (Scenario): {scenario_utility_score} (BMI: {scenario_bmi_contributions[scenario_name]})")

        # Waist Circumference: Penalize high waist circumference
        if scenario_waist_contributions[scenario_name] < 0:
            scenario_utility_score += scenario_waist_contributions[scenario_name] * scenario_w["waist"]
        print(f"After Waist Circumference Contribution (Scenario): {scenario_utility_score} (Waist: {scenario_waist_contributions[scenario_name]})")

        # Fiber Intake Contribution: Reward positive contributions
        scenario_utility_score += scenario_fiber_contributions[scenario_name]
        print(f"After Fiber Contribution (Scenario): {scenario_utility_score} (Fiber: {scenario_fiber_contributions[scenario_name]})")

        # Vitamin C Contribution: Reward positive contributions
        scenario_utility_score += scenario_vitamin_c_contributions[scenario_name]
        print(f"After Vitamin C Contribution (Scenario): {scenario_utility_score} (Vitamin C: {scenario_vitamin_c_contributions[scenario_name]})")

        # Calorie Intake Contribution: Reward positive contributions
        scenario_utility_score += scenario_calorie_contributions[scenario_name]
        print(f"After Calorie Contribution (Scenario): {scenario_utility_score} (Calorie Intake: {scenario_calorie_contributions[scenario_name]})")

        # Blood Pressure Contribution: Penalize high blood pressure
        if scenario_bp_contributions[scenario_name] < 0:
            scenario_utility_score += scenario_bp_contributions[scenario_name] * scenario_w["bp_untreated"]
        print(f"After Blood Pressure Contribution (Scenario): {scenario_utility_score} (BP: {scenario_bp_contributions[scenario_name]})")

        # High Cholesterol History Contribution: No change unless specific conditions are identified
        scenario_utility_score += scenario_bpq080_contributions[scenario_name]
        print(f"After High Cholesterol History Contribution (Scenario): {scenario_utility_score} (BPQ080: {scenario_bpq080_contributions[scenario_name]})")

        # Diabetes History Contribution: Penalize untreated diabetes
        if scenario_diabetes_contributions[scenario_name] < 0:
            scenario_utility_score += scenario_diabetes_contributions[scenario_name] * scenario_w["diabetes_untreated"]
            warnings.append("Warning: You have diabetes but are not taking medication. Please consult a healthcare provider.")
        print(f"After Diabetes Contribution (Scenario): {scenario_utility_score} (DIABETES: {scenario_diabetes_contributions[scenario_name]})")

        # Depression Contribution: Penalize for depression
        scenario_utility_score += scenario_depression_contributions[scenario_name]
        print(f"After Depression Contribution (Scenario): {scenario_utility_score} (Depression: {scenario_depression_contributions[scenario_name]})")

        # Poor Appetite Contribution: Penalize for poor appetite
        scenario_utility_score += scenario_poor_appetite_contributions[scenario_name]
        print(f"After Poor Appetite Contribution (Scenario): {scenario_utility_score} (Appetite: {scenario_poor_appetite_contributions[scenario_name]})")

        # Sedentary Time Contribution: Penalize more for high sedentary time
        if scenario_sedentary_contributions[scenario_name] < 0:  # If sedentary time is high
            scenario_utility_score += scenario_sedentary_contributions[scenario_name] * scenario_w["sedentary"]
            warnings.append("Warning: You have too much sedentary time. Try to increase physical activity.")
        print(f"After Sedentary Time Contribution (Scenario): {scenario_utility_score} (Sedentary: {scenario_sedentary_contributions[scenario_name]})")

        # Saturated Fat Contribution: Penalize if intake is high
        scenario_utility_score += scenario_saturated_fat_contributions[scenario_name]
        print(f"After Saturated Fat Contribution (Scenario): {scenario_utility_score} (Saturated Fats: {scenario_saturated_fat_contributions[scenario_name]})")

        # DPQ100 Contribution: Penalize for difficulty in depression symptoms
        scenario_utility_score += scenario_dpq100_contributions[scenario_name]
        print(f"After Depression Difficulty Contribution (Scenario): {scenario_utility_score} (DPQ100: {scenario_dpq100_contributions[scenario_name]})")

        # Physical Activity Contributions: Reward activity
        for activity, contribution in scenario_physical_activity_contributions[scenario_name].items():
            scenario_utility_score += contribution
            print(f"After {activity} Contribution (Scenario): {scenario_utility_score} (Physical Activity: {contribution})")

        # Insulin Contribution: Penalize or reward based on insulin levels
        scenario_utility_score += scenario_insulin_contributions[scenario_name]
        print(f"After Insulin Contribution (Scenario): {scenario_utility_score} (Insulin: {scenario_insulin_contributions[scenario_name]})")

        # Alcohol Consumption Contribution: Penalize for high alcohol consumption
        scenario_utility_score += scenario_alcohol_contributions[scenario_name]
        print(f"After Alcohol Consumption Contribution (Scenario): {scenario_utility_score} (Alcohol: {scenario_alcohol_contributions[scenario_name]})")

        scenario_utility_score += diet_contribution
        print(f"After Diet Contribution: {scenario_utility_score} (Diet: {diet_contribution})")

        scenario_utility_score += dmdeduc2_contribution  # Contribution for education level
        print(f"After Education Level Contribution: {scenario_utility_score} (Deduc2: {dmdeduc2_contribution})")


        # Apply caps from config
        scenario_utility_score = max(CONFIG["utility"]["min_score"], min(CONFIG["utility"]["max_score"], scenario_utility_score))

        # Final Utility Score
        print(f"Final Utility Score (Scenario) after adjustments: {scenario_utility_score}")

        remaining_years = max(0, CONFIG["utility"]["benchmark_age"] - age_at_screening)

        # Calculate QALY over remaining years
        qaly_scenario = round(scenario_utility_score * remaining_years, 2)

        # Output the results
        print(f"Utility Score (Scenario): {round(scenario_utility_score, 4)}")
        print(f"Estimated remaining years of life: {remaining_years}")
        print(f"Quality-Adjusted Life Years (QALY): {qaly_scenario}")

        scenario_qaly_scores[scenario_name] = qaly_scenario

        # Provide personalized feedback based on the utility score
        if scenario_utility_score >= 0.8:
            print("Your health is in great shape! Keep maintaining your current habits.")
        elif 0.5 <= scenario_utility_score < 0.8:
            print("Your health is average. Consider making improvements in lifestyle.")
        else:
            print("Your health needs attention. Consult a healthcare professional.")

        # Print warnings at the end of the output
        if warnings:
            for warning in warnings:
                print(warning)

        # Reset warnings for the next scenario
        warnings.clear()

    # Build scenario_contributions (used by API and by plots below)
    scenario_contributions = {
        scenario: {
            "HDL": scenario_hdl_contributions[scenario] * 1.4 if scenario_hdl_contributions[scenario] < 0 else scenario_hdl_contributions[scenario],
            "Total Cholesterol": scenario_tc_contributions[scenario] * 1.3 if scenario_tc_contributions[scenario] < 0 else scenario_tc_contributions[scenario],
            "Fasting Glucose": scenario_glucose_contributions[scenario] * 1.3 if scenario_glucose_contributions[scenario] < 0 else scenario_glucose_contributions[scenario],
            "BMI": scenario_bmi_contributions[scenario] * 1.2 if scenario_bmi_contributions[scenario] < 0 else scenario_bmi_contributions[scenario],
            "Waist Circumference": scenario_waist_contributions[scenario] * 1.2 if scenario_waist_contributions[scenario] < 0 else scenario_waist_contributions[scenario],
            "Fiber": scenario_fiber_contributions[scenario],
            "Vitamin C": scenario_vitamin_c_contributions[scenario],
            "Calories": scenario_calorie_contributions[scenario],
            "Blood Pressure": scenario_bp_contributions[scenario] * 1.5 if scenario_bp_contributions[scenario] < 0 else scenario_bp_contributions[scenario],
            "High Cholesterol History": scenario_bpq080_contributions[scenario],
            "Diabetes History": scenario_diabetes_contributions[scenario] * 1.4 if scenario_diabetes_contributions[scenario] < 0 else scenario_diabetes_contributions[scenario],
            "Depression (DPQ010)": scenario_depression_contributions[scenario],
            "Depression (DPQ020)": scenario_depression_contributions[scenario],
            "Poor Appetite": scenario_poor_appetite_contributions[scenario],
            "Sedentary Time": scenario_sedentary_contributions[scenario] * 1.05 if scenario_sedentary_contributions[scenario] < 0 else scenario_sedentary_contributions[scenario],
            "Saturated Fat": scenario_saturated_fat_contributions[scenario],
            "Depression Difficulty (DPQ100)": scenario_dpq100_contributions[scenario],
            "Alcohol": scenario_alcohol_contributions[scenario],
            "Insulin": scenario_insulin_contributions[scenario],
            "Walking/Bicycling (PAQ635)": scenario_physical_activity_contributions[scenario].get("PAQ635", 0),
            "Vigorous Activity (PAQ650)": scenario_physical_activity_contributions[scenario].get("PAQ650", 0),
            "Vigorous Activity Days (PAQ655)": scenario_physical_activity_contributions[scenario].get("PAQ655", 0),
            "Moderate Activity (PAQ665)": scenario_physical_activity_contributions[scenario].get("PAQ665", 0),
            "Moderate Activity Days (PAQ670)": scenario_physical_activity_contributions[scenario].get("PAQ670", 0),
            "Special Diet": diet_contribution,
            "Education Level": dmdeduc2_contribution,
        }
        for scenario in ["best_case", "worst_case", "middle_case_1", "middle_case_2"]
    }

    # API: return result for web app (no plots)
    if api_mode:
        params = list(contributions.keys())
        vals = [float(v) for v in contributions.values()]
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        tornado_labels = [p[0] for p in sorted_contrib]
        tornado_vals = [float(p[1]) for p in sorted_contrib]
        scenario_names = ["best_case", "worst_case", "middle_case_1", "middle_case_2"]
        scenario_param_labels = list(scenario_contributions["best_case"].keys())
        scenario_cumulative = {}
        for scenario in scenario_names:
            utility_score = 1.0
            scenario_cumulative[scenario] = [utility_score]
            for key, value in scenario_contributions[scenario].items():
                adjusted_value = value * 1.4 if value < 0 else value
                utility_score += adjusted_value
                scenario_cumulative[scenario].append(utility_score)
            utility_score = max(CONFIG["utility"]["min_score"], min(CONFIG["utility"]["max_score"], utility_score))
            scenario_cumulative[scenario][-1] = utility_score
        return {
            "utility_score": float(final_utility_score),
            "qaly": float(final_qaly),
            "remaining_years": int(remaining_years),
            "contributions": dict(zip(params, vals)),
            "contribution_labels": params,
            "contribution_values": vals,
            "cumulative_utility_labels": ["Initial"] + params,
            "cumulative_utility_values": [float(x) for x in cumulative_scores],
            "tornado_labels": tornado_labels,
            "tornado_values": tornado_vals,
            "scenario_qaly_scores": {k: float(v) for k, v in scenario_qaly_scores.items()},
            "scenario_contribution_labels": scenario_param_labels,
            "scenario_contribution_values": {s: [float(v) for v in scenario_contributions[s].values()] for s in scenario_names},
            "scenario_cumulative_labels": ["Initial"] + scenario_param_labels,
            "scenario_cumulative_series": {k: [float(x) for x in v] for k, v in scenario_cumulative.items()},
            "scenario_tornado_parameters": scenario_param_labels,
            "scenario_tornado_series": [
                {"scenario": s.replace("_", " ").title(), "values": [float(x) for x in scenario_contributions[s].values()]}
                for s in scenario_names
            ],
            "warnings": main_warnings,
        }

    # Plot separate bar charts for each scenario
    for scenario, contributions in scenario_contributions.items():
        plt.figure(figsize=(12, 8))
        parameters = list(contributions.keys())
        values = list(contributions.values())
        colors = ['green' if v > 0 else 'red' for v in values]

        plt.barh(parameters, values, color=colors)
        plt.xlabel('Contribution Value')
        plt.title(f'Utility Score Contribution Breakdown ({scenario.replace("_", " ").title()})')
        plt.axvline(0, color='black', linestyle='--')
        plt.tight_layout()
        if not api_mode:
            plt.show()

    # Cumulative utility score progression
    scenarios = ["best_case", "worst_case", "middle_case_1", "middle_case_2"]
    colors = {"best_case": "green", "worst_case": "red",
              "middle_case_1": "orange", "middle_case_2": "purple"}
    cumulative_scores = {}


    # Compute cumulative utility scores per scenario
    for scenario in scenarios:
        utility_score = 1.0  # Initial utility score
        cumulative_scores[scenario] = [utility_score]
        contributions = scenario_contributions[scenario]

        # Apply contributions dynamically
        for key, value in contributions.items():
            adjusted_value = value * 1.4 if value < 0 else value  # Weight only negative contributions
            utility_score += adjusted_value  # Apply contribution
            cumulative_scores[scenario].append(utility_score)  # Save current score progression
            #print(f"After {key} Contribution ({scenario}): {utility_score:.3f} ({key}: {adjusted_value:.4f})")

        # Cap the final score only at the end
        utility_score = max(CONFIG["utility"]["min_score"], min(CONFIG["utility"]["max_score"], utility_score))
        cumulative_scores[scenario][-1] = utility_score  # Adjust the last value in progression

    # Plot all scenarios in one cumulative graph
    plt.figure(figsize=(12, 8))
    for scenario in scenarios:
        plt.plot(range(len(cumulative_scores[scenario])),
                 cumulative_scores[scenario],
                 marker='o', linestyle='-', color=colors[scenario],
                 label=scenario.replace("_", " ").title())

    plt.xticks(range(len(contributions)+1), ["Initial"] + list(contributions.keys()), rotation=90)
    plt.ylabel('Utility Score')
    plt.title('Cumulative Utility Score Progression for Scenarios (Cap at the End)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if not api_mode:
        plt.show()

    # Tornado chart for grouped scenario comparison
    parameters = list(scenario_contributions["best_case"].keys())
    bar_width = 0.2
    x = np.arange(len(parameters))

    plt.figure(figsize=(12, 10))

    # Plot contributions for each scenario
    for i, scenario in enumerate(scenarios):
        contributions = list(scenario_contributions[scenario].values())
        plt.barh(x - (bar_width * (len(scenarios)/2) + (i * bar_width)),
                 contributions, height=bar_width,
                 label=scenario.replace("_", " ").title(), color=colors[scenario])

    plt.yticks(x, parameters)
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel('Contribution Value')
    plt.title('Tornado Chart: Scenario Comparison for Utility Score Contributions')
    plt.legend()
    plt.tight_layout()
    if not api_mode:
        plt.show()

if __name__ == "__main__":
    qaly()