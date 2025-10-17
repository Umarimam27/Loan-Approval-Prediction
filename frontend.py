import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request

# --- CONFIGURATION ---
MODEL_FILE_PATH = r'C:\Users\umari\OneDrive\Desktop\FSDS Projects\Loan Approval Prediction\all_loan_prediction_models.pkl'
# 🚨 FIX 1: Specify which single model you want to use from the dictionary
MODEL_NAME_TO_USE = 'RandomForest' 

# 🚨 FIX 2: List of all 14 columns the model expects after preprocessing.
# This order and list are CRITICAL for the model to work.
EXPECTED_COLUMNS = [
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Gender_Male', 'Married_Yes', 'Dependents_1', 
    'Dependents_2', 'Dependents_3+', 'Education_Not Graduate', 
    'Self_Employed_Yes', 'Property_Area_Semiurban', 'Property_Area_Urban'
]

# Default values for features NOT collected by your simple form
DEFAULT_VALUES = {
    'Gender': 'Male',
    'Married': 'Yes',
    'Dependents': '0', # Defaulting to '0' dependents
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'CoapplicantIncome': 0.0, # Defaulting to no coapplicant income
    'LoanAmount': 146.4,      # Defaulting to mean loan amount (estimate)
    'Loan_Amount_Term': 360.0 # Defaulting to mode term
}

# Mapping for the Property_Area input from HTML (0, 1, 2)
PROPERTY_AREA_MAPPING = {
    '0': 'Rural',
    '1': 'Urban',
    '2': 'Semiurban'
}
# ---------------------

# --- Load the Model Globally (Corrected) ---
try:
    with open(MODEL_FILE_PATH, 'rb') as file:
        all_models = pickle.load(file)
        # Load the selected model from the dictionary
        model_to_use = all_models[MODEL_NAME_TO_USE]
    print(f"✅ Successfully loaded model: {MODEL_NAME_TO_USE}")
except Exception as e:
    print(f"❌ Error loading model file {MODEL_FILE_PATH} or key '{MODEL_NAME_TO_USE}': {e}")
    model_to_use = None
    
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model_to_use is None:
        return render_template('index.html', prediction="Error: Model failed to load in memory.")

    try:
        # 1. Get user input from the form and map Property Area
        data = request.form.to_dict()
        user_credit_history = float(data['Credit_History'])
        user_property_area = PROPERTY_AREA_MAPPING[data['Property_Area']]
        user_income = float(data['Income'])
        
        # 2. Build the full input data structure
        input_data = DEFAULT_VALUES.copy()
        
        # 3. Overwrite/add user-provided values
        input_data['Credit_History'] = user_credit_history
        input_data['Property_Area'] = user_property_area
        input_data['ApplicantIncome'] = user_income
        
        # Convert dictionary to a DataFrame (required for pandas preprocessing)
        input_df = pd.DataFrame([input_data])
        
        # 4. Replicate Preprocessing (One-Hot Encoding) - CRITICAL STEP
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        input_processed = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        
        # 5. Align Columns (Final critical step: match feature names and order)
        final_input = input_processed.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
        
        # 6. Make Prediction
        raw_prediction = model_to_use.predict(final_input)[0]
        
        # 7. Format Output
        result = "✅ Loan Approved!" if raw_prediction == 1 else "❌ Loan Rejected."

        return render_template('index.html', prediction=result)

    except Exception as e:
        # This catches errors like unconvertible float values or column misalignment issues
        print(f"Prediction Error: {e}")
        return render_template('index.html', prediction=f"Prediction Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)