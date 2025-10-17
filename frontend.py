import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request

# --- CONFIGURATION ---
# IMPORTANT: Ensure this path is correct on your local machine.
MODEL_FILE_PATH = r'C:\Users\umari\OneDrive\Desktop\FSDS Projects\Loan Approval Prediction\all_loan_prediction_models.pkl'
MODEL_NAME_TO_USE = 'RandomForest' 

# CRITICAL: This list must exactly match the features your model was trained on.
EXPECTED_COLUMNS = [
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Gender_Male', 'Married_Yes', 'Dependents_1', 
    'Dependents_2', 'Dependents_3+', 'Education_Not Graduate', 
    'Self_Employed_Yes', 'Property_Area_Semiurban', 'Property_Area_Urban'
]

# Robust defaults for all input features to prevent crashes on missing/empty data
DEFAULT_INPUTS = {
    # Categorical Defaults
    'Gender': 'Male',
    'Married': 'Yes',
    'Dependents': '0',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'Property_Area': 'Semiurban',
    
    # Numerical Defaults
    'ApplicantIncome': 5000.0,
    'CoapplicantIncome': 0.0, 
    'LoanAmount': 146.4,      
    'Loan_Amount_Term': 360.0, 
    'Credit_History': 1.0 
}

# ---------------------

# --- Load the Model Globally ---
try:
    with open(MODEL_FILE_PATH, 'rb') as file:
        all_models = pickle.load(file)
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
        data = request.form.to_dict()
        input_data = {}

        # --- Helper function for robust retrieval and type conversion ---
        def get_value_or_default(key, default_type=str):
            raw_value = data.get(key)
            default = DEFAULT_INPUTS[key]
            
            # If the value is missing or an empty string, return the default value.
            if not raw_value:
                return default
            
            # Otherwise, try to convert it to the requested type (float or str)
            try:
                # If we expect a string (like 'Dependents'), ensure it's converted to string
                if default_type == str:
                     return str(raw_value)
                return default_type(raw_value)
            except ValueError as ve:
                print(f"Warning: Could not convert input '{key}': {raw_value} to {default_type}. Error: {ve}")
                # Fallback if conversion fails (e.g., non-numeric string for float)
                return default

        # --- Populate Categorical Features (Strings) ---
        input_data['Gender'] = get_value_or_default('Gender')
        input_data['Married'] = get_value_or_default('Married')
        input_data['Education'] = get_value_or_default('Education')
        input_data['Self_Employed'] = get_value_or_default('Self_Employed')
        input_data['Property_Area'] = get_value_or_default('Property_Area')
        input_data['Dependents'] = get_value_or_default('Dependents') # Stays as string ('0', '1', '2', '3+')

        # --- Populate Numerical Features (Floats) ---
        input_data['ApplicantIncome'] = get_value_or_default('ApplicantIncome', float)
        input_data['CoapplicantIncome'] = get_value_or_default('CoapplicantIncome', float)
        input_data['LoanAmount'] = get_value_or_default('LoanAmount', float)
        input_data['Loan_Amount_Term'] = get_value_or_default('Loan_Amount_Term', float)
        input_data['Credit_History'] = get_value_or_default('Credit_History', float)

        # Convert dictionary to a DataFrame
        input_df = pd.DataFrame([input_data])
        
        # 3. Replicate Preprocessing (One-Hot Encoding)
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        input_processed = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        
        # 4. Align Columns (CRITICAL: match feature names and order)
        final_input = input_processed.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
        
        # 5. Make Prediction
        raw_prediction = model_to_use.predict(final_input)[0]
        
        # 6. Format Output
        result = "✅ Loan Approved!" if raw_prediction == 1 else "❌ Loan Rejected."

        return render_template('index.html', prediction=result)

    except Exception as e:
        # Catch any unexpected errors that bypass the safe conversion (e.g., model misalignment)
        print(f"A broader system error occurred during prediction: {e}")
        return render_template('index.html', prediction=f"Prediction System Error: An unexpected error occurred. Details: {type(e).__name__}")

if __name__ == '__main__':
    app.run(debug=True)
