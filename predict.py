import joblib
import pandas as pd

# Load the model
model = joblib.load("loan_model.pkl")

# Sample user input (you can later replace with actual input)
data = {
    'Gender': [1],  # Male=1, Female=0
    'Married': [1],
    'Dependents': [0],
    'Education': [0],  # Graduate=0, Not Graduate=1
    'Self_Employed': [0],
    'ApplicantIncome': [5000],
    'CoapplicantIncome': [0],
    'LoanAmount': [150],
    'Loan_Amount_Term': [360],
    'Credit_History': [1],
    'Property_Area': [2]  # Urban=2, Semiurban=1, Rural=0
}

# Convert to DataFrame
input_df = pd.DataFrame(data)

# Make prediction
prediction = model.predict(input_df)

# Show result
print("✅ Prediction:", "Loan Approved" if prediction[0] == 1 else "Loan Rejected")
