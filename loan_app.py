import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
with open("loan_approval_ml_app/loan_model.pkl", "rb") as f:
    model = pickle.load(f)

# Page config
st.set_page_config(page_title="Loan Approval App", page_icon="💰", layout="centered")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1995/1995574.png", width=100)
    st.markdown("## 📋 Navigation")
    st.write("Fill the form → Click Predict → Get result 🎯")
    st.markdown("---")
    st.markdown("**Model Info:**")
    st.write("✅ Accuracy: 79%")
    st.write("📊 Logistic Regression")
    st.markdown("[GitHub](https://github.com/) | [Feedback](https://forms.gle/)")

# Header
st.title("🏦 Smart Loan Approval Prediction App")
st.markdown("### Welcome! Let's check if your loan can be approved 💸")
st.markdown("---")

# Input fields
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income (₹)", value=5000)
CoapplicantIncome = st.number_input("Coapplicant Income (₹)", value=0)

# Show loan amount up to ₹10 Lakhs
LoanAmount = st.slider("Loan Amount (₹)", min_value=100000, max_value=1000000, step=10000, value=500000)
LoanAmount_1000s = LoanAmount // 1000  # Convert to 1000s

Loan_Amount_Term = st.slider("Loan Term (in months)", min_value=12, max_value=480, step=12, value=360)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Submit button
if st.button("✅ Predict"):
    # Prepare binary encoded features
    features = np.array([[
        Gender == "Male",
        Married == "Yes",
        int(Dependents.replace("3+", "3")),
        Education == "Graduate",
        Self_Employed == "Yes",
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount_1000s,
        Loan_Amount_Term,
        Credit_History,
        0 if Property_Area == "Urban" else (1 if Property_Area == "Semiurban" else 2)
    ]])

    with st.spinner("🔍 Predicting..."):
        prediction = model.predict(features)
        result = "✅ Loan Approved" if prediction[0] == 1 else "❌ Loan Rejected"

    st.success(f"📢 Prediction Result: {result}")

    with st.expander("📋 Loan Summary"):
        st.write({
            "Gender": Gender,
            "Married": Married,
            "Dependents": Dependents,
            "Education": Education,
            "Self_Employed": Self_Employed,
            "ApplicantIncome": ApplicantIncome,
            "CoapplicantIncome": CoapplicantIncome,
            "LoanAmount (₹)": LoanAmount,
            "Loan_Amount_Term": Loan_Amount_Term,
            "Credit_History": Credit_History,
            "Property_Area": Property_Area
        })

# EMI Calculator
st.markdown("---")
st.subheader("🧲 EMI Calculator")
loan_amt = st.number_input("Loan Amount for EMI (₹)", min_value=10000, max_value=10000000, step=10000, value=500000)
interest_rate = st.slider("Annual Interest Rate (%)", min_value=1.0, max_value=20.0, step=0.1, value=8.5)
tenure = st.slider("Loan Tenure (Years)", min_value=1, max_value=30, step=1, value=10)

if st.button("Calculate EMI"):
    monthly_rate = interest_rate / (12 * 100)
    months = tenure * 12
    emi = (loan_amt * monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
    st.success(f"Monthly EMI: ₹{emi:,.2f}")

# Batch Prediction with CSV
st.markdown("---")
st.subheader("📁 Batch Prediction with CSV Upload")
file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    data = pd.read_csv(file)
    try:
        required_cols = [
            "Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
            "Credit_History", "Property_Area"
        ]
        data = data[required_cols].copy()

        # Preprocess
        data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
        data["Married"] = data["Married"].map({"Yes": 1, "No": 0})
        data["Dependents"] = data["Dependents"].replace("3+", "3").astype(int)
        data["Education"] = data["Education"].map({"Graduate": 1, "Not Graduate": 0})
        data["Self_Employed"] = data["Self_Employed"].map({"Yes": 1, "No": 0})
        data["Property_Area"] = data["Property_Area"].map({"Urban": 0, "Semiurban": 1, "Rural": 2})
        data["LoanAmount"] = data["LoanAmount"] // 1000

        prediction = model.predict(data)
        data['Prediction'] = np.where(prediction == 1, 'Approved', 'Rejected')
        st.write(data)

        # Chart
        st.subheader("📊 Prediction Distribution")
        chart_data = data['Prediction'].value_counts()
        st.bar_chart(chart_data)
    except Exception as e:
        st.error(f"Error: {e}")

# Footer tip
st.markdown("---")
st.info("💡 Tip: Maintain good credit history and stable income to improve loan chances.")
