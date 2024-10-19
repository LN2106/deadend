import streamlit as st
import pandas as pd
import joblib

# Function to load the pre-trained Logistic Regression model
def load_model():
    model = joblib.load('/workspaces/deadend/model/log_reg_82_model.joblib')
    return model

# Function to make predictions using the loaded model
def make_prediction(model, data):
    predictions = model.predict(data)
    return predictions

# Function to convert numerical predictions to string labels
def convert_predictions(predictions):
    return ["Churn" if pred == 1 else "No Churn" for pred in predictions]

# Main app layout
st.title("Customer Churn Prediction App")

# Sidebar navigation
page = st.sidebar.selectbox("Choose a page", ["CSV Upload", "Manual Data Entry"])

# Load the model
model = load_model()

# Page 1: CSV Upload
if page == "CSV Upload":
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Upload your encoded and scaled CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", data.head())

        # Make predictions
        if st.button('Predict'):
            predictions = make_prediction(model, data)
            label_predictions = convert_predictions(predictions)

            # Create a DataFrame to show original data with predictions
            result_df = data.copy()
            result_df['Predicted Churn'] = label_predictions
            st.write("Predictions with Original Data:", result_df)

# Page 2: Manual Data Entry
elif page == "Manual Data Entry":
    st.header("Enter Customer Data Manually")

    # Form for manual data entry
    with st.form("manual_data_form"):
        # Checkboxes for binary choices
        gender = st.checkbox("Is the customer male?", value=True)
        senior_citizen = st.checkbox("Is the customer a senior citizen?", value=False)
        partner = st.checkbox("Does the customer have a partner?", value=False)
        dependents = st.checkbox("Does the customer have dependents?", value=False)
        phone_service = st.checkbox("Does the customer have phone service?", value=True)
        multiple_lines = st.checkbox("Does the customer have multiple lines?", value=False)
        online_security = st.checkbox("Does the customer have online security?", value=False)
        online_backup = st.checkbox("Does the customer have online backup?", value=False)
        device_protection = st.checkbox("Does the customer have device protection?", value=False)
        tech_support = st.checkbox("Does the customer have tech support?", value=False)
        streaming_tv = st.checkbox("Does the customer use streaming TV?", value=False)
        streaming_movies = st.checkbox("Does the customer use streaming movies?", value=False)
        paperless_billing = st.checkbox("Does the customer use paperless billing?", value=True)

        # Number inputs for continuous features
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=50.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=100.0)
        clv = st.number_input("Customer Lifetime Value (CLV)", min_value=0.0, max_value=100000.0, value=1000.0)
        avg_monthly_charges = st.number_input("Avg Monthly Charges", min_value=0.0, max_value=500.0, value=50.0)

        # Checkboxes for Internet Service (mutually exclusive options)
        internet_service_dsl = st.checkbox("DSL Internet Service", value=False)
        internet_service_fiber = st.checkbox("Fiber Optic Internet Service", value=True)
        internet_service_no = st.checkbox("No Internet Service", value=False)

        # Checkboxes for Contract Type (mutually exclusive options)
        contract_month_to_month = st.checkbox("Month-to-Month Contract", value=True)
        contract_one_year = st.checkbox("One Year Contract", value=False)
        contract_two_year = st.checkbox("Two Year Contract", value=False)

        # Checkboxes for Payment Method (mutually exclusive options)
        payment_bank_transfer = st.checkbox("Bank Transfer (Automatic)", value=False)
        payment_credit_card = st.checkbox("Credit Card (Automatic)", value=True)
        payment_electronic_check = st.checkbox("Electronic Check", value=False)
        payment_mailed_check = st.checkbox("Mailed Check", value=False)

        # Submit button in the form
        submitted = st.form_submit_button("Predict")

    # Process the form data for prediction
    if submitted:
        # Convert inputs to the same format as the model expects
        input_data = pd.DataFrame({
            'gender': [1 if gender else 0],
            'SeniorCitizen': [1 if senior_citizen else 0],
            'Partner': [1 if partner else 0],
            'Dependents': [1 if dependents else 0],
            'tenure': [tenure],
            'PhoneService': [1 if phone_service else 0],
            'MultipleLines': [1 if multiple_lines else 0],
            'OnlineSecurity': [1 if online_security else 0],
            'OnlineBackup': [1 if online_backup else 0],
            'DeviceProtection': [1 if device_protection else 0],
            'TechSupport': [1 if tech_support else 0],
            'StreamingTV': [1 if streaming_tv else 0],
            'StreamingMovies': [1 if streaming_movies else 0],
            'PaperlessBilling': [1 if paperless_billing else 0],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'InternetService_DSL': [1 if internet_service_dsl else 0],
            'InternetService_Fiber optic': [1 if internet_service_fiber else 0],
            'InternetService_No': [1 if internet_service_no else 0],
            'Contract_Month-to-month': [1 if contract_month_to_month else 0],
            'Contract_One year': [1 if contract_one_year else 0],
            'Contract_Two year': [1 if contract_two_year else 0],
            'PaymentMethod_Bank transfer (automatic)': [1 if payment_bank_transfer else 0],
            'PaymentMethod_Credit card (automatic)': [1 if payment_credit_card else 0],
            'PaymentMethod_Electronic check': [1 if payment_electronic_check else 0],
            'PaymentMethod_Mailed check': [1 if payment_mailed_check else 0],
            'CLV': [clv],
            'AvgMonthlyCharges': [avg_monthly_charges]
        })

        # Make predictions
        predictions = make_prediction(model, input_data)
        label_predictions = convert_predictions(predictions)

        # Display prediction result
        st.write(f"Prediction: {label_predictions[0]}")
