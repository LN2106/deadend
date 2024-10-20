import numpy as np
from sklearn.linear_model import LogisticRegression
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import shap  # Import SHAP library


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

# Function to convert DataFrame to CSV for download
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Function to suggest customer retention strategies
def suggest_retention_strategies(data):
    strategies = []

    for index, row in data.iterrows():
        if row['Churn'] == 'Churn':  # Only provide suggestions for customers predicted to churn
            suggestion = f"Customer {index + 1}: "
            
            # Check contract type
            if row['Contract_Month-to-month'] == 1:
                suggestion += "Offer a longer-term contract with a discount. "
            elif row['Contract_Two year'] == 1:
                suggestion += "Highlight the benefits of staying with the two-year contract. "

            # Check tenure (lower tenure customers are more likely to churn)
            if row['tenure'] < 12:
                suggestion += "Provide loyalty rewards to encourage them to stay longer. "
            elif row['tenure'] > 36:
                suggestion += "Send a personalized thank you for being a long-term customer. "

            # Check payment method (electronic checks are associated with higher churn)
            if row['PaymentMethod_Electronic check'] == 1:
                suggestion += "Offer incentives to switch to automatic payment methods (bank transfer/credit card). "

            # Gender-specific suggestions (optional, just to show different kinds of personalizations)
            if row['gender'] == 1:  # Assuming 1 is male
                suggestion += "Send personalized offers based on their past service usage. "
            else:
                suggestion += "Provide exclusive deals to maintain customer satisfaction. "
            
            # Add the completed suggestion
            strategies.append(suggestion)

    return strategies


# Store the data in session state for persistence
if "uploaded_data" not in st.session_state:
    st.session_state["uploaded_data"] = None

if "predicted_data" not in st.session_state:
    st.session_state["predicted_data"] = None

# Main app layout
st.title("Customer Churn Prediction App")

# Sidebar navigation
page = st.sidebar.selectbox("Choose a page", ["CSV Upload", "Manual Data Entry", "Visualization", "Explainable AI (XAI)", "Customer Retention Suggestions", "3D Visualization"])

# Load the model
model = load_model()
st.write("Model loaded successfully")  # Debugging statement

# Page 1: CSV Upload
if page == "CSV Upload":
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Upload your encoded and scaled CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state["uploaded_data"] = data  # Store the uploaded data in session_state
        st.write("Uploaded Data Preview:", data.head())

        # Make predictions
        if st.button('Predict'):
            predictions = make_prediction(model, data)
            label_predictions = convert_predictions(predictions)

            # Add the predictions to the DataFrame
            result_df = data.copy()
            result_df['Churn'] = label_predictions
            st.session_state["predicted_data"] = result_df  # Store the result in session_state
            st.write("Predictions with Original Data:", result_df)

            # Provide download link for the resulting DataFrame with predictions
            csv_data = convert_df_to_csv(result_df)
            st.download_button(
                label="Download predictions as CSV",
                data=csv_data,
                file_name='predicted_churn.csv',
                mime='text/csv',
            )

# Page 4: Customer Retention Suggestions
elif page == "Customer Retention Suggestions":
    st.header("Customer Retention Suggestions")

    # Check if the predicted data is available
    if st.session_state["predicted_data"] is not None:
        data = st.session_state["predicted_data"]

        # Generate retention suggestions
        suggestions = suggest_retention_strategies(data)

        # Display the suggestions
        if suggestions:
            for suggestion in suggestions:
                st.write(suggestion)
        else:
            st.write("No retention suggestions available.")
    else:
        st.warning("Please upload data and make predictions first.")


# Page 2: Manual Data Entry
elif page == "Manual Data Entry":
    st.header("Enter Customer Data Manually")

    # Form for manual data entry
    with st.form("manual_data_form"):
        # Section: Customer Demographics
        st.subheader("Customer Demographics")
        gender = st.checkbox("Is the customer male?", value=True)
        senior_citizen = st.checkbox("Is the customer a senior citizen?", value=False)
        partner = st.checkbox("Does the customer have a partner?", value=False)
        dependents = st.checkbox("Does the customer have dependents?", value=False)

        # Section: Phone & Online Services
        st.subheader("Phone & Online Services")
        phone_service = st.checkbox("Does the customer have phone service?", value=True)
        multiple_lines = st.checkbox("Does the customer have multiple lines?", value=False)
        online_security = st.checkbox("Does the customer have online security?", value=False)
        online_backup = st.checkbox("Does the customer have online backup?", value=False)
        device_protection = st.checkbox("Does the customer have device protection?", value=False)
        tech_support = st.checkbox("Does the customer have tech support?", value=False)
        streaming_tv = st.checkbox("Does the customer use streaming TV?", value=False)
        streaming_movies = st.checkbox("Does the customer use streaming movies?", value=False)
        paperless_billing = st.checkbox("Does the customer use paperless billing?", value=True)

        # Section: Financial Information
        st.subheader("Financial Information")
        tenure = st.slider("Tenure (months)", min_value=0, max_value=100, value=1)
        monthly_charges = st.slider("Monthly Charges", min_value=0.0, max_value=500.0, value=50.0)
        total_charges = st.slider("Total Charges", min_value=0.0, max_value=10000.0, value=100.0)
        clv = st.slider("Customer Lifetime Value (CLV)", min_value=0.0, max_value=100000.0, value=1000.0)
        avg_monthly_charges = st.slider("Avg Monthly Charges", min_value=0.0, max_value=500.0, value=50.0)

        # Section: Internet Service
        st.subheader("Internet Service")
        internet_service_dsl = st.checkbox("DSL Internet Service", value=False)
        internet_service_fiber = st.checkbox("Fiber Optic Internet Service", value=True)
        internet_service_no = st.checkbox("No Internet Service", value=False)

        # Section: Contract Type
        st.subheader("Contract Type")
        contract_month_to_month = st.checkbox("Month-to-Month Contract", value=True)
        contract_one_year = st.checkbox("One Year Contract", value=False)
        contract_two_year = st.checkbox("Two Year Contract", value=False)

        # Section: Payment Method
        st.subheader("Payment Method")
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

        # Debugging: Print the input data to verify correctness
        st.write("Input Data for Prediction:", input_data)

        # Make predictions
        predictions = make_prediction(model, input_data)
        label_predictions = convert_predictions(predictions)

        # Display prediction result
        st.write(f"Prediction: {label_predictions[0]}")

# Page 3: Visualization
elif page == "Visualization":
    st.header("Visualizations")

    # Check if the uploaded data is available in session_state
    if st.session_state["predicted_data"] is not None:
        data = st.session_state["predicted_data"]

        # 1. Univariate Analysis
        st.subheader("Univariate Analysis")
        
        # Tenure distribution
        st.write("Distribution of Tenure")
        fig1, ax1 = plt.subplots()
        sns.histplot(data['tenure'], kde=True, ax=ax1)
        st.pyplot(fig1)

        # Monthly Charges distribution
        st.write("Distribution of Monthly Charges")
        fig2, ax2 = plt.subplots()
        sns.histplot(data['MonthlyCharges'], kde=True, ax=ax2)
        st.pyplot(fig2)

        # Total Charges distribution
        st.write("Distribution of Total Charges")
        fig3, ax3 = plt.subplots()
        sns.histplot(data['TotalCharges'], kde=True, ax=ax3)
        st.pyplot(fig3)

        # 2. Bivariate Analysis
        st.subheader("Bivariate Analysis")
        
        # Monthly Charges vs Total Charges with hue by Churn
        st.write("Monthly Charges vs Total Charges")
        fig4, ax4 = plt.subplots()
        sns.scatterplot(x=data['MonthlyCharges'], y=data['TotalCharges'], hue=data['Churn'], ax=ax4)
        st.pyplot(fig4)

        # Tenure vs Monthly Charges by Churn
        st.write("Tenure vs Monthly Charges by Churn")
        fig5, ax5 = plt.subplots()
        sns.scatterplot(x=data['tenure'], y=data['MonthlyCharges'], hue=data['Churn'], ax=ax5)
        st.pyplot(fig5)

        # Contract Type vs Total Charges
        st.write("Contract Type vs Total Charges")
        fig6, ax6 = plt.subplots()
        sns.boxplot(x=data['Contract_Month-to-month'], y=data['TotalCharges'], hue=data['Churn'], ax=ax6)
        st.pyplot(fig6)

        # 3. Multivariate Analysis
        st.subheader("Multivariate Analysis")
        
        # Pairplot for selected features
        st.write("Pairplot of Selected Features")
        selected_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV', 'Churn']
        fig7 = sns.pairplot(data[selected_columns], hue='Churn')
        st.pyplot(fig7)

        # Correlation Heatmap for Selected Columns
        st.write("Correlation Heatmap (Selected Features)")
        selected_columns_heatmap = ['gender', 'tenure', 'MonthlyCharges', 'TotalCharges', 'CLV', 
                                    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year']
        
        # Ensure only available columns are selected
        available_columns = [col for col in selected_columns_heatmap if col in data.columns]
        selected_data = data[available_columns]
        
        fig8, ax8 = plt.subplots()
        sns.heatmap(selected_data.corr(), annot=True, cmap="coolwarm", ax=ax8)
        st.pyplot(fig8)

        

    else:
        st.warning("Please upload data first to see the visualizations.")



# New page for XAI (Explainable AI)
elif page == "Explainable AI (XAI)":
    st.header("Explainable AI - SHAP")

    # Check if the uploaded data and the model are available in session_state
    if st.session_state["predicted_data"] is not None and model is not None:
        data = st.session_state["predicted_data"]

        # Only use the features from your model (excluding the target column 'Churn')
        X = data.drop(columns=['Churn'])

        # Generate SHAP values for your Logistic Regression model
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)

        # Select a specific sample/customer for explanation
        st.subheader("Explain a single prediction")
        customer_index = st.slider("Select Customer Index for SHAP Explanation", 0, len(data)-1, 0)
        customer_data = X.iloc[[customer_index]]

        st.write("Selected Customer's Data:")
        st.dataframe(customer_data)

        # SHAP force plot for individual prediction
        st.subheader("SHAP Force Plot for Selected Customer")
        shap.force_plot(explainer.expected_value, shap_values[customer_index, :], X.iloc[customer_index, :], matplotlib=True)
        st.pyplot(bbox_inches='tight')

        # SHAP summary plot to see feature importance across all customers
        st.subheader("SHAP Summary Plot (Feature Importance Across All Customers)")
        shap.summary_plot(shap_values, X)
        st.pyplot(bbox_inches='tight')

    else:
        st.warning("Please upload data and ensure the model is available for SHAP explanation.")


# Page 4: Customer Retention Suggestions
elif page == "Customer Retention Suggestions":
    st.header("Customer Retention Suggestions")

    # Check if the predicted data is available
    if st.session_state["predicted_data"] is not None:
        data = st.session_state["predicted_data"]

        # Generate retention suggestions
        suggestions = suggest_retention_strategies(data)

        # Display the suggestions
        if suggestions:
            for suggestion in suggestions:
                st.write(suggestion)
        else:
            st.write("No retention suggestions available.")
    else:
        st.warning("Please upload data and make predictions first.")

import plotly.express as px

# Page for 3D Visualization
if page == "3D Visualization":
    st.header("3D Interactive Plot")

    # Check if the uploaded data is available in session_state
    if st.session_state["predicted_data"] is not None:
        data = st.session_state["predicted_data"]

        # Creating a 3D scatter plot
        fig = px.scatter_3d(
            data,
            x='MonthlyCharges',  # Change these as per your dataset
            y='TotalCharges',     # Change these as per your dataset
            z='tenure',           # Change these as per your dataset
            color='Churn',        # This will color the points by churn status
            title='3D Scatter Plot of Monthly Charges, Total Charges, and Tenure',
            labels={'Churn': 'Churn Status'},
            color_continuous_scale=px.colors.sequential.Viridis
        )

        # Show the plot in the Streamlit app
        st.plotly_chart(fig)
    else:
        st.warning("Please upload data first to see the 3D plot.")
