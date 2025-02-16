import streamlit as st
import pandas as pd
import pickle

# Title and description
st.title("Customer Churn Prediction")
st.write(
    """
    This application predicts whether a customer is likely to churn based on their details.
    Provide the required details below to get a prediction.
    """
)

# Load the saved model and encoders
try:
    with open(r"C:\Users\thril\CrossDevice\Thrilochan\Built A Thon\customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    loaded_model = model_data["model"]
    feature_names = model_data["features_names"]

    with open(r"C:\\Users\\thril\\CrossDevice\\Thrilochan\\Built A Thon\\encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# User input function
def user_input_features():
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=1)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.01)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, step=0.01)

    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }
    return pd.DataFrame([data])

# Get user input
input_data_df = user_input_features()

# Encode categorical features with additional checks
try:
    for column, encoder in encoders.items():
        # Ensure the column is present in the input data
        if column not in input_data_df.columns:
            continue
        
        # Print debug info (optional)
        st.write(f"Encoding column: {column}")
        st.write(f"Input data for {column}: {input_data_df[column].unique()}")
        st.write(f"Encoder categories for {column}: {encoder.classes_}")
        
        # Check for unexpected values
        unknown_values = [value for value in input_data_df[column].unique() if value not in encoder.classes_]
        if unknown_values:
            st.error(f"Unexpected values in column '{column}': {unknown_values}. Please check your input.")
            st.stop()
        
        # Apply encoding
        input_data_df[column] = encoder.transform(input_data_df[column])
except Exception as e:
    st.error(f"An error occurred during data encoding: {e}")
    st.stop()

# Prediction
if st.button("Predict"):
    try:
        # Ensure the input data matches the model's expected features
        input_data_df = input_data_df[feature_names]
        
        prediction = loaded_model.predict(input_data_df)
        pred_prob = loaded_model.predict_proba(input_data_df)

        st.subheader("Prediction")
        st.write("Churn" if prediction[0] == 1 else "No Churn")

        st.subheader("Prediction Probability")
        st.write(f"Churn: {pred_prob[0][1]:.2f}, No Churn: {pred_prob[0][0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
