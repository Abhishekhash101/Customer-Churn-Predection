import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load saved model and preprocessing objects
model = load_model('model_trained.h5')
scaler = joblib.load('scaler.pkl')
ohe_geo = joblib.load('ohe_geo.pkl')
ohe_gender = joblib.load('ohe_gender.pkl')

# Streamlit UI
st.title("Customer Churn Prediction")

st.write("Enter customer details below:")

# Collect user input
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', ohe_gender.categories_[0])
age = st.number_input('Age', min_value=18, max_value=100, value=40)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=3)
balance = st.number_input('Balance', min_value=0.0, max_value=300000.0, value=60000.0)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, max_value=300000.0, value=50000.0)

if st.button('Predict Churn'):
    # Prepare input as DataFrame
    input_dict = {
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    }
    input_df = pd.DataFrame(input_dict)

    # Apply one-hot encoding (must match training)
    geo_encoded = ohe_geo.transform(input_df[['Geography']]).toarray()
    gender_encoded = ohe_gender.transform(input_df[['Gender']]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))
    gender_df = pd.DataFrame(gender_encoded, columns=ohe_gender.get_feature_names_out(['Gender']))

    # Combine all features
    input_processed = pd.concat(
        [input_df.drop(['Geography', 'Gender'], axis=1), geo_df, gender_df], axis=1
    )

    # Ensure columns in correct order (match training)
    model_features = scaler.feature_names_in_
    input_processed = input_processed[model_features]

    # Scale features
    input_scaled = scaler.transform(input_processed)

    # Predict
    prediction = model.predict(input_scaled)[0][0]
    st.subheader(f"Churn Probability: {prediction:.2%}")

    if prediction >= 0.5:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is unlikely to churn.")
