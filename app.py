# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page configuration ---
st.set_page_config(
    page_title="Bank Churn Prediction App",
    page_icon="🏦",
    layout="wide"
)

# --- Load model ---
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- Load dataset for visualization ---
df = pd.read_csv('Bank customers.csv')

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2331/2331940.png", width=150)
st.sidebar.title("Bank Churn Dashboard")
st.sidebar.markdown("Use the sidebar to input customer details and get churn prediction!")

# --- Customer input ---
st.sidebar.header("Enter Customer Details")

def user_input_features():
    Customer_Age = st.sidebar.slider('Customer Age', int(df['Customer_Age'].min()), int(df['Customer_Age'].max()), 40)
    Gender = st.sidebar.selectbox('Gender', df['Gender'].unique())
    Dependent_count = st.sidebar.slider('Number of Dependents', int(df['Dependent_count'].min()), int(df['Dependent_count'].max()), 0)
    Education_Level = st.sidebar.selectbox('Education Level', df['Education_Level'].unique())
    Marital_Status = st.sidebar.selectbox('Marital Status', df['Marital_Status'].unique())
    Income_Category = st.sidebar.selectbox('Income Category', df['Income_Category'].unique())
    Months_on_book = st.sidebar.slider('Months on Book', int(df['Months_on_book'].min()), int(df['Months_on_book'].max()), 12)
    Total_Relationship_Count = st.sidebar.slider('Total Relationship Count', int(df['Total_Relationship_Count'].min()), int(df['Total_Relationship_Count'].max()), 3)
    Months_Inactive_12_mon = st.sidebar.slider('Months Inactive', int(df['Months_Inactive_12_mon'].min()), int(df['Months_Inactive_12_mon'].max()), 1)
    Contacts_Count_12_mon = st.sidebar.slider('Contacts in Last 12 months', int(df['Contacts_Count_12_mon'].min()), int(df['Contacts_Count_12_mon'].max()), 2)
    Credit_Limit = st.sidebar.number_input('Credit Limit', float(df['Credit_Limit'].min()), float(df['Credit_Limit'].max()), 5000.0)
    Total_Revolving_Bal = st.sidebar.number_input('Total Revolving Balance', float(df['Total_Revolving_Bal'].min()), float(df['Total_Revolving_Bal'].max()), 1000.0)
    Avg_Open_To_Buy = st.sidebar.number_input('Average Open to Buy', float(df['Avg_Open_To_Buy'].min()), float(df['Avg_Open_To_Buy'].max()), 4000.0)
    Total_Amt_Chng_Q4_Q1 = st.sidebar.number_input('Total Amount Change Q4/Q1', float(df['Total_Amt_Chng_Q4_Q1'].min()), float(df['Total_Amt_Chng_Q4_Q1'].max()), 1.2)
    Total_Trans_Amt = st.sidebar.number_input('Total Transaction Amount', float(df['Total_Trans_Amt'].min()), float(df['Total_Trans_Amt'].max()), 2000.0)
    Total_Trans_Ct = st.sidebar.number_input('Total Transaction Count', int(df['Total_Trans_Ct'].min()), int(df['Total_Trans_Ct'].max()), 50)
    Total_Ct_Chng_Q4_Q1 = st.sidebar.number_input('Total Count Change Q4/Q1', float(df['Total_Ct_Chng_Q4_Q1'].min()), float(df['Total_Ct_Chng_Q4_Q1'].max()), 1.0)
    Avg_Utilization_Ratio = st.sidebar.slider('Avg Utilization Ratio', float(df['Avg_Utilization_Ratio'].min()), float(df['Avg_Utilization_Ratio'].max()), 0.3)

    data = {
        'Customer_Age': Customer_Age,
        'Gender': Gender,
        'Dependent_count': Dependent_count,
        'Education_Level': Education_Level,
        'Marital_Status': Marital_Status,
        'Income_Category': Income_Category,
        'Months_on_book': Months_on_book,
        'Total_Relationship_Count': Total_Relationship_Count,
        'Months_Inactive_12_mon': Months_Inactive_12_mon,
        'Contacts_Count_12_mon': Contacts_Count_12_mon,
        'Credit_Limit': Credit_Limit,
        'Total_Revolving_Bal': Total_Revolving_Bal,
        'Avg_Open_To_Buy': Avg_Open_To_Buy,
        'Total_Amt_Chng_Q4_Q1': Total_Amt_Chng_Q4_Q1,
        'Total_Trans_Amt': Total_Trans_Amt,
        'Total_Trans_Ct': Total_Trans_Ct,
        'Total_Ct_Chng_Q4_Q1': Total_Ct_Chng_Q4_Q1,
        'Avg_Utilization_Ratio': Avg_Utilization_Ratio
    }

    features = pd.DataFrame(data, index=[0])

    # --- One-hot encode categorical variables ---
    categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category']
    features = pd.get_dummies(features, columns=categorical_cols)

    return features

# Get user input
input_df = user_input_features()

# --- Align input columns with model ---
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# --- Display Prediction ---
st.subheader("Customer Churn Prediction")
try:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.markdown(f"**Predicted Card Category:** {prediction[0]}")
    st.markdown("**Prediction Probabilities:**")
    st.dataframe(pd.DataFrame(prediction_proba, columns=model.classes_))
except Exception as e:
    st.error(f"Error in prediction: {e}")

# --- Show input data for debugging ---
st.subheader("Input Data")
st.write(input_df)

# --- Data Visualization ---
st.subheader("Bank Customer Data Overview")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Customers by Card Category**")
    fig, ax = plt.subplots()
    sns.countplot(x='Card_Category', data=df, palette="coolwarm", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    st.markdown("**Credit Limit by Income Category**")
    fig2, ax2 = plt.subplots()
    sns.barplot(x='Income_Category', y='Credit_Limit', data=df, palette="viridis", ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)
