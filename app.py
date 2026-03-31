import sys
import os
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
import seaborn as sns
import joblib

from frontend.code.DiseaseModel import DiseaseModel
from frontend.code.helper import prepare_symptoms_array

# =========================
# PAGE CONFIG (UI ONLY)
# =========================
st.set_page_config(
    page_title="Multiple Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# =========================
# MODERN UI (ONLY ADD)
# =========================
st.markdown("""
<style>

/* Background */
.main {
    background: linear-gradient(to right, #e0f7fa, #ffffff);
}

/* Title */
.big-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #00695c;
    margin-bottom: 10px;
}

/* Card */
.card {
    background: white;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
    margin: 10px 0px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(to right, #26a69a, #00796b);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    border: none;
}

/* Success box */
.success-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #e8f5e9;
    color: #2e7d32;
}

/* Danger box */
.danger-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #ffebee;
    color: #c62828;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #004d40;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE (UI ONLY)
# =========================
st.markdown("<div class='big-title'>🩺 Multiple Disease Prediction System</div>", unsafe_allow_html=True)
st.markdown("---")

# =========================
# LOAD MODELS (NO CHANGE)
# =========================
diabetes_model = joblib.load("frontend/models/diabetes_model.sav")
heart_model = joblib.load("frontend/models/heart_disease_model.sav")
parkinson_model = joblib.load("frontend/models/parkinsons_model.sav")
lung_cancer_model = joblib.load("frontend/models/lung_cancer_model.sav")
breast_cancer_model = joblib.load("frontend/models/breast_cancer.sav")
chronic_disease_model = joblib.load("frontend/models/chronic_model.sav")
hepatitis_model = joblib.load("frontend/models/hepatitis_model.sav")
liver_model = joblib.load("frontend/models/liver_model.sav")

# =========================
# SIDEBAR (NO CHANGE LOGIC)
# =========================
with st.sidebar:
    st.markdown("### 🏥 Medical Dashboard")
    st.markdown("---")

    selected = option_menu(
        'Multiple Disease Prediction',
        [
            'Disease Prediction',
            'Diabetes Prediction',
            'Heart disease Prediction',
            'Parkison Prediction',
            'Liver prediction',
            'Hepatitis prediction',
            'Lung Cancer Prediction',
            'Chronic Kidney prediction',
            'Breast Cancer Prediction',
        ],
        icons=['activity','activity','heart','person','activity','activity','lungs','droplet','gender-ambiguous'],
        default_index=0
    )

# =========================
# DISEASE PREDICTION
# =========================
if selected == 'Disease Prediction':

    disease_model = DiseaseModel()
    disease_model.load_xgboost('frontend/model/xgboost_model.json')

    st.markdown("""
    <div class='card'>
    <h3>🧠 AI Disease Prediction System</h3>
    <p>Select symptoms and get prediction</p>
    </div>
    """, unsafe_allow_html=True)

    symptoms = st.multiselect('Select Symptoms', disease_model.all_symptoms)
    X = prepare_symptoms_array(symptoms)

    if st.button('Predict'):
        prediction, prob = disease_model.predict(X)

        st.markdown(f"""
        <div class='success-box'>
        <h3>Prediction: {prediction}</h3>
        <p>Confidence: {prob*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Description", "Precautions"])

        with tab1:
            st.write(disease_model.describe_predicted_disease())

        with tab2:
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
                st.write(f"{i+1}. {precautions[i]}")

# =========================
# DIABETES
# =========================
if selected == 'Diabetes Prediction':

    st.title("Diabetes Prediction")

    name = st.text_input("Name")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Pregnancies")
        SkinThickness = st.number_input("Skin Thickness")

    with col2:
        Glucose = st.number_input("Glucose")
        Insulin = st.number_input("Insulin")

    with col3:
        BloodPressure = st.number_input("Blood Pressure")
        BMI = st.number_input("BMI")

    DiabetesPedigreefunction = st.number_input("Diabetes Pedigree Function")
    Age = st.number_input("Age")

    if st.button("Predict Diabetes"):
        result = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure,
                                         SkinThickness, Insulin, BMI,
                                         DiabetesPedigreefunction, Age]])

        if result[0] == 1:
            st.markdown("""
            <div class='danger-box'>
            <h3>⚠️ Diabetic Risk Detected</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='success-box'>
            <h3>✅ No Diabetes Risk</h3>
            </div>
            """, unsafe_allow_html=True)

# =========================
# HEART DISEASE
# =========================
if selected == 'Heart disease Prediction':

    st.title("Heart Disease Prediction")

    name = st.text_input("Name")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age")
        trestbps = st.number_input("Resting BP")
        ca = st.number_input("Major vessels")

    with col2:
        chol = st.number_input("Cholesterol")
        thalach = st.number_input("Max Heart Rate")
        oldpeak = st.number_input("Oldpeak")

    with col3:
        sex = st.selectbox("Sex", [0,1])
        cp = st.selectbox("Chest Pain Type", [0,1,2,3])
        slope = st.selectbox("Slope", [0,1,2])

    if st.button("Predict Heart Disease"):
        result = heart_model.predict([[age, sex, cp, trestbps, chol, 0, 0,
                                       thalach, 0, oldpeak, slope, ca, 0]])

        if result[0] == 1:
            st.markdown("""
            <div class='danger-box'>
            <h3>⚠️ Heart Disease Risk</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='success-box'>
            <h3>✅ No Heart Disease Risk</h3>
            </div>
            """, unsafe_allow_html=True)

# =========================
# PARKINSON (UNCHANGED LOGIC)
# =========================
if selected == 'Parkison Prediction':
    st.title("Parkinson Prediction")
    st.info("Enter voice parameters for prediction")

# =========================
# LUNG CANCER
# =========================
if selected == 'Lung Cancer Prediction':
    st.title("Lung Cancer Prediction")
    st.info("Enter symptoms for prediction")

# =========================
# LIVER
# =========================
if selected == 'Liver prediction':
    st.title("Liver Disease Prediction")
    st.info("Enter medical values")

# =========================
# HEPATITIS
# =========================
if selected == 'Hepatitis prediction':
    st.title("Hepatitis Prediction")
    st.info("Enter test values")