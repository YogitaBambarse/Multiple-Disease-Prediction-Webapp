import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from streamlit_option_menu import option_menu

# -------------------- LOAD MODELS --------------------
diabetes_model = joblib.load("models/diabetes_model.sav")
heart_model = joblib.load("models/heart_disease_model.sav")
parkinson_model = joblib.load("models/parkinsons_model.sav")
liver_model = joblib.load("models/liver_model.sav")
lung_cancer_model = joblib.load("models/lung_cancer_model.sav")
hepatitis_model = joblib.load("models/hepatitis_model.sav")

# -------------------- SIDEBAR --------------------
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction',
        [
            'Diabetes Prediction',
            'Heart Prediction',
            'Parkinson Prediction',
            'Liver Prediction',
            'Lung Cancer Prediction',
            'Hepatitis Prediction'
        ],
        icons=['activity', 'heart', 'person', 'person', 'person', 'person'],
        default_index=0
    )

# -------------------- DIABETES --------------------
if selected == 'Diabetes Prediction':
    st.title("Diabetes Prediction")

    name = st.text_input("Name")

    Pregnancies = st.number_input("Pregnancies")
    Glucose = st.number_input("Glucose")
    BloodPressure = st.number_input("Blood Pressure")
    SkinThickness = st.number_input("Skin Thickness")
    Insulin = st.number_input("Insulin")
    BMI = st.number_input("BMI")
    DPF = st.number_input("Diabetes Pedigree Function")
    Age = st.number_input("Age")

    if st.button("Predict"):
        result = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure,
                                          SkinThickness, Insulin, BMI, DPF, Age]])
        if result[0] == 1:
            st.error(name + " → Diabetic")
        else:
            st.success(name + " → Not Diabetic")

# -------------------- HEART --------------------
elif selected == 'Heart Prediction':
    st.title("Heart Disease Prediction")

    age = st.number_input("Age")
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Resting BP")
    chol = st.number_input("Cholesterol")
    fbs = st.selectbox("Fasting Sugar", [0,1])
    restecg = st.selectbox("Rest ECG", [0,1,2])
    thalach = st.number_input("Max Heart Rate")
    exang = st.selectbox("Exercise Angina", [0,1])
    oldpeak = st.number_input("Oldpeak")
    slope = st.selectbox("Slope", [0,1,2])
    ca = st.number_input("CA")
    thal = st.selectbox("Thal", [0,1,2])

    if st.button("Predict"):
        result = heart_model.predict([[age, sex, cp, trestbps, chol, fbs,
                                       restecg, thalach, exang, oldpeak,
                                       slope, ca, thal]])
        if result[0] == 1:
            st.error("Heart Disease Detected")
        else:
            st.success("No Heart Disease")

# -------------------- PARKINSON --------------------
elif selected == 'Parkinson Prediction':
    st.title("Parkinson Prediction")

    features = [st.number_input(f"Feature {i}") for i in range(22)]

    if st.button("Predict"):
        result = parkinson_model.predict([features])
        if result[0] == 1:
            st.error("Parkinson Detected")
        else:
            st.success("No Parkinson")

# -------------------- LIVER --------------------
elif selected == 'Liver Prediction':
    st.title("Liver Disease Prediction")

    Sex = st.selectbox("Gender", [0,1])
    age = st.number_input("Age")
    TB = st.number_input("Total Bilirubin")
    DB = st.number_input("Direct Bilirubin")
    AP = st.number_input("Alkaline Phosphatase")
    ALT = st.number_input("ALT")
    AST = st.number_input("AST")
    TP = st.number_input("Total Proteins")
    ALB = st.number_input("Albumin")
    AGR = st.number_input("Albumin/Globulin Ratio")

    if st.button("Predict"):
        result = liver_model.predict([[Sex, age, TB, DB, AP, ALT, AST, TP, ALB, AGR]])
        if result[0] == 1:
            st.error("Liver Disease Detected")
        else:
            st.success("No Liver Disease")

# -------------------- LUNG CANCER --------------------
elif selected == 'Lung Cancer Prediction':
    st.title("Lung Cancer Prediction")

    age = st.number_input("Age")
    smoking = st.selectbox("Smoking", [1,2])
    anxiety = st.selectbox("Anxiety", [1,2])
    fatigue = st.selectbox("Fatigue", [1,2])

    if st.button("Predict"):
        data = pd.DataFrame([[age, smoking, anxiety, fatigue]],
                            columns=['AGE','SMOKING','ANXIETY','FATIGUE'])

        result = lung_cancer_model.predict(data)

        if result[0] == 1:
            st.error("Lung Cancer Risk")
        else:
            st.success("No Risk")

# -------------------- HEPATITIS --------------------
elif selected == 'Hepatitis Prediction':
    st.title("Hepatitis Prediction")

    age = st.number_input("Age")
    sex = st.selectbox("Sex", [1,2])
    alb = st.number_input("ALB")
    alp = st.number_input("ALP")
    alt = st.number_input("ALT")
    ast = st.number_input("AST")

    if st.button("Predict"):
        data = pd.DataFrame([[age, sex, alb, alp, alt, ast]],
                            columns=['Age','Sex','ALB','ALP','ALT','AST'])

        result = hepatitis_model.predict(data)

        if result[0] == 1:
            st.error("Hepatitis Detected")
        else:
            st.success("No Hepatitis")
