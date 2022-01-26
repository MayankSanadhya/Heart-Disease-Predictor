import pandas as pd
import pickle
import streamlit as st

gender = ['M','F']
chest_pain_type = ['ASY','NAP','ATA','TA']
fasting_bs = ['Yes','No']
resting_ecg = ['Normal','LVH','ST']
exerciseAngina = ['Y','N']
stSlope = ['Flat','Up','Down']

pipe = pickle.load(open('pipe.pkl','rb'))

st.title('Heart Disease Predictor')

col1,col2 = st.columns(2)

with col1:
    age = st.number_input("Age")

with col2:
    sex = st.selectbox("Select the Gender",gender)

chest_pain = st.selectbox("Select the chest pain type",chest_pain_type)

col3,col4,col5 = st.columns(3)

with col3:
    BP = st.number_input("Resting Blood Pressure")

with col4:
    cholestrol = st.number_input("Cholestrol")

with col5:
    fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120mg/dl',fasting_bs)

Resting_ECG = st.selectbox('Select the Resting ECG',resting_ecg)

col6,col7,col8 = st.columns(3)

with col6:
    max_heart_rate = st.number_input('Max. Heart Rate')

with col7:
    exercise_angina = st.selectbox('Exercise Angina - Y/N',exerciseAngina)

with col8:
    oldpeak = st.number_input('OldPeak',min_value=-5,max_value=10)

st_slope = st.selectbox('Select ST Slope',stSlope)

if st.button('Predict Probability'):
    input_df = pd.DataFrame({'Age':[age],'Sex':[sex],'ChestPainType':[chest_pain],'RestingBP':[BP],'Cholesterol':[cholestrol],
                             'FastingBS':[fasting_blood_sugar],'RestingECG':[Resting_ECG],'MaxHR':[max_heart_rate],
                             'ExerciseAngina':[exercise_angina],'Oldpeak':[oldpeak],'ST_Slope':[st_slope]})
    st.table(input_df)

    result = pipe.predict_proba(input_df)
    Yes = result[0][0]

    st.subheader('Chance of having Heart Disease' + "- " + str(round(Yes * 100)) + "%")

