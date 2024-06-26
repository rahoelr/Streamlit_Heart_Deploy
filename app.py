import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/rahoelr/UTS_AI/main/heart.csv')
    return df

df = load_data()

# Preprocess data
def preprocess_data(df):
    df_X = df.drop(['HeartDisease'], axis=1)
    df_y = df['HeartDisease']
    encoder = LabelEncoder()
    for col in df_X.columns:
        if df_X[col].dtype == 'object':
            df_X[col] = encoder.fit_transform(df_X[col])
    X = df_X.astype(float).values
    y = df_y.astype(float).values
    return X, y

X, y = preprocess_data(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit UI
st.title('Heart Disease Prediction')

col1, col2 = st.columns(2)

# Create form for user inputs
with st.form(key="information", clear_on_submit=True):
    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=40)
        chest_pain_type = st.selectbox('Chest Pain Type', ('ATA', 'NAP', 'ASY', 'TA'))
        cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
        resting_ecg = st.selectbox('Resting ECG', ('Normal', 'ST', 'LVH'))
        exercise_angina = st.selectbox('Exercise Angina', ('N', 'Y'))
        st_slope = st.selectbox('ST Slope', ('Up', 'Flat', 'Down'))
    with col2:
        sex = st.selectbox('Sex', ('M', 'F'))
        resting_bp = st.number_input('Resting BP', min_value=0, max_value=300, value=120)
        fasting_bs = st.selectbox('Fasting BS', (0, 1))
        max_hr = st.number_input('Max HR', min_value=0, max_value=220, value=120)
        oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=0.0)

    # Submit button inside the form
    submit_button = st.form_submit_button(label='Predict')

# Check if the form has been submitted
if submit_button:
    # Convert input data to dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain_type],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    # Encode the input data automatically using LabelEncoder
    for column in input_data.columns:
        if input_data[column].dtype == 'object':
            le = LabelEncoder()
            input_data[column] = le.fit_transform(input_data[column])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    input_data_scaled = pd.DataFrame(input_data_scaled, columns=input_data.columns)  # Convert back to DataFrame

    # Predict using the model
    prediction = model.predict(input_data_scaled)[0]

    # Save the prediction
    def save_prediction(data, prediction):
        csv_filename = 'prediction_history.csv'
        if not os.path.isfile(csv_filename):
            df = pd.DataFrame(columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'Prediction'])
            df.to_csv(csv_filename, index=False)
        with open(csv_filename, 'a') as f:
            data['Prediction'] = prediction
            df = pd.DataFrame(data, index=[0])
            df.to_csv(f, header=False, index=False)
    
    save_prediction(input_data.iloc[0].to_dict(), prediction)

    # Display the prediction result
    if prediction == 1:
        st.write('The model predicts that this person **has heart disease**.')
    else:
        st.write('The model predicts that this person **does not have heart disease**.')

# Display Prediction History
if st.checkbox('Show Prediction History'):
    if os.path.isfile('prediction_history.csv'):
        history = pd.read_csv('prediction_history.csv')
        st.write(history)
    else:
        st.write("No prediction history available.")
