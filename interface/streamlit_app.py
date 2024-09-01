# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the saved model
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Streamlit app
st.title('Municipal Solid Waste Prediction')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Preprocessing steps as in the training script
    data['ticket_date'] = pd.to_datetime(data['ticket_date'], errors='coerce')
    data.set_index('ticket_date', inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns
    data.drop(non_numeric_columns, axis=1, inplace=True)
    data['week_of_year'] = data.index.isocalendar().week
    data['day_of_year'] = data.index.dayofyear
    data['is_month_start'] = data.index.is_month_start.astype(int)
    data['is_month_end'] = data.index.is_month_end.astype(int)

    features = ['lag_1', 'lag_2', 'rolling_7', 'day_of_week', 'month', 'week_of_year', 'day_of_year', 'is_month_start', 'is_month_end']
    X = pd.get_dummies(data[features], columns=['day_of_week', 'month'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    predictions = best_model.predict(X_scaled)

    st.write('Predictions:')
    st.write(predictions)
    
    st.line_chart(predictions)
