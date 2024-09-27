import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load_model('anarchy_prediction_model.h5')

# Load and fit the scaler
data = pd.read_csv('game_theory_anarchy_dataset.csv')
X = data.drop(columns=['anarchy'])
scaler = StandardScaler().fit(X)

# Streamlit app
st.title("Anarchy Prediction from Game Theory Scenarios")

# Example input fields
player_strategy = st.number_input("Player Strategy", min_value=0.0, max_value=1.0, value=0.5)
payoff = st.number_input("Payoff", min_value=-10.0, max_value=10.0, value=0.0)
nash_equilibrium = st.number_input("Nash Equilibrium", min_value=0.0, max_value=1.0, value=0.5)

# Predict button
if st.button("Predict Anarchy"):
    # Prepare the input vector
    input_features = np.array([player_strategy, payoff, nash_equilibrium])
    input_features = scaler.transform([input_features])
    
    # Make the prediction
    prediction = model.predict(input_features)
    
    # Display the result
    if prediction[0] > 0.5:
        st.write("The scenario likely represents anarchy.")
    else:
        st.write("The scenario likely does not represent anarchy.")
