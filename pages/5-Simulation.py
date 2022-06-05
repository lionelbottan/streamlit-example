import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

st.markdown("# Simulation")
st.sidebar.markdown("# Simulation")

st.subheader("Lecture des données")

df=pd.read_csv('data/echantillon.csv') #Read our data dataset
st.write(df.head()) 

if st.button("Predict"):
    picklefile = open("modeles/dt.pkl", "rb")
    modele = pickle.load(picklefile)
    
    features = ["RainToday_Num","Rain_J-1","Rain_J-2","MinTemp","MaxTemp","Sunshine","Evaporation",
            "Humidity3pm","Humidity9am","Pressure9am","Pressure3pm","Cloud3pm","Cloud9am", 
            "Wind9am_cos","Wind3pm_cos","WindGust_cos","Wind9am_sin","Wind3pm_sin","WindGust_sin", 
            "Mois","Clim_type_det"]
    prediction = modele.predict(df)
    st.write(prediction)