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
    
    prediction = modele.predict(df)
    st.write(prediction) 