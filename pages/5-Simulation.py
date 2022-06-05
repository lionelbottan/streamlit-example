import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.markdown("# Simulation")
st.sidebar.markdown("# Simulation")

df=pd.read_csv('data/echantillon.csv') #Read our data dataset
st.write(df.head()) 


picklefile = open("modeles/dt.pkl", "rb")
model = pickle.load(picklefile)