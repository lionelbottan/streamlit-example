import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.markdown("# Simulation")
st.sidebar.markdown("# Simulation")

model = joblib.load('modeles/xgbc_ru.joblib')