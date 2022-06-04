import streamlit as st

st.markdown("# Simulation")
st.sidebar.markdown("# Simulation")

model = joblib.load('modeles/xgbc_ru.joblib')