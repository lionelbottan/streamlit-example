import streamlit as st
from streamlit_shap import st_shap

import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import shap
shap.initjs() # for visualization

def main():
    st.sidebar.title("RainsBerry")
    
    st.set_page_config(
    page_title="RainsBerry - Météo",
    page_icon="👋",
    layout="wide",)

    from PIL import Image
    image = Image.open('images/RainsBerry.jpg')
    st.image(image, caption='RainsBerry')

    Simu = st.checkbox('Simulations')
    if Simu:
        simulation()

if __name__ == "__main__":
    main()
