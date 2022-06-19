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

from Simulations import Simulations
from Modelisations import Modelisations


def main():
    st.sidebar.title("RainsBerry")
    
    #st.set_page_config(
    #page_title="RainsBerry - Météo",
    #page_icon="👋",
    #layout="wide",)

    Menu = st.sidebar.radio(
     "Menu",
     ('Le Projet Météo', 'PreProcessing','DataViz','Modelisations','Performances','Simulations','Conclusion'))

    if Menu == 'Le Projet Météo':
        from PIL import Image
        image = Image.open('images/RainsBerry.jpg')
        st.image(image, caption='RainsBerry')
    if Menu == 'PreProcessing':
        PreProcessing()
    if Menu == 'DataViz':
        DataViz()
    if Menu == 'Modelisations':
        Modelisations()       
    if Menu == 'Simulations':
        simulation()


    st.sidebar.text("")
    st.sidebar.text("Projet DataScientest")
    st.sidebar.text("Promotion DataScientist Octobre 2021")
    st.sidebar.text("Lionel Bottan")  
    st.sidebar.text("Julien Coquard")  
    st.sidebar.text("Samuel Guérin")      
    
def PreProcessing():
    from PIL import Image
    image = Image.open('images/Climats.jpg')
    st.image(image, caption='Climats')
    
def DataViz():
    st.header("DataViz")


if __name__ == "__main__":
    main()
