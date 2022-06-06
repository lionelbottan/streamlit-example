import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
import pickle
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import shap
shap.initjs() # for visualization

@st.cache
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

#Chargement du modele
picklefile = open("modeles/xgboost.pkl", "rb")
modele = pickle.load(picklefile)  

#Definition des features
features = ["RainToday_Num","Rain_J-1","Rain_J-2","MinTemp","MaxTemp","Sunshine","Evaporation",
    "Humidity3pm","Humidity9am","Pressure9am","Pressure3pm","Cloud3pm","Cloud9am", 
    "Wind9am_cos","Wind3pm_cos","WindGust_cos","Wind9am_sin","Wind3pm_sin","WindGust_sin", 
    "Mois","Clim_type_det"]
            
st.markdown("# Simulation")
st.sidebar.markdown("# Simulation")

st.subheader("Lecture des données")

Data = st.selectbox("DataFrame: " , ["echantillon","Sydney","AliceSprings","Darwin","Perth","Hobart"])

if ( Data == "echantillon"):
    df=pd.read_csv('data/echantillon.csv') #Read our data dataset
if ( Data == "Sydney"):
    df=pd.read_csv('data/Sydney.csv') #Read our data dataset
if ( Data == "AliceSprings"):
    df=pd.read_csv('data/AliceSprings.csv') #Read our data dataset
if ( Data == "Darwin"):
    df=pd.read_csv('data/Darwin.csv') #Read our data dataset
if ( Data == "Perth"):
    df=pd.read_csv('data/Perth.csv') #Read our data dataset
if ( Data == "Hobart"):
    df=pd.read_csv('data/Hobart.csv') #Read our data dataset    

st.write("Nombre de lignes : ", df.shape[0]) 
st.write("Nombre de colonnes : ", df.shape[1]) 

st.subheader("DataViz")

#Part des jours de pluie
fig = plt.figure(figsize=(3,3))
x = df.RainTomorrow_Num.value_counts(normalize=True)
colors = sns.color_palette('pastel')[0:5]
labels = ['Pas de pluie', 'Pluie']
plt.pie(x, labels = labels, colors = colors, autopct='%.0f%%')
plt.title("Part des jours de pluie")
st.write(fig)

if st.button("Correlation"):
    fig, ax = plt.subplots(figsize=(15,6))
    ListeCrit = ["RainTomorrow_Num","MinTemp","MaxTemp","Sunshine","Evaporation","Humidity3pm"]
    sns.heatmap(df[ListeCrit].corr(), cmap="YlGnBu",annot=True,ax=ax)
    st.write(fig)

if st.button("Distribution"):
    fig, ax = plt.subplots(figsize=(15,6))
    ax.title.set_text("Distribution annuelle des pluies")
    sns.lineplot(ax=ax,data=df, x="Mois", y="Rainfall")
    st.write(fig)

if st.button("Impact de RainTomorrow"):
    fig, ax = plt.subplots(figsize=(20,4))
    plt.subplot(131)
    sns.histplot(data=df, x="Sunshine",hue="RainTomorrow_Num",bins=20, multiple="layer", thresh=None)
    plt.subplot(132)
    sns.histplot(data=df, x="MinTemp",hue="RainTomorrow_Num",bins=20, thresh=None)
    plt.subplot(133)
    sns.histplot(data=df, x="Humidity3pm",hue="RainTomorrow_Num",bins=20)
    st.write(fig)


st.subheader("Prédiction")

if st.button("Predict"):  
#Courbe de ROC
    probs = modele.predict_proba(df[features])
    y_test =  df["RainTomorrow_Num"]
    fpr, tpr, seuils = sklearn.metrics.roc_curve(y_test, probs[:,1], pos_label=1)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    fig = plt.figure(figsize=(15,6))
    plt.plot(fpr, tpr, color='purple',  linestyle='--', lw=1, label='Model (auc = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle=':', label='Aléatoire (auc = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux faux positifs')
    plt.ylabel('Taux vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right");
    st.pyplot(fig)
#Predictions
    prediction = modele.predict(df[features])
    predDf = pd.DataFrame(prediction,columns=["prediction"])
    Sortie = pd.concat([df[["Date","Location","Climat_Koppen","Clim_type_det","RainTomorrow_Num"]],predDf],axis=1)
    st.write(Sortie)

st.subheader("Interprétabilité")
    
if st.button("Importance des features"):
    explainer = shap.TreeExplainer(modele)
    shap_values = explainer.shap_values(df[features])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st_shap(shap.summary_plot(shap_values, df[features]))