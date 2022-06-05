
st.markdown("# Simulation")
st.sidebar.markdown("# Simulation")

picklefile = open("modeles/xgboost.pkl", "rb")
model = pickle.load(picklefile)

df=pd.read_csv('data/echantillon.csv') #Read our data dataset
st.write(df.head()) 