
st.markdown("# Simulation")
st.sidebar.markdown("# Simulation")

model = joblib.load('modeles/xgbc_ru.joblib')

picklefile = open("modeles/xgboost.pkl", "rb")
    model = pickle.load(picklefile)