import streamlit as st

st.markdown("# Page 2 ❄️")
st.sidebar.markdown("# Page 2 ❄️")

from PIL import Image
image = Image.open('Climats.jpg')

st.image(image, caption='Climats')
