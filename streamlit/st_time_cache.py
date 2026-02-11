import streamlit as st
import time

@st.cache_data
def load_data():
     time.sleep(3)
     return "Data loaded!"
# Call the function

st.write(load_data())

############################

import streamlit as st
import time

@st.cache_resource
def load_model():
     time.sleep(5)
     return "Model loaded!"

st.write(load_model())