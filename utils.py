import pickle
import base64
import streamlit as st


def serialize_model(trained_mod):
    serialized_model = pickle.dumps(trained_mod)
    b64 = base64.b64encode(serialized_model).decode()
    href = f'<a href="data:file/serialized_model;base64,{b64}" download="trained_model.pkl">Download Trained Model</a>'
    st.markdown(href, unsafe_allow_html=True)
