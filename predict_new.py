import streamlit as st
import pandas as pd


def make_new_preds():
    st.write(
        "Your model has been trained. Do you want to make predictions on new data?"
    )
    yes = st.button("Yes")
    no = st.button("No")

    # Right now we can't use multiple uploaders on the same page
    if yes:
        pred_data = st.file_uploader(
            "Upload Dataset To Make New Predictions", type=["csv", "txt"]
        )
        pred_df = pd.read_csv(pred_data)
