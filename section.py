import pandas as pd
import streamlit as st


class Section:
    def __init__(self, file) -> None:
        self.file = file
        self.process_file()

    def process_file(self):
        try:
            df = pd.read_csv(self.file)
            st.write(f"Number of rows: {df.shape[0]}")
            st.write(f"Number of columns: {df.shape[1]}")
            st.write("Dataset summary")
            st.table(df.describe())
            self.df = df

        except Exception as ex:
            st.write("There was an error reading your data file")
            st.write(ex)
