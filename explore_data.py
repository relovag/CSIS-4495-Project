import streamlit as st
from section import Section
import seaborn as sns
import matplotlib.pyplot as plt


class EDA(Section):
    def __init__(self, file) -> None:
        super(EDA, self).__init__(file)

    def run_eda(self):
        st.subheader("Exploratory Data Analysis")

        if st.checkbox("Correlation Plot"):
            self.create_heatmap()

        plot_type = st.selectbox("Select Plot Type", ["area", "bar", "line"])
        features_container = st.beta_container()
        chosen_feats = features_container.multiselect(
            "Choose your features", self.df.columns.tolist()
        )

        if st.button("Generate Plot"):
            if plot_type == "area":
                st.area_chart(self.df[chosen_feats])
            elif plot_type == "bar":
                st.bar_chart(self.df[chosen_feats])
            elif plot_type == "line":
                st.line_chart(self.df[chosen_feats])

    def create_heatmap(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(self.df.select_dtypes("number").corr(), annot=True, ax=ax)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        st.pyplot(fig)
