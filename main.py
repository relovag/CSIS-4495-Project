import streamlit as st
import constants
import model_builder


def main():
    st.set_page_config("Auto ML", "assets/favicon.ico")
    st.markdown(constants.HIDE_DEFAULTS, unsafe_allow_html=True)
    st.title("Automated Machine Learning CSIS 4495")
    st.subheader("Zito Relova 300306471")

    file = st.file_uploader("Upload Dataset", type=["csv", "txt"])

    if file:
        builder = model_builder.ModelBuilder(file)
        builder.run_model_builder()


if __name__ == "__main__":
    main()
