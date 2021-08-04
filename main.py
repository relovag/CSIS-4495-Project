import streamlit as st
import constants
import model_builder
import explore_data


def main():
    st.set_page_config(*constants.PAGE_CONFIG)
    st.markdown(constants.HIDE_DEFAULTS, unsafe_allow_html=True)
    st.title(constants.APPLICATION_TITLE)
    st.subheader(constants.APPLICATION_SUBHEADER)

    file = st.file_uploader("Upload Dataset", type=["csv", "txt"])
    choice = st.sidebar.selectbox("Select Activity", constants.ACTIVITIES, 1)

    if file:
        if choice == "Model Building":
            builder = model_builder.ModelBuilder(file)
            builder.run_model_builder()
        elif choice == "EDA":
            eda = explore_data.EDA(file)
            eda.run_eda()


if __name__ == "__main__":
    main()
