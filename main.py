import pandas as pd
import streamlit as st
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from models import models
from predict_new import make_new_preds
from utils import serialize_model

random_state = 42


def main():
    st.set_page_config(layout="wide")
    st.title("Automated Machine Learning CSIS 4495")
    st.subheader("Zito Relova 300306471")

    data = st.file_uploader("Upload Dataset", type=["csv", "txt"])
    if data:
        df = pd.read_csv(data)
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        st.write("Dataset head")
        st.dataframe(df.head())

        cols = df.columns.tolist()
        y = st.selectbox("Select the target variable: ", cols)

        if y:
            features_container = st.beta_container()
            select_all = st.checkbox("Select all")
            feats_list = cols.copy()
            feats_list.remove(y)
            select_args = ["Choose your features", feats_list]

            if select_all:
                select_args.append(feats_list)
                chosen_feats = features_container.multiselect(*select_args)
            else:
                chosen_feats = features_container.multiselect(*select_args)

        chosen_model = st.selectbox("Select your model: ", list(models.keys()))
        scoring = "accuracy"
        if st.button("Train Model"):
            # kfold = model_selection.KFold()
            feats = pd.get_dummies(df[chosen_feats])
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                feats, df[y], random_state=random_state
            )
            trained_mod = models[chosen_model].fit(X_train, y_train)
            pred = trained_mod.predict(X_test)
            acc = accuracy_score(y_test, pred)
            # cv = model_selection.cross_val_score(models[chosen_model], feats, df[y], cv=kfold, scoring=scoring)
            st.write(f"Your model scored {acc:.2f}")
            # make_new_preds()
            serialize_model(trained_mod)


if __name__ == "__main__":
    main()
