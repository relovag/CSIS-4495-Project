from re import S
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import base64
import constants
from sklearn import model_selection
from metrics import metrics
from models import models
from section import Section
import SessionState


class ModelBuilder(Section):
    def __init__(self, file):
        super(ModelBuilder, self).__init__(file)
        self.state = SessionState.get(train_model=False)

    def run_model_builder(self):
        st.subheader("Model Building")
        st.text("")

        self.train_model()

    def train_model(self):
        cols = self.df.columns.tolist()
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

        st.subheader("Model Selection")
        # chosen_model = st.empty()
        # compare_models = st.checkbox("Compare different models")

        # with chosen_model:
        #     if compare_models:
        #         chosen_model = st.multiselect(
        #             "Select your models: ", list(models.keys())
        #         )
        #     else:
        chosen_model = st.selectbox("Select your model: ", list(models.keys()))

        st.subheader("Metric Selection")
        chosen_metric = st.selectbox("Select your metric", list(metrics.keys()))

        if st.button("Train Model") or self.state.train_model:
            self.state.train_model = True
            self.chosen_feats = chosen_feats
            self.target = y
            feats = pd.get_dummies(self.df[chosen_feats])
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                feats, self.df[y], random_state=constants.RANDOM_STATE
            )
            trained_model = models[chosen_model].fit(X_train, y_train)
            self.trained_model = trained_model

            pred = self.trained_model.predict(X_test)
            metric = metrics[chosen_metric]
            metric_score = metric[0](y_test, pred, **metric[1])
            st.write(
                f"Your chosen model {chosen_model}, scored {metric_score:.2f} on {X_test.shape[0]} test cases."
            )
            st.subheader(f"Download your trained model")

            self.serialize_model(self.trained_model)
            self.predict_on_new_file()
            self.predict_interface()

    def predict_on_new_file(self):
        try:
            st.subheader("You can also make predictions on new data below")
            new_file = st.file_uploader("New Prediction Data", type=["csv", "txt"])
            if new_file:
                pred_data = pd.read_csv(new_file)
                feats = pd.get_dummies(pred_data[self.chosen_feats])
                preds = self.trained_model.predict(feats)
                st.dataframe(preds)

        except Exception as ex:
            st.write("There was an error reading your prediction file")
            st.write(ex)

    def predict_interface(self):
        try:
            feat_list = []
            for feat in self.df[self.chosen_feats].select_dtypes("number"):
                feat_list.append(
                    st.slider(
                        feat, float(self.df[feat].min()), float(self.df[feat].max())
                    )
                )
            make_prediction = st.button("Predict")
            if make_prediction:
                pred = self.make_individual_prediction(feat_list)
                pred_string = (
                    f"<h3>The model predicted <strong>{pred[0]}</strong>.</h3>"
                )
                st.markdown(pred_string, unsafe_allow_html=True)
                # st.subheader(f"The model predicted {pred[0]}")

        except Exception as ex:
            st.write("There was an error creating the prediction interface")
            st.write(ex)

    def make_individual_prediction(self, feats):
        feats = np.array(feats).reshape(1, -1)
        pred = self.trained_model.predict(feats)
        return pred

    def serialize_model(self, trained_model):
        try:
            serialized_model = pickle.dumps(trained_model)
            b64 = base64.b64encode(serialized_model).decode()
            href = f'<a href="data:file/serialized_model;base64,{b64}" \
                download="trained_model.pkl">Download Trained Model</a>'
            st.markdown(href, unsafe_allow_html=True)

        except Exception as ex:
            st.write("There was an error serializing your trained model")
            st.write(ex)
