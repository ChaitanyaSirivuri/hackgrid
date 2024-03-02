import streamlit as st
import pandas as pd
import os
import pandas_profiling
import streamlit_pandas_profiling
import classification
import regression
import clustering
import summary

import warnings

warnings.filterwarnings("ignore")

# Initialize session state
if "model_filename" not in st.session_state:
    st.session_state.model_filename = None

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image("./img.png")
    st.title("AutoML")
    choice = st.radio(
        "Navigation", ["Upload", "Exploratory data analysis", "Model Training", "Summary"])
    st.info("This application runs through all possible machine learning models for any given Machine Learning Technique and finds the most optimal model for your dataset.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset", type=["csv"])
    if file:
        df = pd.read_csv(file, index_col=None)
        st.dataframe(df)
        df.to_csv('dataset.csv', index=None)

if choice == "Exploratory data analysis":
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    streamlit_pandas_profiling.st_profile_report(profile_df)

if choice == "Model Training":
    st.title("Model Training")
    model = st.selectbox(
        "Select Model", ["Classification", "Regression", "Clustering"])
    if model == "Classification":
        st.title("Classification")
        target = st.selectbox("Select Target Variable", df.columns)
        if st.button("Run"):
            results, model_filename = classification.start_classification(
                df, target)
            st.dataframe(results)
            st.session_state.model_filename = model_filename
            st.success(
                f"Models trained successfully")
        if st.button("Save") and st.session_state.model_filename:
            classification.save_best_model(
                model, st.session_state.model_filename)
            st.success(
                f"Model saved as {st.session_state.model_filename}")
    if model == "Regression":
        st.title("Regression")
        target = st.selectbox("Select Target Variable", df.columns)
        if st.button("Run"):
            results, model_filename = regression.start_regression(df, target)
            st.dataframe(results)
            st.session_state.model_filename = model_filename
            st.success(
                f"Models trained successfully")
        if st.button("Save") and st.session_state.model_filename:
            regression.save_best_model(
                model, st.session_state.model_filename)
            st.success(
                f"Model saved as {st.session_state.model_filename}")

    if model == "Clustering":
        st.title("Clustering")
        if st.button("Run"):
            results, model_filename = clustering.start_clustering(df)
            st.dataframe(results)
            st.session_state.model_filename = model_filename
            st.success(
                f"Models trained successfully")
        if st.button("Save") and st.session_state.model_filename:
            clustering.save_best_model(
                model, st.session_state.model_filename)
            st.success(
                f"Model saved as {st.session_state.model_filename}")

if choice == "Summary":
    st.title("Summary")
    # extract columns from df
    columns = df.columns[0:-1]
    types = df.iloc[-1].values
    st.markdown(summary.generate_prompt(columns, types))
