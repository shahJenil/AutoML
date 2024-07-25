import streamlit as st
import pandas as pd
import os
from pydantic_settings import BaseSettings
import ydata_profiling

from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

st.set_page_config(
    page_title="AutoML"
)
with st.sidebar:
    st.title("BestModel")
    st.info("The most accurate Machine Learning model for your dataset is now just a click away!")
    choice = st.radio("Navigation", ["Upload","Profiling","Models","Download"])

if choice == "Upload":
    st.title("Upload your Data for Modelling")
    file = st.file_uploader("Upload your dataset : ")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Data Analysis on Provided Dataset")
    try:
        profile_report = df.profile_report()
        st_profile_report(profile_report)
    except NameError as e:
        st.error("Please Upload A Dataset.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    
    

if choice == "Models":
    st.title("Automated Training To Find The Best Machine Learning Model")
    try:
        target = st.selectbox("Select your target :", df.columns)
        if st.button("Train Model"):
            setup(df,target=target, verbose=False)
            setup_df = pull()
            st.info("This is the ML experiment settings")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info("This is the Machine Learning Model")
            st.dataframe(compare_df)
            best_model
            save_model(best_model, 'best_model')
    except NameError as e:
        st.error("Please Upload A Dataset.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

if choice == "Download":
    st.title("Download The Best Model In '.pkl' Format")
    try:
        with open("best_model.pkl", 'rb') as f:
            st.download_button("Download The Model", f, 'trained_model.pkl')
    except NameError as e:
        st.error("Please Upload A Dataset.")
    except Exception as e:
        st.error(f"Please Upload A Dataset.")

