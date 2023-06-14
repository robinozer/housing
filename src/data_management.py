import streamlit as st
import pandas as pd
import numpy as np
import joblib
from converters import ConvertToInt64


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_housing_data():
    df = pd.read_csv("outputs/datasets/collection/house_prices_records.csv")
    return df

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_cleaned_housing_data():
    df = pd.read_csv("outputs/datasets/cleaned/house_prices_records_cleaned.csv")
    return df

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_cleaned_inherited_houses_data():
    df = pd.read_csv("outputs/datasets/cleaned/inherited_houses_cleaned.csv")
    return df

def load_pkl_file(file_path):
    return joblib.load(file_path)