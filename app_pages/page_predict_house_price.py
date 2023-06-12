import streamlit as st
from src.data_management import (
    load_housing_data,
    load_pkl_file,
    load_cleaned_inherited_houses_data,
)
import numpy as np
import pandas as pd


def page_predict_house_price_body():

    # load files for predicting house prices
    version = "v2"
    pipeline = load_pkl_file(
        f"outputs/ml_pipeline/predict_housing/{version}/best_regressor_pipeline.pkl"
    )
    best_features = pd.read_csv(
        f"outputs/ml_pipeline/predict_housing/{version}/X_train.csv"
    ).columns.to_list()

    # this is the inherited houses (cleaned to have matching data types with the main dataset)
    inherited_df = pd.read_csv(
        "outputs/datasets/cleaned/inherited_houses_cleaned.csv")
    


    st.write("### Predict sale prices on inherited houses")
    st.info( #text borrowed from project description handbook
        f"* The client is interested in predicting the house sales price"
        f" from her four inherited houses, and any other house in Ames, Iowa."
    )
    st.write("---")





def DrawInputsWidgets():

	df = load_housing_data()
	percentageMin, percentageMax = 0.5, 1.0

    # we create input widgets for 4 features	
	col1, col2, = st.columns(2)

	# We are using these features to feed the ML pipeline
		
 	# create an empty DataFrame, which will be the live data
	X_live = pd.DataFrame([], index=[0]) 
	
	# from here on we draw the widget based on the variable type (numerical or categorical)
	# and set initial values
	with col1:
		feature = 'GrLivArea'
		st_widget = st.number_input(
	 		label= feature,
			min_value= df[feature].min()*percentageMin,
			max_value= df[feature].max()*percentageMax,
			value= df[feature].median()
			)
	X_live[feature] = st_widget


	with col2:
		feature = "OverallQual"
		st_widget = st.selectbox(
			label= feature,
			options= df[feature].sort_values(ascending=True).unique()
			)
	X_live[feature] = st_widget

	return X_live
    