import streamlit as st
from src.data_management import load_housing_data, load_pkl_file
import numpy as np
import pandas as pd
import joblib

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
    inherited_df = pd.read_csv("outputs/datasets/cleaned/inherited_houses_cleaned.csv")

    # predict prices of inherited houses with ML pipeline from 05-Regression pipeline notebook
    st.write("### House sale prices from client's inherited houses")
    st.info(f"* The table below shows the four inherited houses profile")
    
    st.write(inherited_df.head())
    inherited_df = inherited_df.filter(best_features)
    house_price_prediction = pipeline.predict(inherited_df).round(0)
    inherited_df['Predicted House Sale Price'] = house_price_prediction
    st.write(
        f"* The table below shows the predicted sale prices for the four houses, together with the house features used in the prediction, "
        f"which are the four most important variables we saw in the House Price Study page: 'OverallQuality', 'TotalBsmtSF', '2ndFlrSF' and 'GarageArea'."
    )
    st.write(inherited_df.head())

    # calculate sum of inherited houses predicted prices
    sum = inherited_df['Predicted House Sale Price'].sum()
    st.write(
        f"* The sum of the predicted sale prices for the four houses is: &nbsp; &nbsp; &nbsp;{sum}  \n"
    )

    st.write("---")

    # predict price of any other house in Ames, Iowa
    st.write("### Predict house sale prices in Ames, Iowa  \n")
    st.write("* The following 4 variables 'Overall Quality',  "
             "'TotalBsmtSF', '2ndFlrSF', and 'GarageArea' are needed for the ML model to predict the price.")

    # create input fields for live data
    X_live = DrawInputsWidgets()
    # predict on live data
    if st.button("Run Predictive Analysis"):
        house_price_prediction = pipeline.predict(X_live.filter(best_features)).round(0)
        st.write(
            f"* The predicted sale price for the house is: &nbsp; &nbsp; &nbsp;{house_price_prediction[0]}  \n"
        )

def DrawInputsWidgets():

    df = load_housing_data()
    percentageMin, percentageMax = 0.5, 1.0

    # we create input widgets for 4 features
    col1, col2, col3, col4 = st.beta_columns(4)

    # We are using these features to feed the ML pipeline

    # create an empty DataFrame, which will be the live data
    X_live = pd.DataFrame([], index=[0])

    # from here on we draw the widget based on the variable type (numerical or categorical)
    # and set initial values
    with col1:
        feature = 'OverallQual'
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min() * percentageMin,
            max_value=df[feature].max() * percentageMax,
            value=df[feature].median()
        )
        X_live[feature] = st_widget

    with col2:
        feature = 'TotalBsmtSF'
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min() * percentageMin,
            max_value=df[feature].max() * percentageMax,
            value=df[feature].median()
        )
        X_live[feature] = st_widget

    with col3:
        feature = '2ndFlrSF'
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min() * percentageMin,
            max_value=df[feature].max() * percentageMax,
            value=df[feature].median()
        )
        X_live[feature] = st_widget

    with col4:
        feature = 'GarageArea'
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min() * percentageMin,
            max_value=df[feature].max() * percentageMax,
            value=df[feature].median()
        )
        X_live[feature] = st_widget

    return X_live
