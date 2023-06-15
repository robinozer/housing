import streamlit as st
import pandas as pd
import joblib
from src.data_management import load_cleaned_housing_data, load_pkl_file


def page_predict_house_price_body():
    st.write("### House Price Predictor")
    # Load the pipeline directly using joblib.load
    version = "v3"
    pipeline = joblib.load(f"outputs/ml_pipeline/predict_housing/{version}/best_regressor_pipeline.pkl")
    best_features = pd.read_csv(f"outputs/ml_pipeline/predict_housing/{version}/X_train.csv").columns.to_list()
    

    # this are the inherited houses
    inherited_df = pd.read_csv("outputs/datasets/cleaned/inherited_houses_cleaned.csv")

    # predict prices of inherited houses with ML pipeline from 05-Regression Pipeline notebook
    st.write("### Sale prices for the client's 4 inherited houses")
    st.info("* The table below displays the profile of the four inherited houses")

    st.write(inherited_df.head())
    inherited_df = inherited_df.filter(best_features)
    house_price_prediction = pipeline.predict(inherited_df).round(0)
    inherited_df['Predicted House Sale Price'] = house_price_prediction
    st.info(
        "* The table below displays predicted sale prices for each of the "
        "four inherited houses, along with the features that were used in the prediction.\n "
        "* These were the four most important features that the model was trained on: "
        "'OverallQuality', 'TotalBsmtSF', '2ndFlrSF', and 'GarageArea'."
    )
    st.write(inherited_df.head())

    # calculate sum of inherited houses predicted prices
    sum_prices = inherited_df['Predicted House Sale Price'].sum()
    st.write(
        f"* In total, the predicted sale price for all of the four houses is: {sum_prices}\n"
    )

    st.write("---")

    # predict price of any other house in Ames, Iowa
    st.write("### Predict the sale price for any house in Ames, Iowa  \n")
    st.info("* Please enter the values for the following 4 variables "
            "for the ML model to predict the price.\n"
            "Please note there may be some discrepancy between predicted and actual sale price.")

    # create input fields for live data
    X_live = DrawInputsWidgets()
    # predict on live data
    if st.button("Run Predictive Analysis"):
        house_price_prediction = pipeline.predict(X_live.filter(best_features)).round(0)
        st.write(
            f"* The predicted sale price for the house is (in USD): {house_price_prediction[0]}\n"
        )


def DrawInputsWidgets():
    df = load_cleaned_housing_data()
    percentage_min, percentage_max = 0.5, 1.0

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
            min_value=df[feature].min() * percentage_min,
            max_value=df[feature].max() * percentage_max,
            value=df[feature].median()
        )
        X_live[feature] = st_widget

    with col2:
        feature = 'TotalBsmtSF'
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min() * percentage_min,
            max_value=df[feature].max() * percentage_max,
            value=df[feature].median()
        )
        X_live[feature] = st_widget

    with col3:
        feature = '2ndFlrSF'
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min() * percentage_min,
            max_value=df[feature].max() * percentage_max,
            value=df[feature].median()
        )
        X_live[feature] = st_widget

    with col4:
        feature = 'GarageArea'
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min() * percentage_min,
            max_value=df[feature].max() * percentage_max,
            value=df[feature].median()
        )
        X_live[feature] = st_widget

    return X_live

