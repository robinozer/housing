import streamlit as st
import pandas as pd
import joblib
from src.data_management import load_cleaned_housing_data
from src.converters import ConvertToInt64


def page_predict_house_price_body():

    # Load the pipeline directly using joblib.load
    version = "v2"
    pipeline = joblib.load(f"outputs/ml_pipeline/predict_housing/{version}/best_regressor_pipeline.pkl")
    best_features = pd.read_csv(f"outputs/ml_pipeline/predict_housing/{version}/X_train.csv").columns.to_list()


    # this is the inherited houses (cleaned to have matching data types with the main dataset)
    inherited_df = pd.read_csv("outputs/datasets/cleaned/inherited_houses_cleaned.csv")

    # predict prices of inherited houses with ML pipeline from 05-Regression pipeline notebook
    st.write("### House sale prices from client's inherited houses")
    st.info("* The table below shows the four inherited houses profile")

    st.write(inherited_df.head())
    inherited_df = inherited_df.filter(best_features)
    house_price_prediction = pipeline.predict(inherited_df).round(0)
    inherited_df['Predicted House Sale Price'] = house_price_prediction
    st.write(
        "* The table below shows the predicted sale prices for the "
        "four houses, together with the house features used in the prediction, "
        "which are the four most important variables we saw in the House Price Study page: "
        "'OverallQuality', 'TotalBsmtSF', '2ndFlrSF', and 'GarageArea'."
    )
    st.write(inherited_df.head())

    # calculate sum of inherited houses predicted prices
    sum_prices = inherited_df['Predicted House Sale Price'].sum()
    st.write(
        "* The sum of the predicted sale prices for the four houses is: {sum_prices}\n"
    )

    st.write("---")

    # predict price of any other house in Ames, Iowa
    st.write("### Predict house sale prices in Ames, Iowa  \n")
    st.write("* The following 4 variables 'OverallQuality', 'TotalBsmtSF', "
             "'2ndFlrSF', and 'GarageArea' are needed for the ML model to predict the price.")

    # create input fields for live data
    X_live = DrawInputsWidgets()
    # predict on live data
    if st.button("Run Predictive Analysis"):
        house_price_prediction = pipeline.predict(X_live.filter(best_features)).round(0)
        st.write(
            "* The predicted sale price for the house is: {house_price_prediction[0]}\n"
        )
