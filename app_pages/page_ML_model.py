import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file
from src.evaluate_rgr import regression_performance, regression_evaluation, regression_evaluation_plots


def page_ML_model_body():

    # load house price pipeline files
    version = 'v2'
    pipeline = load_pkl_file(f"outputs/ml_pipeline/predict_housing/{version}/best_regressor_pipeline.pkl")
    feat_importance = plt.imread(f"outputs//ml_pipeline/predict_housing/{version}/features_importance.png")
    X_train = pd.read_csv(f"outputs//ml_pipeline/predict_housing/{version}/X_train.csv")
    X_test = pd.read_csv(f"outputs//ml_pipeline/predict_housing/{version}/X_test.csv")
    y_train =  pd.read_csv(f"outputs//ml_pipeline/predict_housing/{version}/y_train.csv").squeeze()
    y_test =  pd.read_csv(f"outputs//ml_pipeline/predict_housing/{version}/y_test.csv").squeeze()

    st.write("### ML pipeline")

    # summary of model performance
    st.info(
        f"* An R2 score of at least 0.8 on both train and test sets was acceptable to the client."
        f"* This pipeline achieves 0.906 on the train set and and 0.80 on the train set and test set respectively  \n"    
    )
    st.write("---")

    # show pipeline steps
    st.write("* **ML pipeline to predict house sale price**")
    st.write(pipeline)
    st.write("---")

    # show best features and their importance for the ML model
    st.write("* **The features the model was trained on and their importance**")
    st.info("We see that the most important variable for predicting the sale price is 'OverallQual'")
    st.write(X_train.columns.to_list())
    st.image(feat_importance)
    st.write("---")

    # evaluate performance on train and test sets
    regression_performance(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pipeline=pipeline)
    
    st.write("---")
     
    # Plot predicted versus actual sale price for train and test sets
    st.write("* **Predicted versus actual sale price scatterplot**")
    regression_evaluation_plots(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pipeline=pipeline, alpha_scatter=0.5)