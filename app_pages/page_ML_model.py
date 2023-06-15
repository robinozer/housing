import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file
from src.evaluate_rgr import regression_performance, regression_evaluation, regression_evaluation_plots


def page_ML_model_body():
    # load house price pipeline files (v3)
    version = 'v3'
    pipeline = load_pkl_file(f"outputs/ml_pipeline/predict_housing/{version}/best_regressor_pipeline.pkl")
    feat_importance = plt.imread(f"outputs//ml_pipeline/predict_housing/{version}/features_importance.png")
    X_train = pd.read_csv(f"outputs//ml_pipeline/predict_housing/{version}/X_train.csv")
    X_test = pd.read_csv(f"outputs//ml_pipeline/predict_housing/{version}/X_test.csv")
    y_train = pd.read_csv(f"outputs//ml_pipeline/predict_housing/{version}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"outputs//ml_pipeline/predict_housing/{version}/y_test.csv").squeeze()

    st.write("### ML pipeline")
    # summary of model performance
    st.info(
        f"* An R2 score of at least 0.8 on both train and test sets was acceptable to the client. \n"
        f"* This pipeline achieves 0.906 on the train set and and 0.832 on the test set, as seen below."    
    )
    st.write("---")

    # show pipeline steps
    st.write("* **This is the whole ML pipeline to predict house sale price**")
    st.write(pipeline)
    st.write("---")

    # show best features and their importance for the ML model
    st.write("* **These are the best features the model was trained on. The plot demonstrates their importance**")
    st.info("Together, these variables predict sale price in our model.")
    st.write(X_train.columns.to_list())
    st.image(feat_importance)
    st.write("---")

    # evaluate performance on train and test sets
    train_evaluation = regression_evaluation(X_train, y_train, pipeline)
    test_evaluation = regression_evaluation(X_test, y_test, pipeline)

    # display evaluation table for train set
    st.write("* **Evaluation Metrics for Train Set**")
    st.dataframe(train_evaluation)

    st.write("---")

    # display evaluation table for test set
    st.write("* **Evaluation Metrics for Test Set**")
    st.dataframe(test_evaluation)

    st.write("---")

    # Plot predicted versus actual sale price for train and test sets
    st.write("* **Predicted versus actual sale price scatterplot**")
    st.write("* As we can see the dots generally match the red line. \n"
            "In the test set we can see dots straying from the line for houses with higher prices."
            " This can be because the model is not as good at accurately predicting the most expensive houses.")
    regression_evaluation_plots(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pipeline=pipeline, alpha_scatter=0.5)
