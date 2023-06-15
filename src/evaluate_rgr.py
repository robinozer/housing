import streamlit as st
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Code copied from notebook '05-Regression Pipeline'
def regression_performance(X_train, y_train, X_test, y_test, pipeline):
    print("Model Evaluation \n")
    print("* Train Set")
    regression_evaluation(X_train, y_train, pipeline)
    print("* Test Set")
    regression_evaluation(X_test, y_test, pipeline)

# Here is the original code from notebook '05-Regression Pipeline'
# for evaluation. It has been edited below to show a df with evaluation metrics
# on Streamlit.
"""
def regression_evaluation(X, y, pipeline):
    prediction = pipeline.predict(X)
    print('R2 Score:', r2_score(y, prediction).round(3))
    print('Mean Absolute Error:', mean_absolute_error(y, prediction).round(3))
    print('Mean Squared Error:', mean_squared_error(y, prediction).round(3))
    print('Root Mean Squared Error:', np.sqrt(
        mean_squared_error(y, prediction)).round(3))
    print("\n")
"""

def regression_evaluation(X, y, pipeline):
    prediction = pipeline.predict(X)
    r2 = r2_score(y, prediction).round(3)
    mae = mean_absolute_error(y, prediction).round(3)
    mse = mean_squared_error(y, prediction).round(3)
    rmse = np.sqrt(mean_squared_error(y, prediction)).round(3)

    evaluation_df = pd.DataFrame(
        {
            "R2 Score": [r2],
            "Mean Absolute Error": [mae],
            "Mean Squared Error": [mse],
            "Root Mean Squared Error": [rmse],
        }
    )

    return evaluation_df

# Code copied from notebook '05-Regression Pipeline'
def regression_evaluation_plots(X_train, y_train, X_test, y_test, pipeline, alpha_scatter=0.5):
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    sns.scatterplot(x=y_train, y=pred_train, alpha=alpha_scatter, ax=axes[0])
    sns.lineplot(x=y_train, y=y_train, color='red', ax=axes[0])
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predictions")
    axes[0].set_title("Train Set")

    sns.scatterplot(x=y_test, y=pred_test, alpha=alpha_scatter, ax=axes[1])
    sns.lineplot(x=y_test, y=y_test, color='red', ax=axes[1])
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predictions")
    axes[1].set_title("Test Set")

    st.pyplot(fig)
