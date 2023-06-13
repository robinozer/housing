import plotly.express as px
import numpy as np
import streamlit as st
from sklearn.preprocessing import KBinsDiscretizer
from src.data_management import load_cleaned_housing_data

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def page_house_prices_study_body():

    # load data
    df = load_cleaned_housing_data()

    # hard copied from EDA notebook
    vars_to_study = ['1stFlrSF', 'GarageArea', 'GrLivArea', 
                    'OverallQual', 'TotalBsmtSF', 'YearBuilt']


    st.write("### House Prices Study")
    st.info(
        f"* The client is interested in discovering how the house attributes correlate with the sale price."
        f" Therefore, the client expects data visualizations of the correlated variables against the sale price to show that.")

    # inspect data
    if st.checkbox("Inspect Customer Base"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))

    st.write("---")

    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted in the notebook to better understand how "
        f"the variables are correlated to sale prices. \n"
        f"The most correlated variables are: **{vars_to_study}**"
    )

    # Text based on "02-EDA" notebook - "Conclusions and Next steps" section
    st.info(
        f"The correlation indications and plots below interpretation converge. "
        f"It is indicated that: \n"
        f"* Houses with larger first floor areas typically sell for higher price. \n"
        f"* Houses with larger garages typically sell for higher price. \n"
        f"* Houses with a larger above-ground living area typically sell for higher price.\n"
        f"* Houses with overall higher quality typically sell for higher price. \n"
        f"* Houses with a larger basement typically sell for higher price. \n"
        f"* Houses that are newer typically  sell for higher price. \n"
    )

    # Code copied from "02-EDA" notebook - "EDA on selected variables" section
    df_eda = df.filter(vars_to_study + ['SalePrice'])

    # Individual plots per variable
    if st.checkbox("Sale Price per Variable"):
        st.write(
            f"* Here you can see the distribution of each chosen variable against the target Sale Price."
        )
        sale_price_per_variable(df_eda)

    # Parallel plot
    if st.checkbox("Parallel Plot"):
        st.write(
            f"* Since the variables are all numerical (the target Sale Price being continuous numerical), "
            f"the plot has categorized the values as shown below. "
            f"* Hover over the variable categories in order to visualize how the 6 variables are connected."
            f"* The color map indicates price range."
        )
        parallel_plot(df_eda)


# function created using "02-EDA" notebook - "Variables Distribution by Sale Price" section
def sale_price_per_variable(df_eda):
    target_var = 'SalePrice'
    vars_to_study = ['1stFlrSF', 'GarageArea', 'GrLivArea', 'OverallQual', 'TotalBsmtSF', 'YearBuilt']
    for col in vars_to_study:
        if df_eda[col].dtype == 'object':
            plot_categorical(df_eda, col, target_var)
            print("\n\n")
        else:
            plot_numerical(df_eda, col, target_var)
            print("\n\n")


# code copied from "02-EDA" notebook - "Variables Distribution by Sale Price" section
def plot_categorical(rdf, col, target_var):
    df = load_cleaned_housing_data()
    fig = plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x=col, hue=target_var, order=df[col].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)

# code copied from "02-EDA" notebook - "Variables Distribution by Sale Price" section
def plot_numerical(df, col, target_var):
    df = load_cleaned_housing_data()
    fig = plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=col, y=target_var)
    plt.title(f"{col} vs {target_var}", fontsize=20, y=1.05)
    st.pyplot(fig)


# function created using "02-EDA" notebook code - Parallel Plot section
def parallel_plot(df_eda):
    top_features = ['1stFlrSF', 'GarageArea', 'GrLivArea', 'OverallQual', 'TotalBsmtSF', 'YearBuilt']

    df_parallel = df_eda[top_features].copy()
    disc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    for column in df_parallel.columns:
        df_parallel[column] = disc.fit_transform(df_parallel[[column]])

    labels_map = {
        0: '<20%',
        1: '20% to 40%',
        2: '40% to 60%',
        3: '60% to 80%',
        4: '80% to 100%'
    }

    for column in df_parallel.columns:
        df_parallel[column] = df_parallel[column].replace(labels_map)
    df_parallel['SalePrice'] = df_eda['SalePrice']
    fig = px.parallel_categories(df_parallel, color='SalePrice', color_continuous_scale='edge', width=750, height=500)
    st.plotly_chart(fig)
    