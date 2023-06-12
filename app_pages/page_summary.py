import streamlit as st


def page_summary_body():

    st.write("### Quick Project Summary")

    st.write(
        f" This is a ML app for visualizing and predicting\n"
        f" house prices in in Ames, Iowa.\n" 
        f" The main goals of the app are to allow users to explore\n"
        f" the correlation between various house attributes and sale prices,\n"
        f" as well as to provide accurate predictions for house sale prices.\n"
         )

    st.info(
        f"The dataset used in this project contains nearly 1.5 thousand rows "
        f"and represents housing records from Ames, Iowa. It includes"
        f"information on various house attributes such as floor area, basement,"
        f"garage, kitchen and year built."
        f"The dataset covers houses built between 1872 and 2010.")

    # copied from README file - "Business Requirements" section
    st.success(
        f"** The project has 2 business requirements:** \n \n "
        f"**1.** The client is interested in discovering how the house attributes "
        f"correlate with the sale price. Therefore, the client expects data "
        f"visualisations of the correlated variables against the sale price "
        f"to show that. \n\n "
        f"**2.** The client is interested in predicting the house sale price from "
        f"her four inherited houses and any other house in Ames, Iowa. ")

    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://https://github.com/robinozer/housing).")