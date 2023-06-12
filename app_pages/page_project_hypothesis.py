import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    # conclusions taken from "02-EDA" notebook
    st.success(
        f"We suspect that the size of a house would be a significant predictor of its price."
        f"The correlation study in House Prices Study supports this, as 4 out of 6"
        f"of the highest correlated variables were related to size of a house. \n\n"

        f"The House Prices Study showed that the overall quality of a house as well as"
        f"the year it was built were highly correlated with the sale price."
        f"Particularly newer houses built post-2000 demonstrate a strong correlation."
        f"These insights will be used for further investigations."
    )
