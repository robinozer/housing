import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    # conclusions from "02-EDA" notebook
    st.success(
        f"* We suspect that the area of a house (the size in SF) would be a significant predictor of its price."
        f" The correlation study in House Prices Study supports this, as 4 out of 6"
        f" of the variables with highest correlation to the target were related to size of a house. \n\n"
        f"* The House Prices Study further showed that the overall quality of a house as well as "
        f"the year it was built were highly correlated with the sale price. "
        f"Particularly newer houses built post-2000 demonstrate a strong correlation. "
        f"These insights will be used for further investigations."
    )