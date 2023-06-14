import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_house_prices_study import page_house_prices_study_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_predict_house_price import page_predict_house_price_body
from app_pages.page_ML_model import page_ML_model_body

app = MultiPage(app_name= "House Price Prediction")

# Add app pages using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("House Prices Study", page_house_prices_study_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("House Prices Predictor", page_predict_house_price_body)
app.add_page("ML Regressor Model", page_ML_model_body)

app.run() # Run the app