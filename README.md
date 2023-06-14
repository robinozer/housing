# House Price Predictor
This is a ML app for visualizing and predicting house prices in in Ames, Iowa. The main goals of the app are to allow users to explore the correlation between various house attributes and sale prices, as well as to provide accurate predictions for house sale prices based on a selected few variables.

## PLACE A LINK TO THE APP HERE

## Dataset Content
* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace. 
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|


## Business Requirements
As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.


## Hypothesis and validation
*  We suspect that the area of a house (the size in SF) would be correlated with its sale price. In order to validate the project hypothesis, we explored the data and conducted a correlation analysis.

Given the output from the correlation analyses (found in notebook 02-EDA), we might describe the correlations as follows:

For Pearson's Correlation (in descending order):

- **OverallQual**: The correlation of 0.79 with SalePrice suggests a very strong positive linear relationship.
- **GrLivArea, GarageArea, TotalBsmtSF, 1stFlrSF**: These features show strong positive linear relationships with **SalePrice**, with correlation coefficients ranging from around 0.6 to 0.71.
- **YearBuilt, YearRemodAdd, MasVnrArea, GarageYrBlt**: With correlation coefficients ranging from approximately 0.47 to 0.52, these variables indicate moderate positive linear relationships with **SalePrice**.
- **BsmtFinSF1**: This shows a weak positive linear relationship with **SalePrice**, with a correlation coefficient of about 0.39.

For Spearman's Correlation:

- **OverallQual**: The correlation of 0.81 with SalePrice suggests a very strong positive monotonic relationship.
- **GrLivArea, YearBuilt, GarageArea, TotalBsmtSF, 1stFlrSF, YearRemodAdd, GarageYrBlt**: These features have strong positive monotonic relationships with **SalePrice**, with correlation coefficients ranging from around 0.57 to 0.73.
- **OpenPorchSF, LotArea**: With correlation coefficients of approximately 0.46 and 0.48, these variables indicate moderate positive monotonic relationships with SalePrice.

Fort both methods:
- **OverallQual** is the variable that has the highest positive correlation with **SalePrice** in both methods. This suggests that as the overall quality of a house increases, so does its sale price.

- **GrLivArea** is the second most correlated variable according to both methods. This indicates that the above ground living area in square feet is a significant predictor of the sale price, with larger living areas commanding higher prices.

- Variables such as **GarageArea, TotalBsmtSF, 1stFlrSF** are also highly positively correlated with **SalePrice**. This means that these features of the house also significantly contribute to its price.

- **YearBuilt** is more correlated when using Spearman's method compared to Pearson's. This might suggest a non-linear relationship between **YearBuilt** and **SalePrice**. It's reasonable to expect that newer houses would sell for more.

- **OpenPorchSF and LotArea** appear in the top 10 for Spearman correlation but not Pearson, suggesting potential non-linear relationships with **SalePrice**.


In conclusion, after OverallQual, the four variables with highest correlation to sale price are **GrLivArea, GarageArea, TotalBsmtSF, 1stFlrSF**. They all measure the area of a house in square feet, thereby validating our hypothesis.


## The rationale to map the business requirements to the Data Visualisations and ML tasks
### Business Requirement 1: Correlation Study and Data Visualization
As a client, I want to gain insights into the factors influencing the sale price of houses. To fulfill this requirement, the following user stories have been addressed:
- As a client, I want to visually explore the house records data to identify important variables that impact the sale price.
- As a client, I want to read an analysis to understand the strength of the relationships between variables and the sale price.
- As a client, I want to visualize the correlation between key variables and the sale price through interactive plots, enabling a better understanding of their impact.

### Business Requirement 2: Predict House Prices in Ames, Iowa
As a client, I want to accurately predict house prices in Ames, Iowa. To fulfill this requirement, the following user stories have been addressed:
- As a client, I want to access and analyze the records of inherited houses to gather information about house attributes.
- As a client, I want to use a machine learning model to predict the prices of my four inherited houses in Ames, Iowa, leveraging key variables for accurate predictions.
- As a client, I want to utilize the same machine learning model to predict the price of any other house in Ames, Iowa, by providing relevant variables and obtaining an estimated sale price.

## ML Business Case
### To address the client's requirements and achieve the desired outcomes, the ML business case is structured as follows:
  1. Business Requirements:
   - The client aims to understand the correlation between house attributes and sale prices. This involves visualizing the correlated variables in relation to the sale price.
    - The client wants to predict the sale prices for their four inherited houses as well as for any other house in Ames, Iowa.
  2. Is there a business requirement that can be answered with conventional data analysis?
    - Conventional data analysis is used to explore the correlation between house attributes and sale prices.
  3. Dashboard or API Endpoint:
    - The client specifically requires a dashboard to visualize the insights and predictions.
  4. Successful Project Outcome for the Client:
    - The client considers the project successful if it provides a comprehensive study showcasing the most relevant variables correlated with sale prices.
    - Additionally, the ability to accurately predict the sale prices for the client's four inherited houses and other houses in Ames, Iowa is crucial.
  5. Epics and User Stories:
    - Data collection and cleaning.
    - Data visualization and preparation.
    - Model training, optimization and validation.
    - Dashboard planning, designing, and development.
    - Dashboard deployment and release.
  6. Ethical or Privacy Concerns:
    - No ethical or privacy concerns are identified since the client is utilizing a publicly available dataset.
  7. Suggested Model:
    - Based on the data, a regression model is suggested, where sale price is the target variable.
  8. Model Inputs and Intended Outputs:
    - The model will take house attribute information as inputs (specifically our 4 best variables) and provide the predicted sale price as the output.
  9. Performance Goal Criteria:
    - The agreed-upon performance goal for the predictions is an R2 score of at least 0.8 on both the train and test sets.
  10. Client Benefits:
    - The client will benefit by maximizing the sales price for their inherited properties through accurate predictions and a comprehensive understanding of the key variables influencing house prices.

## Dashboard Design
- **Quick Project Summary Page**:
    - Provides a brief overview of the project, including a summary of the dataset.
    - Includes a link to this readme file for reference.
    - States the business requirements for the project. 
- **House Prices Study Page**:
    - This page fulfills the first project requirement, which is highlighted in an information box.
    - The middle section summarizes the correlation study and states its conclusions.
    - The page includes three checkboxes that implement the user stories related to the first project requirement:
        - The first displays a table showing a sample of the dataset.
        - The second shows scatterplots of correlated variables against sale price.
        - The third is a parallell plot to visualize how the 6 variables are connected, color coded in sale price.
- **Project Hypothesis Page**:
    - States the project hypothesis and presents its validation.
    - Explains additional insights that can be investigated further.
- **House Prices Predictor Page**:
    - This page addresses the second project requirement.
    - Includes four input widgets and a button that enables users to predict the sale price based on the provided inputs.
- **ML Model Page**:
    - Begins with a general conclusion about the performance of the ML model.
    - Presents an overview of the pipeline steps used in the model.
    - Includes a bar plot showing the importance of each feature in the training set.
    - Evaluates the ML model by computing the R2 score and three different error measures.

## Unfixed Bugs
* No bugs were left unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Set the runtime.txt Python version to python-3.8.12.
2. Create a Heroku Procfile with instructions on runtime.
3. Use Streamlit configuration settings to create a setup.sh file.
4. Create requirements.txt with the libraries and versions used in the project.
5. Log in to Heroku and create an App
6. At the Deploy tab, select GitHub as the deployment method.
7. Select your repository name and click Search. Once it is found, click Connect.
8. Select the branch you want to deploy, then click Deploy Branch.
9. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.

## Main Data Analysis and Machine Learning Libraries
- numpy==1.18.5
- pandas==1.4.2
- matplotlib==3.3.1
- seaborn==0.11.0
- pandas-profiling==3.1.0
- plotly==4.12.0
- ppscore==1.2.0
- streamlit==0.85.0
- feature-engine==1.0.2
- imbalanced-learn==0.8.0
- scikit-learn==0.24.2
- xgboost==1.2.1
- yellowbrick==1.3
- Jinja2==3.1.1
- MarkupSafe==2.0.1
- protobuf==3.20
- ipywidgets==8.0.2
- lightgbm==3.3.5

## Credits 

* A number of code cells were either borrowed or adapted from the Walkthrough project 'Churnometer' by Code Institute.
