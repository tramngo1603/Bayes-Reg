# Prediction of California Housing Prices using Bayesian Regression

This project aims to develop and compare various regression models for predicting housing prices based on the California Housing dataset. The notebook explores different techniques for data preprocessing, feature engineering, model development, and evaluation.

## Dataset

The dataset used in this project is the California Housing dataset from the Scikit-learn library. It contains information about housing prices and various features such as median income, housing age, number of rooms, and geographic coordinates.

## Notebook Overview

1. **Data Loading & Processing**
   - Load the California Housing dataset
   - Check for missing values and handle them using imputation
   - Perform data exploration and statistical analysis

2. **Feature Engineering**
   - Check for multicollinearity using the Variance Inflation Factor (VIF)
   - Perform feature selection using correlation analysis, statistical tests (`SelectKBest`), and Recursive Feature Elimination (RFE)
   - Create new relevant features using polynomial transformations

3. **Model Development**
   - Select the best traditional model (Linear Regression, Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, Gradient Boosting) using GridSearchCV
   - Develop a Bayesian Ridge model using Scikit-learn's BayesianRidge
   - Implement a custom Bayesian Regression model from scratch

4. **Model Training & Evaluation**
   - Split the data into training and testing sets
   - Train the selected models on the training set
   - Evaluate the models using mean squared error (MSE) and R-squared (R²) metrics on the test set

## Dependencies

The following Python libraries are required to run the notebook:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- StatsModels
- SciPy
