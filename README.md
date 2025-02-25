# ⚡🔋 Steel industry energy consumption forecasting
 
This repository contains a machine learning project aimed at forecasting energy consumption in the steel industry. 

The complete project is available in the Jupyter notebook titled **steel-industry-energy-consumption-forecasting.ipynb** in this repository.

## Table of content
 - [Introduction](https://github.com/herrerovir/ML-steel-industry-energy-consumption-forecasting/blob/main/README.md#Introduction)
 - [Goal](https://github.com/herrerovir/ML-steel-industry-energy-consumption-forecasting/blob/main/README.md#Goal)
 - [Dependencies](https://github.com/herrerovir/ML-steel-industry-energy-consumption-forecasting/blob/main/README.md#Dependencies)
 - [Dataset](https://github.com/herrerovir/ML-steel-industry-energy-consumption-forecasting/blob/main/README.md#Dataset)
 - [Project overview](https://github.com/herrerovir/ML-steel-industry-energy-consumption-forecasting/blob/main/README.md#Project-overview)
 - [Data loading](https://github.com/herrerovir/ML-steel-industry-energy-consumption-forecasting/blob/main/README.md#Data-loading)
 - [Data cleaning](https://github.com/herrerovir/ML-steel-industry-energy-consumption-forecasting/blob/main/README.md#Data-cleaning)
 - [Data exploration](https://github.com/herrerovir/ML-steel-industry-energy-consumption-forecasting/blob/main/README.md#Data-exploration)
 - [Linear regression model](https://github.com/herrerovir/ML-steel-industry-energy-consumption-forecasting/blob/main/README.md#Linear-regression-model)
 - [Insights](https://github.com/herrerovir/ML-steel-industry-energy-consumption-forecasting/blob/main/README.md#Insights)
 - [Learning outcomes](https://github.com/herrerovir/ML-steel-industry-energy-consumption-forecasting/blob/main/README.md#Learning-outcomes)

## Introduction

The steel industry is crucial to modern manufacturing but is also a major consumer of energy, leading to high operational costs and environmental impacts. As demand for steel rises, optimizing energy consumption becomes increasingly urgent.

## Goal

This machine learning project focuses on analyzing energy consumption patterns within the steel industry, focusing on data from DAEWOO Steel Co. Ltd in Gwangyang, South Korea, which produces various coils, steel plates, and iron plates. By leveraging historical data and operational parameters, the project aims to identify key factors influencing energy use and develop predictive models to enhance energy efficiency. Ultimately, this initiative seeks to provide actionable insights that promote sustainability and reduce the carbon footprint of steel production.

## Dependencies

The following tools are required to carry out this project:

* Python 3
* Jupyter Notebooks
* Python libraries: 
    - Numpy
    - Pandas
    - Matplotlib.pyplot
    - Seaborn
    - Scikit-learn

## Dataset

The dataset used for this project was sourced from the UC Irvine Machine Learning Repository. It is available in a CSV file uploaded to this repository under the name "steel-industry-data"

The dataset consists of:
* 35040 rows
* 11 columns

## Project overview

* Data loading
* Data cleaning
* Data exploration
* Linear regression model

## Data loading

The CSV dataset is loaded into Jupyter Notebooks as a Pandas DataFrame. 

## Data cleaning

Data cleaning is a essential step in this project. It involves identifying and correcting errors and inconsistencies in a dataset, ensuring high-quality data for analysis. It is crucial because it improves accuracy, maintains consistency, handles missing values, enhances model performance ultimately leading to more reliable insights.

## Data exploration

To extract valuable insights from the dataset, a thorough exploratory analysis was conducted. This analysis included both univariate and bivariate approaches.

Univariate analysis focused on examining each variable in the dataset individually.

Bivariate analysis involved exploring the relationships between pairs of variables to identify any dependencies or correlations.

## Linear regression model

A linear regression model is selected for this machine learning project because the target variable is continuous and the relationships among the features are linear. Here are key reasons for this choice:

* **Simplicity and interpretability:** the model is easy to understand, with coefficients that clearly illustrate relationships between features and the target variable.
* **Assumptions of linearity:** linear regression effectively captures the linear dynamics between features and the target variable, leading to accurate predictions.
* **Efficiency:** it is computationally efficient to train, requiring less processing power, making it suitable for large datasets.
* **Baseline model:** it serves as a solid baseline, helping evaluate the performance of more complex models if needed.
* **Reduced overfitting risk:** its simplicity minimizes the risk of overfitting, particularly with a manageable number of features.

These factors make linear regression an effective choice for this project.

## Insights

The main goal of this project is to identify key factors influencing energy use and develop predictive models to enhance energy efficiency. A linear regression model was built and evaluated using several metrics: the coefficient of determination, mean squared error, root mean squared error, and mean absolute error—all indicating strong model performance. Additionally, the results include visualizations comparing the model's predictions to the actual values.

![Linear Regression Model Visualization](https://github.com/user-attachments/assets/10f6f6fd-cd1d-48a0-877d-a573df4b3afc)

## Learning outcomes:

* **Data preprocessing:** gain skills in cleaning and preparing data, including handling missing values and feature selection.

* **Linear regression application:** understand the principles of linear regression, model fitting, and interpretation of coefficients in the context of energy consumption.

* **Model evaluation:** learn to apply different evaluation metrics to assess forecasting accuracy.

* **Practical experience with tools and libraries:** acquire hands-on experience using libraries like Pandas, NumPy, and Scikit-learn for data manipulation and modeling.

* **Visualization skills:** enhance abilities in data visualization to effectively present findings and communicate relationships between variables and energy consumption.
