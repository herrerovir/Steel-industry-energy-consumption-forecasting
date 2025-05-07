# ⚡🔋 Steel Industry Energy Consumption Forecasting

This repository contains a machine learning project that forecasts energy consumption in the steel industry using linear regression.

## Introduction

The steel industry is crucial to modern manufacturing but is also a major consumer of energy, leading to high operational costs and environmental impacts. As demand for steel rises, optimizing energy consumption becomes increasingly urgent.

## 🎯 Goal

The objective is to develop a predictive model that accurately forecasts energy consumption in a steel manufacturing setting. By analyzing historical data and training a machine learning model, the project aims to provide actionable insights that promote sustainability and reduce the carbon footprint of steel production.

## 🔄 Project Overview

This project is structure in four main phases:

- Loading and cleaning a real-world dataset from the steel industry  
- Exploring and analyzing relationships between energy metrics  
- Building a linear regression model to forecast consumption  
- Evaluating the model performance

## 🧰 Dependencies

The libraries used:

- `pandas` – Data manipulation  
- `numpy` – Numerical computation  
- `matplotlib` and `seaborn` – Data visualization  
- `scikit-learn` – Machine learning

## 💻 How to Run the Project

1. **Clone the Repository**

   Start by cloning the repository to your local machine using the following command:

   ```shell
   git clone https://github.com/herrerovir/Steel-industry-energy-consumption-forecasting.git
   ```

   Change to the project directory:

   ```shell
   cd Steel-industry-energy-consumption-forecasting
   ```

2. **Install Dependencies**

   Install the required dependencies listed in the `requirements.txt`:

   ```shell
   pip install -r requirements.txt
   ```

   This will install all necessary libraries such as pandas, numpy, matplotlib, and seaborn.

3. **Run the Jupyter Notebook**

   After installing the dependencies, you can run the Jupyter notebook to perform the data analysis. To start the notebook, use the following command:

   ```shell
   jupyter notebook notebooks/Steel-industry-energy-consumption-forecasting.ipynb
   ```

## 📂 Repository Structure

```
Steel-industry-energy-consumption-forecasting/
│
├── data/
│   └── raw/
│       └── steel_industry_data.csv                             # Original dataset
│   └── processed/
│       └── steel_industry_cleaned_data.csv                     # Clean and processed dataset
│
├── model/
│   └── linear-regression-model.pkl                             # Trained model
│
├── notebooks/
│   └── Steel-industry-energy-consumption-forecasting.ipynb     # Jupyter Notebook with the full analysis
│
├── results/
│   └── figures/
│       └── Correlation-heatmap.png                             # Visualizations
│       └── Linear-regression-model-actual-vs-predicted         # Visualizations
│   └── model-results/
│       └── model-results.txt                                   # Results from the model as txt file
│
├── requirements.txt                                            # Requirements file
│
└── README.md                                                   # Project overview and documentation
```

## 🧠 Technical Skills Demonstrated

- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Model development and evaluation  
- Regression techniques in `scikit-learn`  
- Visualization and interpretation of model results

## 📊 Dataset

The dataset used for this analysis is a CSV file (`steel-industry-data.csv`) containing key features:

- Energy consumption  
- CO₂ emissions  
- Reactive power  
- Power factor  
- Load type

## 🧪 Steps

1. **Data Loading** – Read the CSV into a DataFrame  
2. **Data Cleaning** – Handle missing data and outliers  
3. **EDA** – Visualize and understand feature relationships  
4. **Model Training** – Use linear regression to predict energy use  
5. **Evaluation** – Use MAE, RMSE, and R² for model assessment

## 📉 Linear Regression Model

A linear regression model is selected for this machine learning project because the target variable is continuous and the relationships among the features are linear. Here are key reasons for this choice:

- **Simplicity and interpretability:** the model is easy to understand, with coefficients that clearly illustrate relationships between features and the target variable.
- **Assumptions of linearity:** linear regression effectively captures the linear dynamics between features and the target variable, leading to accurate predictions.
- **Efficiency:** it is computationally efficient to train, requiring less processing power, making it suitable for large datasets.
- **Baseline model:** it serves as a solid baseline, helping evaluate the performance of more complex models if needed.
- **Reduced overfitting risk:** its simplicity minimizes the risk of overfitting, particularly with a manageable number of features.

These factors make linear regression an effective choice for this project.

## 💡 Insights

The main goal of this project is to identify key factors influencing energy use and develop predictive models to enhance energy efficiency. A linear regression model was built and evaluated using several metrics: the coefficient of determination, mean squared error, root mean squared error, and mean absolute error—all indicating strong model performance. Additionally, the results include visualizations comparing the model's predictions to the actual values.

## 📚 Learning Outcomes

- Hands-on experience in the machine learning workflow  
- Improved understanding of industrial energy metrics  
- Enhanced skills in data visualization and modeling  
- Learned how to interpret and communicate regression results 
