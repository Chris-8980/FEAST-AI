# SmartFood Rescue

### Food Demand Forecasting and Classification using Artificial Neural Networks (ANN)

## About the Project

SmartFood Rescue is a machine learning and deep learning project focused on predicting food demand using historical food order data.

The primary objective is to help food businesses make better food preparation decisions, reduce overproduction and minimize food wastage.

The project uses two Artificial Neural Network (ANN) models:

- **ANN Regression:** Predicts the expected number of food orders.
- **ANN Classification:** Categorizes food demand into Low, Medium and High.

## Problem Statement

Food businesses often struggle to estimate customer demand accurately, resulting in excess food preparation or insufficient availability.

SmartFood Rescue aims to address this problem through data-driven demand forecasting and classification.

## Project Objectives

1. Forecast the expected number of food orders.
2. Classify food demand into Low, Medium and High categories.
3. Help businesses make informed food preparation decisions.
4. Reduce unnecessary food production and food wastage.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PyTorch
- Google Colab
- Streamlit

## Dataset

**Dataset:** Genpact Food Demand Forecasting

The dataset contains historical food demand information, including pricing, promotions, meal categories, food centres and order quantities.

Dataset files are maintained separately in Google Drive.

## Development Roadmap

| Day | Development Task | Status |
|---|---|---|
| Day 1 | Dataset Preparation and EDA | Completed |
| Day 2 | ANN Regression | Pending |
| Day 3 | Regression Evaluation | Pending |
| Day 4 | ANN Classification | Pending |
| Day 5 | Classification Evaluation | Pending |
| Day 6 | Application Development | Pending |
| Day 7 | Testing and Documentation | Pending |


## Current Progress

**Day 1 – Completed**

- Dataset inspection
- Data cleaning
- Dataset merging
- Exploratory Data Analysis
- Feature engineering
- Prepared dataset export

 Day 2: ANN Regression Using PyTorch — Completed

Performed feature preprocessing using StandardScaler and OneHotEncoder.

Built an Artificial Neural Network (ANN) regression model using PyTorch.

Trained the model using Adam optimizer and L1Loss.

Implemented early stopping.

Evaluated model performance using MAE, MSE, RMSE, and R².

Created actual vs predicted and error distribution visualizations.

Performed category-wise MAE and percentage MAE analysis.

Best Model Configuration:

Architecture: 128 → 64 → 1

Loss Function: L1Loss

Optimizer: Adam

Learning Rate: 0.001

Validation MAE: 78.87

Validation R²: 0.7996

Status: Completed

## Author

Computer Science Engineering Student

