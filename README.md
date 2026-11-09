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

Dataset files and trained model artifacts are stored separately in Google Drive.

## Development Roadmap

| Day | Development Task | Status |
|---|---|---|
| Day 1 | Dataset Preparation and EDA | Completed |
| Day 2 | ANN Regression and Evaluation | Completed |
| Day 3 | ANN Classification and Evaluation | Completed |
| Day 4 | Application Development and Model Integration | Upcoming |
| Day 5 | Application Features and Prediction Workflow | Upcoming |
| Day 6 | Testing and Performance Improvements | Upcoming |
| Day 7 | Final Documentation and Project Completion | Upcoming |


## Current Progress

### Day 1 – Dataset Preparation and EDA

- Dataset inspection and understanding.
- Data cleaning and preprocessing.
- Dataset merging.
- Exploratory Data Analysis (EDA).
- Feature engineering.
- Prepared dataset export.

**Status:** Completed

### Day 2 – ANN Regression Using PyTorch

- Performed feature preprocessing using StandardScaler and OneHotEncoder.
- Built an Artificial Neural Network (ANN) regression model using PyTorch.
- Trained the model using Adam optimizer and L1Loss.
- Implemented early stopping.
- Evaluated performance using MAE, MSE, RMSE and R².
- Created actual vs. predicted and error distribution visualizations.
- Performed category-wise MAE and percentage MAE analysis.

**Best Model Configuration:**

| Parameter | Value |
|---|---|
| Architecture | 128 → 64 → 1 |
| Loss Function | L1Loss |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Validation MAE | 78.87 |
| Validation R² | 0.7996 |

**Status:** Completed

### Day 3 – ANN Classification Using PyTorch

- Reused the Day 2 preprocessing pipeline.
- Created three demand categories: Low, Medium and High.
- Generated chronological training and validation datasets.
- Built an ANN classification model using PyTorch.
- Used CrossEntropyLoss and Adam optimizer.
- Implemented early stopping.
- Evaluated performance using accuracy, precision, recall and F1-score.
- Generated classification report and confusion matrix.
- Predicted demand categories for 32,573 test records.

**Model Configuration:**

| Parameter | Value |
|---|---|
| Architecture | 217 → 128 → 64 → 3 |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Best Validation Loss | 0.5243 |

**Classification Evaluation:**

| Metric | Score |
|---|---:|
| Accuracy | 76.73% |
| Precision (Macro) | 77.78% |
| Recall (Macro) | 76.80% |
| F1-Score (Macro) | 77.13% |

**Test Predictions:**

| Demand Category | Predictions |
|---|---:|
| Low | 9,953 |
| Medium | 13,524 |
| High | 9,096 |
| **Total** | **32,573** |

**Saved Artifacts:**
- `ann_classifier.pth`
- `demand_thresholds.pkl`
- Reused `preprocessor.pkl` from Day 2.

**Status:** Completed


## Author

Computer Science Engineering Student

