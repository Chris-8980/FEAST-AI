# FEAST AI
### Food Estimation and Analysis using Smart Technology

**An ANN-Based Food Demand Forecasting, Classification and Surplus Estimation System.**

## About the Project

FEAST AI is a machine learning and deep learning project focused on food demand forecasting, demand classification and surplus estimation using historical food order data.

The primary objective is to help food businesses make better food preparation decisions, reduce overproduction and minimize food wastage.

The project uses two Artificial Neural Network (ANN) models:

- **ANN Regression:** Predicts the expected number of food orders.
- **ANN Classification:** Categorizes food demand into Low, Medium and High.

In addition, the system estimates potential food surplus and provides recommendations to support better surplus management.

## Problem Statement

Food businesses often struggle to estimate customer demand accurately, resulting in excess food preparation or insufficient availability.

FEAST AI aims to address this problem through data-driven demand forecasting, classification and surplus estimation, helping food businesses make informed food preparation and surplus management decisions.

## Project Objectives

1. Forecast the expected number of food orders.
2. Classify food demand into Low, Medium and High categories.
3. Estimate potential surplus based on prepared quantity and predicted demand.
4. Provide recommendations for managing potential food surplus.
5. Reduce unnecessary food production and food wastage.

## Key Features

- **Food Demand Forecasting:** Predicts expected food orders using ANN Regression.
- **Demand Classification:** Categorizes food demand into Low, Medium and High using ANN Classification.
- **Surplus Estimation:** Estimates potential surplus by comparing prepared quantity with predicted orders.
- **Food Rescue Recommendations:** Suggests actions based on the estimated surplus.
- **Interactive Streamlit Application:** Allows users to enter food-related information and view predictions and recommendations.

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

Dataset setup instructions are available in the `food_dataset/` directory.

The prepared training dataset (`train_prepared.csv`) is excluded from GitHub because of its large file size.

Before running the Streamlit application, ensure that `train_prepared.csv` is placed inside the root-level `food_dataset/` directory.

## Development Roadmap

| Day | Development Task | Status |
|---|---|---|
| Day 1 | Dataset Preparation and EDA | Completed |
| Day 2 | ANN-Based Food Demand Regression Using PyTorch | Completed |
| Day 3 | ANN-Based Food Demand Classification Using PyTorch | Completed |
| Day 4 | Model Integration and Test Predictions | Completed |
| Day 5 | ANN Classification Evaluation and Performance Analysis | Completed |
| Day 6 | FEAST AI Streamlit Application Development | Completed |
| Day 7 | Final Testing, Documentation and GitHub Integration | In Progress |


## Current Progress

### Day 1 – Dataset Preparation and EDA

- Dataset inspection and understanding.
- Data cleaning and preprocessing.
- Dataset merging.
- Exploratory Data Analysis (EDA).
- Feature engineering.
- Prepared dataset export.

**Status:** Completed

### Day 2 – ANN-Based Food Demand Regression Using PyTorch
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

### Day 3 – ANN-Based Food Demand Classification Using PyTorch

- Reused the Day 2 preprocessing pipeline.
- Created three demand categories: Low, Medium and High.
- Generated chronological training and validation datasets.
- Built an ANN classification model using PyTorch.
- Used CrossEntropyLoss and Adam optimizer.
- Implemented early stopping.
- Evaluated performance using accuracy, precision, recall and F1-score.
- Generated a classification report and confusion matrix.
- Predicted demand categories for 32,573 test records.

**Model Configuration:**

| Parameter | Value |
|---|---|
| Architecture | 217 → 128 → 64 → 3 |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Best Validation Loss | 0.5243 |

**Validation Classification Performance:**

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

### Day 4 – Model Integration and Test Predictions

- Loaded the trained ANN regression and classification models.
- Loaded the saved preprocessing pipeline and demand thresholds.
- Reconstructed both ANN architectures using PyTorch.
- Preprocessed the test dataset using the saved preprocessor.
- Generated food order quantity predictions using ANN Regression.
- Classified food demand into Low, Medium and High categories using ANN Classification.
- Reversed the logarithmic transformation to obtain predicted order quantities.
- Validated predictions by checking missing values, infinite values, regression statistics and demand distribution.
- Inspected the top 10 predicted food order quantities.
- Exported the combined predictions into a CSV file.

**Test Prediction Summary:**

| Parameter | Value |
|---|---:|
| Total Test Records | 32,573 |
| Low Demand Predictions | 9,953 |
| Medium Demand Predictions | 13,524 |
| High Demand Predictions | 9,096 |

**Saved Output:**
- `smartfood_predictions.csv`

**Status:** Completed

### Day 5 – ANN Classification Evaluation and Performance Analysis

- Reloaded the saved ANN classification model.
- Recreated the chronological training and validation split.
- Applied the existing preprocessing pipeline to validation data.
- Generated predictions on 95,200 validation records.
- Evaluated model performance using accuracy, precision, recall and F1-score.
- Generated a detailed classification report.
- Created original and normalized confusion matrices.
- Saved evaluation metrics, classification reports and confusion matrix visualizations.

**Validation Classification Performance:**
| Metric | Score |
|---|---:|
| Accuracy | 76.73% |
| Precision (Macro) | 77.78% |
| Recall (Macro) | 76.80% |
| F1-Score (Macro) | 77.13% |

**Saved Evaluation Artifacts:**
- `classification_metrics.csv`
- `classification_report.csv`
- `confusion_matrix.csv`
- `normalized_confusion_matrix.csv`
- `confusion_matrix.png`
- `normalized_confusion_matrix.png`

**Status:** Completed


### Day 6 – FEAST AI Streamlit Application Development

- Developed an interactive Streamlit application.
- Integrated the trained ANN regression and classification models.
- Loaded the saved preprocessing pipeline and demand thresholds.
- Implemented user input fields for food centre, meal, pricing, promotion and other relevant features.
- Generated food demand predictions.
- Classified demand into Low, Medium and High.
- Implemented surplus estimation using prepared quantity and predicted orders.
- Added food rescue recommendations based on estimated surplus.
- Designed the application interface using Streamlit.

**Status:** Completed


## Installation and Execution

### 1. Clone the Repository

git clone https://github.com/Chris-8980/FEAST-AI.git

### 2. Navigate to the Project Directory

cd FEAST-AI

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Prepare the Dataset

The prepared training dataset is not included in this repository because it exceeds GitHub's file upload limit.

Place the required `train_prepared.csv` file inside:

`food_dataset/train_prepared.csv`

Refer to `food_dataset/README.md` for dataset setup instructions.

### 5. Run the Application

python -m streamlit run SmartFood_Rescue/app.py

## Author

Computer Science Engineering Student

