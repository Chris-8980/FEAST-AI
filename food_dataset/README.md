# FEAST AI – Dataset

This folder contains the dataset required by the FEAST AI application.

## Required File

- `train_prepared.csv`

## Dataset Setup

The prepared training dataset is not included in this repository because of its file size.

To run the application:

1. Obtain the Genpact Food Demand Forecasting dataset.
2. Run the relevant data preparation steps from the project notebooks.
3. Generate `train_prepared.csv`.
4. Place the generated file inside this folder.

## Expected Structure

```text
food_dataset/
└── train_prepared.csv
```

The Streamlit application uses this file to populate its input options.
