## Store Sales - Time Series Forecasting

This repository contains a machine learning solution for the Kaggle
**Store Sales - Time Series Forecasting**(https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition. The project uses
**Gradient Boosted Decision Trees (LightGBM)** to forecast grocery sales
for Corporación Favorita stores in Ecuador.

------------------------------------------------------------------------

## Project Overview

The objective of this project is to predict the **unit sales for
thousands of items** sold across multiple Favorita stores. Accurate
sales forecasting helps:

-   Reduce food waste caused by overstocking\
-   Prevent lost revenue due to stockouts\
-   Improve inventory planning

------------------------------------------------------------------------

## Key Engineering Features

### Time-Series Features

-   Lag features to capture previous sales behavior\
-   Rolling mean features (7-day and 30-day windows) to detect trends
    and seasonality

### Economic Indicators

-   Incorporates daily oil prices, which strongly influence Ecuador's
    economy

### Calendar Events

-   National and local holidays\
-   Bi-monthly payday effect (15th and end of the month)

### Advanced Modeling

-   Optuna for automated hyperparameter optimization\
-   Model ensembling for more stable predictions

### Automated Reporting

The pipeline generates 9 diagnostic plots to analyze: - Model
performance\
- Prediction accuracy\
- Feature importance

------------------------------------------------------------------------

## Project Structure

    .
    ├── data/                   # Dataset files from Kaggle
    ├── report_figures/         # Generated diagnostic plots
    ├── main.py                 # Main training and prediction script
    ├── submission.csv          # Final Kaggle submission file
    └── README.md               # Project documentation

------------------------------------------------------------------------

## Getting Started

### 1. Prerequisites

Make sure **Poetry** is installed. Then install the project
dependencies:

``` bash
poetry install
```

------------------------------------------------------------------------

### 2. Data Setup

Download the dataset from:

https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

Place the following files inside the `data/` directory:

-   `train.csv`\
-   `test.csv`\
-   `stores.csv`\
-   `oil.csv`\
-   `holidays_events.csv`\
-   `transactions.csv`

------------------------------------------------------------------------

### 3. Running the Pipeline

Run the training and prediction pipeline inside the Poetry environment:

``` bash
poetry run python main.py
```

------------------------------------------------------------------------

## 🔧 Advanced Usage

### Hyperparameter Tuning (Optuna)

Run automated hyperparameter optimization:

``` bash
python main.py --tune --n-trials 50
```

------------------------------------------------------------------------

### Model Ensembling

Train multiple models with different random seeds and average the
predictions:

``` bash
python main.py --ensemble 3
```

This can improve prediction stability and overall model performance.

------------------------------------------------------------------------

## Evaluation Metric

The model is evaluated using **Root Mean Squared Logarithmic Error
(RMSLE)**:

RMSLE = sqrt( (1/n) \* Σ (log(1 + y_pred) - log(1 + y_true))² )

The script automatically calculates this metric using a **21-day
validation hold-out set** before training on the full dataset.

------------------------------------------------------------------------

## Visualizing Results

After running the script, check the `report_figures/` folder for
generated plots.

Examples include:

-   **01_sales_over_time.png** -- Historical sales trends\
-   **06_actual_vs_predicted.png** -- Scatter plot comparing actual vs
    predicted sales\
-   **07_feature_importance.png** -- Most important features influencing
    the model

------------------------------------------------------------------------

## Built With

-   LightGBM -- Gradient Boosting Framework\
-   Optuna -- Hyperparameter Optimization\
-   Pandas -- Data Manipulation\
-   Matplotlib -- Data Visualization