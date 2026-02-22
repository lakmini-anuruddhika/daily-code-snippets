# Store Sales Time Series Forecasting

Solution for the [Kaggle Store Sales Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview) (Getting Started) competition.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place competition data in the `data/` folder:

- `train.csv`, `test.csv`
- `stores.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`
- `sample_submission.csv` (optional)

## Run

From this directory:

```bash
python train_and_submit.py
```

This will:

1. Load and merge all datasets
2. Add date, holiday, oil, and store features
3. Train a LightGBM model (target: `log1p(sales)` for RMSLE)
4. Optionally print validation RMSLE (last 2 weeks of train)
5. Write **submission.csv** in the required format  
6. Save **report figures** in `report_figures/` (for lab reports; use `--no-figures` to skip)

### Report figures (Lab 02)

By default, the script saves visualizations in the **`report_figures/`** folder:

| File | Description |
|------|-------------|
| `01_sales_over_time.png` | Daily total sales (last 12 months) |
| `02_sales_distribution.png` | Distribution of sales (log scale) |
| `03_sales_by_family.png` | Top 15 product families by total sales |
| `04_oil_price_over_time.png` | Oil price time series |
| `05_holidays_over_time.png` | National holidays (markers) |
| `06_actual_vs_predicted.png` | Validation: actual vs predicted sales |
| `07_feature_importance.png` | Top 20 LightGBM feature importances |
| `08_residuals_distribution.png` | Validation residuals (log scale) |
| `09_validation_metrics.png` | RMSLE, MAE, RMSE on validation set |

Use these in your report (data cleaning, evaluation metrics, solution description). To skip generating figures: `python train_and_submit.py --no-figures`.

### Fine-tuning hyperparameters

**Option 1 – Optuna (recommended)**  
Install Optuna, then run tuning before training:

```bash
pip install optuna
python train_and_submit.py --tune --n-trials 50
```

- `--n-trials`: number of trials (default 30).
- `--tune-timeout`: max seconds for tuning (default 3600).
- Validation uses the last 2 weeks of train; best params are then used to train on the full dataset.

**Option 2 – Random search (no Optuna)**  
Uses scikit-learn’s `RandomizedSearchCV`:

```bash
python train_and_submit.py --tune --use-random-search --n-trials 20
```

**Other options**

- `--val-days 14`: use last N days of train as validation (default 14).

## Submission

- Upload `submission.csv` to Kaggle via **Submit to Competition** (from a notebook output or local upload).
- Format: `id,sales` with one prediction per test row.

## Evaluation

Metric: **Root Mean Squared Logarithmic Error (RMSLE)**

\[
\text{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\big(\log(p_i+1) - \log(a_i+1)\big)^2}
\]

The script trains on \(\log(1 + \text{sales})\), then converts predictions back with \(\exp(\text{pred}) - 1\) and clips to non-negative.
