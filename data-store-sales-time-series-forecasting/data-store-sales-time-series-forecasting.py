"""
Store Sales Time Series Forecasting - Kaggle Getting Started
Train a model and generate submission.csv (RMSLE evaluation).
Fine-tune with: python train_and_submit.py --tune [--n-trials 50]

================================================================================
TUNING GUIDE – What to change to reduce RMSLE
================================================================================

1. MODEL HYPERPARAMETERS (lgb_params below, or use --tune)
   - n_estimators:    more trees → often better (try 600–1200); balance with time
   - learning_rate:   lower (e.g. 0.02–0.05) + more trees usually helps
   - max_depth:       lower = less overfit (try 6–10)
   - num_leaves:      main complexity knob (try 32–128)
   - min_child_samples: higher = less overfit (try 30–80)
   - subsample:       row sampling (0.6–0.9)
   - colsample_bytree: column sampling (0.6–0.9)
   - reg_alpha, reg_lambda: L1/L2; add e.g. 0.1–1.0 to reduce overfitting

2. FEATURE ENGINEERING (in add_date_features, merge_oil, build_features)
   - Date: add payday (e.g. 15th, last day of month), quarter, is_month_start/end
   - Oil:   add more rolling windows (e.g. 14, 60), or oil YoY change
   - Holiday: use "transferred" flag, or merge local/regional by store city
   - Store×Family: add lag/rolling mean sales per (store_nbr, family) if you add
     historical aggregates (requires building rollups from train and merging to test)

3. VALIDATION
   - --val-days: use 14–30 days; longer val = more stable RMSLE estimate

4. TUNING COMMAND
   - python train_and_submit.py --tune --n-trials 80 --tune-timeout 7200
================================================================================
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Tuple

# Paths (run from project root where data/ lives)
DATA_DIR = Path(__file__).resolve().parent / "data"
SUBMISSION_PATH = Path(__file__).resolve().parent / "submission.csv"
# Report images are saved here (same folder as this script)
REPORT_FIGURES_DIR = Path(__file__).resolve().parent / "report_figures"


def load_data():
    """Load and return all competition data."""
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    oil = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    transactions = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["date"])
    return train, test, stores, oil, holidays, transactions


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-based features (keep minimal for better public leaderboard)."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def prepare_holidays(holidays: pd.DataFrame) -> pd.DataFrame:
    """One row per date with holiday flags (national and any)."""
    # Exclude "Transfer" type - those are moved holidays
    h = holidays[holidays["type"] != "Transfer"].copy()
    h["is_holiday"] = 1
    national = h[h["locale"] == "National"][["date", "is_holiday"]].drop_duplicates()
    national = national.rename(columns={"is_holiday": "is_national_holiday"})
    any_holiday = h[["date", "is_holiday"]].drop_duplicates()
    any_holiday = any_holiday.rename(columns={"is_holiday": "is_holiday_any"})
    return national, any_holiday


def merge_oil(df: pd.DataFrame, oil: pd.DataFrame) -> pd.DataFrame:
    """Merge oil price and fill missing; add rolling features."""
    oil = oil.copy()
    oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce")
    oil = oil.sort_values("date")
    oil["oil_rolling_mean_7"] = oil["dcoilwtico"].rolling(7, min_periods=1).mean()
    oil["oil_rolling_mean_30"] = oil["dcoilwtico"].rolling(30, min_periods=1).mean()
    oil = oil.ffill().bfill()
    return df.merge(oil, on="date", how="left")


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    transactions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature sets for train and test."""
    national_hol, any_hol = prepare_holidays(holidays)

    def _process(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        df = add_date_features(df)
        df = df.merge(stores, on="store_nbr", how="left")
        df = merge_oil(df, oil)
        df = df.merge(national_hol, on="date", how="left")
        df = df.merge(any_hol, on="date", how="left")
        df["is_national_holiday"] = df["is_national_holiday"].fillna(0).astype(int)
        df["is_holiday_any"] = df["is_holiday_any"].fillna(0).astype(int)
        # Transactions: daily store-level (use for train; test we'll merge historical mean or leave NaN and fill)
        if is_train:
            df = df.merge(
                transactions,
                on=["date", "store_nbr"],
                how="left",
            )
            df["transactions"] = df["transactions"].fillna(0)
        else:
            # For test, use mean transactions per store from last period
            trans_agg = (
                transactions.groupby("store_nbr")["transactions"]
                .mean()
                .reset_index()
                .rename(columns={"transactions": "transactions"})
            )
            df = df.merge(trans_agg, on="store_nbr", how="left")
        # Encode categoricals
        for col in ["family", "city", "state", "type"]:
            if col in df.columns:
                df[col] = df[col].astype("category").cat.codes
        return df

    X_train = _process(train, is_train=True)
    X_test = _process(test, is_train=False)
    return X_train, X_test


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error (competition metric)."""
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def get_feature_columns():
    """Column names used as model features (exclude id, date, sales)."""
    return [
        "store_nbr",
        "family",
        "onpromotion",
        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "is_weekend",
        "cluster",
        "city",
        "state",
        "type",
        "dcoilwtico",
        "oil_rolling_mean_7",
        "oil_rolling_mean_30",
        "is_national_holiday",
        "is_holiday_any",
        "transactions",
    ]


def get_time_based_split(X_train: pd.DataFrame, train: pd.DataFrame, val_days: int = 14):
    """Return (X_tr, y_tr, X_val, y_val, val_mask) for time-based validation."""
    cutoff = X_train["date"].max() - pd.Timedelta(days=val_days)
    val_mask = (X_train["date"] > cutoff).values
    return val_mask, cutoff


def tune_hyperparameters_optuna(
    X_tr: pd.DataFrame,
    target: np.ndarray,
    X_train: pd.DataFrame,
    train: pd.DataFrame,
    val_mask: np.ndarray,
    n_trials: int = 30,
    timeout: int = 3600,
) -> Dict[str, Any]:
    """Use Optuna to find best LightGBM hyperparameters (minimizes validation RMSLE)."""
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        raise ImportError("Install optuna: pip install optuna")

    y_val = train.loc[val_mask, "sales"].values
    X_fit = X_tr[~val_mask]
    y_fit = target[~val_mask]
    X_val = X_tr[val_mask]

    def objective(trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "num_leaves": trial.suggest_int("num_leaves", 31, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            **params,
            random_state=42,
            verbosity=-1,
            n_jobs=1,  # avoid oversubscription in parallel trials
        )
        model.fit(X_fit, y_fit)
        pred = np.expm1(model.predict(X_val))
        pred = np.clip(pred, 0, None)
        return rmsle(y_val, pred)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42, n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    best = study.best_params.copy()
    best["random_state"] = 42
    best["verbosity"] = -1
    best["n_jobs"] = -1
    print(f"Best validation RMSLE: {study.best_value:.4f}")
    print("Best params:", best)
    return best


def tune_hyperparameters_random_search(
    X_tr: pd.DataFrame,
    target: np.ndarray,
    val_mask: np.ndarray,
    train: pd.DataFrame,
    n_iter: int = 20,
) -> Dict[str, Any]:
    """Fallback: use sklearn RandomizedSearchCV (no Optuna)."""
    from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
    from sklearn.metrics import make_scorer
    import lightgbm as lgb
    from scipy.stats import loguniform, randint

    y_val = train.loc[val_mask, "sales"].values
    X_val = X_tr[val_mask]

    # Single time-based split: -1 = train, 0 = validation
    split = np.where(val_mask, 0, -1)
    pds = PredefinedSplit(split)

    def _rmsle_scorer(y_true, y_pred):
        y_pred = np.clip(y_pred, 0, None)
        return -rmsle(y_true, np.expm1(y_pred))

    param_dist = {
        "n_estimators": randint(200, 1000),
        "learning_rate": loguniform(1e-2, 0.15),
        "max_depth": randint(4, 12),
        "num_leaves": randint(31, 200),
        "min_child_samples": randint(10, 80),
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    }
    model = lgb.LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1)
    scorer = make_scorer(_rmsle_scorer, greater_is_better=True)
    cv = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=n_iter,
        scoring=scorer,
        cv=pds,
        random_state=42,
        n_jobs=1,
        verbose=1,
    )
    cv.fit(X_tr, target)
    best = {**cv.best_params_, "random_state": 42, "verbosity": -1, "n_jobs": -1}
    val_pred = np.expm1(cv.predict(X_val))
    print(f"Best validation RMSLE: {rmsle(y_val, np.clip(val_pred, 0, None)):.4f}")
    print("Best params:", best)
    return best


def save_report_figures(
    train: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    model: Any,
    feature_cols: list,
    y_val: np.ndarray,
    val_pred_sales: np.ndarray,
    output_dir: Path = REPORT_FIGURES_DIR,
) -> None:
    """Save report figures to output_dir (default: report_figures/)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving report figures to: {output_dir.absolute()}")

    # 1. Sales over time (last 12 months)
    fig, ax = plt.subplots(figsize=(10, 4))
    daily = train.groupby("date")["sales"].sum().reset_index()
    cutoff_1y = daily["date"].max() - pd.Timedelta(days=365)
    daily = daily[daily["date"] >= cutoff_1y]
    ax.plot(daily["date"], daily["sales"], color="steelblue", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total daily sales")
    ax.set_title("Daily total sales (last 12 months)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "01_sales_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Sales distribution (log scale)
    fig, ax = plt.subplots(figsize=(6, 4))
    sales_nonzero = train["sales"][train["sales"] > 0]
    ax.hist(np.log1p(sales_nonzero), bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("log(1 + sales)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of sales (log scale)")
    plt.tight_layout()
    plt.savefig(output_dir / "02_sales_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Top 15 product families by total sales
    fig, ax = plt.subplots(figsize=(8, 5))
    by_family = train.groupby("family")["sales"].sum().sort_values(ascending=True).tail(15)
    by_family.plot(kind="barh", ax=ax, color="steelblue", alpha=0.8)
    ax.set_xlabel("Total sales")
    ax.set_ylabel("Product family")
    ax.set_title("Top 15 product families by total sales")
    plt.tight_layout()
    plt.savefig(output_dir / "03_sales_by_family.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Oil price over time
    fig, ax = plt.subplots(figsize=(10, 4))
    oil_plot = oil.copy()
    oil_plot["dcoilwtico"] = pd.to_numeric(oil_plot["dcoilwtico"], errors="coerce")
    oil_plot = oil_plot.dropna(subset=["dcoilwtico"])
    ax.plot(oil_plot["date"], oil_plot["dcoilwtico"], color="darkgreen", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Oil price (dcoilwtico)")
    ax.set_title("Oil price over time")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "04_oil_price_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. National holidays over time
    fig, ax = plt.subplots(figsize=(10, 3))
    national = holidays[holidays["locale"] == "National"][["date"]].drop_duplicates()
    ax.scatter(national["date"], np.ones(len(national)), marker="|", s=100, color="coral", label="National holiday")
    ax.set_xlabel("Date")
    ax.set_yticks([])
    ax.set_title("National holidays")
    ax.legend(loc="upper right")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "05_holidays_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 6. Actual vs predicted (validation)
    fig, ax = plt.subplots(figsize=(5, 5))
    sample_size = min(5000, len(y_val))
    idx = np.random.RandomState(42).choice(len(y_val), sample_size, replace=False)
    ax.scatter(y_val[idx], val_pred_sales[idx], alpha=0.3, s=10, color="steelblue")
    max_val = max(y_val.max(), val_pred_sales.max())
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="Perfect prediction")
    ax.set_xlabel("Actual sales")
    ax.set_ylabel("Predicted sales")
    ax.set_title("Validation: Actual vs predicted sales")
    ax.legend()
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig(output_dir / "06_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 7. Feature importance
    if hasattr(model, "feature_importances_"):
        fig, ax = plt.subplots(figsize=(8, 6))
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        imp.tail(20).plot(kind="barh", ax=ax, color="steelblue", alpha=0.8)
        ax.set_xlabel("Importance")
        ax.set_title("Top 20 feature importances (LightGBM)")
        plt.tight_layout()
        plt.savefig(output_dir / "07_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 8. Validation residuals
    fig, ax = plt.subplots(figsize=(6, 4))
    residuals = np.log1p(y_val) - np.log1p(np.clip(val_pred_sales, 0, None))
    ax.hist(residuals, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (log scale)")
    ax.set_ylabel("Count")
    ax.set_title("Validation residuals")
    plt.tight_layout()
    plt.savefig(output_dir / "08_residuals_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 9. Validation metrics text
    rmsle_val = rmsle(y_val, val_pred_sales)
    mae_val = np.mean(np.abs(y_val - val_pred_sales))
    from sklearn.metrics import mean_squared_error
    rmse_val = np.sqrt(mean_squared_error(y_val, val_pred_sales))
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    text = "Validation metrics\n\nRMSLE = {:.4f}  (competition metric)\nMAE = {:.2f}\nRMSE = {:.2f}".format(rmsle_val, mae_val, rmse_val)
    ax.text(0.1, 0.5, text, fontsize=12, family="monospace", verticalalignment="center")
    plt.tight_layout()
    plt.savefig(output_dir / "09_validation_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {len(list(output_dir.glob('*.png')))} figures in {output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Train and optionally tune LightGBM for Store Sales.")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning (Optuna) before training")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials when using --tune (default: 30)")
    parser.add_argument("--tune-timeout", type=int, default=3600, help="Max seconds for tuning (default: 3600)")
    parser.add_argument("--use-random-search", action="store_true", help="Use RandomizedSearchCV instead of Optuna (no extra dep)")
    parser.add_argument("--val-days", type=int, default=14, help="Last N days of train used for validation (default: 14)")
    parser.add_argument("--no-figures", action="store_true", help="Skip generating report figures")
    args = parser.parse_args()

    print("Loading data...")
    train, test, stores, oil, holidays, transactions = load_data()

    print("Building features...")
    X_train, X_test = build_features(
        train, test, stores, oil, holidays, transactions
    )

    # Target: log1p(sales) for RMSLE
    target = np.log1p(train["sales"])
    feature_cols = [c for c in get_feature_columns() if c in X_train.columns]
    X_tr = X_train[feature_cols].fillna(-1)
    X_te = X_test[feature_cols].fillna(-1)

    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not found. Install with: pip install lightgbm")
        print("Falling back to mean prediction per store_nbr x family...")
        # Fallback: predict mean log1p(sales) per (store_nbr, family)
        mean_log_sales = (
            train.assign(log_sales=np.log1p(train["sales"]))
            .groupby(["store_nbr", "family"])["log_sales"]
            .mean()
            .reset_index()
        )
        X_te_merged = test[["store_nbr", "family"]].merge(
            mean_log_sales, on=["store_nbr", "family"], how="left"
        )
        pred_log = X_te_merged["log_sales"].fillna(0).values
        submission = pd.DataFrame({"id": test["id"], "sales": np.expm1(pred_log)})
        submission["sales"] = submission["sales"].clip(lower=0)
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Saved submission to {SUBMISSION_PATH}")
        return

    # Time-based validation split
    val_mask, _ = get_time_based_split(X_train, train, val_days=args.val_days)
    X_val = X_tr[val_mask]
    y_val = train.loc[val_mask, "sales"].values

    # Hyperparameters: simple defaults that tend to generalize well on public LB
    lgb_params: Dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.7,
        "max_depth": 8,
        "num_leaves": 64,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }

    if args.tune:
        print("Running hyperparameter tuning...")
        try:
            if args.use_random_search:
                lgb_params = tune_hyperparameters_random_search(
                    X_tr, target, val_mask, train, n_iter=min(args.n_trials, 50)
                )
            else:
                lgb_params = tune_hyperparameters_optuna(
                    X_tr, target, X_train, train, val_mask,
                    n_trials=args.n_trials,
                    timeout=args.tune_timeout,
                )
        except ImportError as e:
            print(f"Tuning skipped: {e}")
            print("Use --use-random-search for tuning without Optuna, or: pip install optuna")

    model = lgb.LGBMRegressor(**lgb_params)
    if len(X_val) > 0 and not args.tune:
        model.fit(X_tr[~val_mask], target[~val_mask])
        val_pred_sales = np.expm1(model.predict(X_val))
        val_pred_sales = np.clip(val_pred_sales, 0, None)
        print(f"Validation RMSLE: {rmsle(y_val, val_pred_sales):.4f}")
        model.fit(X_tr, target)
    else:
        model.fit(X_tr, target)
        val_pred_sales = np.expm1(model.predict(X_val)) if len(X_val) > 0 else np.array([])
        val_pred_sales = np.clip(val_pred_sales, 0, None) if len(val_pred_sales) > 0 else val_pred_sales

    pred_log = model.predict(X_te)
    pred_sales = np.expm1(pred_log)
    pred_sales = np.clip(pred_sales, 0, None)

    submission = pd.DataFrame({"id": test["id"], "sales": pred_sales})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to {SUBMISSION_PATH}")
    print("First rows:")
    print(submission.head(10))

    if not args.no_figures and len(X_val) > 0:
        save_report_figures(
            train=train,
            oil=oil,
            holidays=holidays,
            model=model,
            feature_cols=feature_cols,
            y_val=y_val,
            val_pred_sales=val_pred_sales,
            output_dir=REPORT_FIGURES_DIR,
        )
    elif not args.no_figures and len(X_val) == 0:
        print("Skipping figures: no validation set.")


if __name__ == "__main__":
    main()
