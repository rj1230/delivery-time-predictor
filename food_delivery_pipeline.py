"""
Food Delivery Time Prediction
=============================

Train and serve a regression model that predicts delivery time (minutes).

Design
------
* ``prepare_features`` is the only place raw data is cleaned. It is stateless
  (no fitted statistics), so it is safe to run once before the split and again
  at inference time on raw rows.
* Everything *stateful* (imputation, encoding, scaling) lives inside one sklearn
  ``Pipeline``, so it is fitted on training folds only and saved together with
  the model. No separate scaler file, no ``get_dummies`` column drift.
* Models are compared with the same CV folds on the training split; the test
  split is touched once, for the selected model.

Usage
-----
    python food_delivery_pipeline.py train --data "Food delivery.csv" --eda
    python food_delivery_pipeline.py predict --input new_orders.csv --output preds.csv
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

log = logging.getLogger("food_delivery")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = Path("Food delivery.csv")
DEFAULT_MODEL_PATH = Path("models/delivery_time_model.joblib")
REPORT_DIR = Path("reports")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
MAX_DISTANCE_KM = 100  # anything above is treated as invalid coordinates

TARGET = "Time_taken(min)"
DATE_COL = "Order_Date"
DATE_FORMAT = "%d-%m-%Y"

MISSING_MARKERS = ["NaN", "nan", "NAN", "NA", "N/A", "null", "NULL", "None", ""]
TRAFFIC_ORDER = ["Low", "Medium", "High", "Jam"]

NUMERIC = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "multiple_deliveries",
    "Vehicle_condition",
    "distance_km",
    "order_hour",
    "order_dow",
]
ORDINAL = ["Road_traffic_density"]
NOMINAL = [
    "Weatherconditions",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]
FEATURES = NUMERIC + ORDINAL + NOMINAL

# Raw columns required to build FEATURES (also what a caller must supply at inference).
RAW_FEATURE_COLUMNS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Time_Orderd",
    DATE_COL,
    "Weatherconditions",
    "Road_traffic_density",
    "Vehicle_condition",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Festival",
    "City",
]


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    yield
    log.info("%s finished in %.1fs", label, time.perf_counter() - start)


# ---------------------------------------------------------------------------
# Stateless cleaning and feature engineering
# ---------------------------------------------------------------------------


def _clean_strings(series: pd.Series) -> pd.Series:
    """Strip whitespace and turn missing-value markers into NaN."""
    s = series.astype("string").str.strip()
    s = s.mask(s.isin(MISSING_MARKERS))
    return s.astype(object).where(s.notna(), np.nan)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in kilometres (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def parse_order_date(raw: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        _clean_strings(raw[DATE_COL]), format=DATE_FORMAT, errors="coerce"
    )


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw delivery rows into the model's feature frame.

    Stateless by design: nothing here is fitted, so it can run before the
    train/test split and again at inference time. Invalid values become NaN and
    are imputed inside the model pipeline.
    """
    missing = [c for c in RAW_FEATURE_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    df = raw[RAW_FEATURE_COLUMNS].copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = _clean_strings(df[col])

    # "conditions Sunny" -> "Sunny"
    df["Weatherconditions"] = (
        df["Weatherconditions"].astype("string").str.split().str[-1]
    )

    for col in (
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "multiple_deliveries",
        "Vehicle_condition",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ratings above 5 are invalid: treat as missing rather than inventing a 5.
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].where(
        df["Delivery_person_Ratings"].between(0, 5)
    )

    distance = pd.Series(
        haversine_km(
            df["Restaurant_latitude"],
            df["Restaurant_longitude"],
            df["Delivery_location_latitude"],
            df["Delivery_location_longitude"],
        ),
        index=df.index,
    )
    df["distance_km"] = distance.where(distance < MAX_DISTANCE_KM)

    hour = pd.to_numeric(df["Time_Orderd"].astype("string").str[:2], errors="coerce")
    df["order_hour"] = hour.where(hour.between(0, 23))
    df["order_dow"] = parse_order_date(raw).dt.dayofweek.astype("float64")

    return df[FEATURES]


def load_training_data(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Read the CSV and return (features, target, order_date) for valid rows."""
    raw = pd.read_csv(path)
    log.info("Loaded %s: %d rows x %d columns", path, *raw.shape)

    # "(min) 25" -> 25
    y = pd.to_numeric(
        raw[TARGET].astype("string").str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    X = prepare_features(raw)
    dates = parse_order_date(raw)

    keep = y.notna() & X["distance_km"].notna()
    dropped = int((~keep).sum())
    if dropped:
        log.warning(
            "Dropped %d rows with missing target or invalid coordinates", dropped
        )

    return (
        X[keep].reset_index(drop=True),
        y[keep].astype(float).reset_index(drop=True),
        dates[keep].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """All fitted preprocessing, kept inside the pipeline (no leakage, one artifact)."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:  # only the linear model needs scaling
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC),
            (
                "ord",
                Pipeline(
                    [
                        (
                            "encode",
                            OrdinalEncoder(
                                categories=[TRAFFIC_ORDER],
                                handle_unknown="use_encoded_value",
                                unknown_value=np.nan,
                            ),
                        ),
                        ("impute", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                ORDINAL,
            ),
            (
                "nom",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                NOMINAL,
            ),
        ],
        verbose_feature_names_out=False,
    )


def make_xgb_pipeline(n_jobs: int, **params) -> Pipeline:
    defaults = dict(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )
    defaults.update(params)
    return Pipeline(
        [("prep", build_preprocessor(False)), ("reg", XGBRegressor(**defaults))]
    )


def make_baselines() -> dict[str, Pipeline]:
    return {
        "linear": Pipeline(
            [("prep", build_preprocessor(True)), ("reg", LinearRegression())]
        ),
        "random_forest": Pipeline(
            [
                ("prep", build_preprocessor(False)),
                (
                    "reg",
                    RandomForestRegressor(
                        n_estimators=100,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost": make_xgb_pipeline(n_jobs=-1),
    }


XGB_PARAM_DIST = {
    "reg__n_estimators": [300, 500, 800],
    "reg__max_depth": [3, 4, 5, 6, 8],
    "reg__learning_rate": [0.03, 0.05, 0.1],
    "reg__subsample": [0.7, 0.8, 0.9, 1.0],
    "reg__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg__min_child_weight": [1, 3, 5],
    "reg__reg_alpha": [0, 0.1, 0.5, 1],
    "reg__reg_lambda": [1, 1.5, 2],
}

CV_SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _cv_metrics(mean_scores: dict[str, float]) -> dict[str, float]:
    """sklearn reports errors as negative scores; flip them back to plain MAE/RMSE."""
    return {
        "mae": -mean_scores["mae"],
        "rmse": -mean_scores["rmse"],
        "r2": mean_scores["r2"],
    }


def split_data(X, y, dates, mode: str):
    """Random split, or a time split that holds out the most recent orders."""
    if mode == "time" and dates.notna().any():
        cutoff = dates.quantile(1 - TEST_SIZE)
        test_mask = (dates > cutoff).to_numpy()
        log.info("Time split: test = orders after %s", cutoff.date())
        return X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]
    if mode == "time":
        log.warning("No parseable order dates; falling back to a random split")
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def error_breakdown(X_test: pd.DataFrame, y_test: pd.Series, pred: np.ndarray) -> None:
    """Where does the model miss? MAE (minutes) by traffic level and city type."""
    abs_err = pd.Series(np.abs(y_test.to_numpy() - pred), index=X_test.index)
    for col in ("Road_traffic_density", "City"):
        table = abs_err.groupby(X_test[col]).agg(["mean", "count"]).round(2)
        log.info("MAE by %s:\n%s", col, table.to_string())


def run_eda(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> None:
    """Save core plots to disk (no blocking plt.show). Uses pre-imputation data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    df = X.assign(**{TARGET: y})

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y, bins=30)
    ax.set(title="Distribution of delivery time", xlabel="Minutes", ylabel="Orders")
    fig.tight_layout()
    fig.savefig(out_dir / "target_distribution.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df.groupby("order_hour")[TARGET].mean().plot(
        kind="bar", ax=axes[0], title="Mean time by order hour"
    )
    df.groupby("Road_traffic_density")[TARGET].mean().reindex(TRAFFIC_ORDER).plot(
        kind="bar", ax=axes[1], title="Mean time by traffic density"
    )
    for ax in axes:
        ax.set_ylabel("Minutes")
    fig.tight_layout()
    fig.savefig(out_dir / "time_by_hour_and_traffic.png", dpi=120)
    plt.close(fig)

    cols = NUMERIC + [TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8
            )
    fig.colorbar(im)
    ax.set_title("Correlation: numeric features")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation.png", dpi=120)
    plt.close(fig)

    log.info("EDA figures saved to %s/", out_dir)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    split: str = "time",
    n_iter: int = 25,
    eda: bool = False,
    refit_full: bool = False,
) -> dict:
    X, y, dates = load_training_data(data_path)
    if eda:
        run_eda(X, y, REPORT_DIR)

    X_train, X_test, y_train, y_test = split_data(X, y, dates, split)
    log.info("Train rows: %d | Test rows: %d", len(X_train), len(X_test))

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    candidates: dict[str, tuple[Pipeline, dict[str, float]]] = {}

    # 1. Baselines: same folds, CV on the training split only.
    for name, pipe in make_baselines().items():
        with timed(f"CV {name}"):
            scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=CV_SCORING)
        candidates[name] = (
            pipe,
            _cv_metrics({k: float(np.mean(scores[f"test_{k}"])) for k in CV_SCORING}),
        )

    # 2. Tune XGBoost only. Search parallelises across candidates (n_jobs=-1)
    #    while each model stays single-threaded, so cores are not oversubscribed.
    with timed(f"XGBoost search ({n_iter} candidates x {CV_FOLDS} folds)"):
        search = RandomizedSearchCV(
            make_xgb_pipeline(n_jobs=1),
            XGB_PARAM_DIST,
            n_iter=n_iter,
            cv=cv,
            scoring=CV_SCORING,
            refit="mae",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        search.fit(X_train, y_train)
    log.info("Best XGBoost parameters: %s", search.best_params_)
    best = search.best_index_
    candidates["xgboost_tuned"] = (
        search.best_estimator_,  # already refit on the full training split
        _cv_metrics(
            {k: float(search.cv_results_[f"mean_test_{k}"][best]) for k in CV_SCORING}
        ),
    )

    # 3. Select on CV MAE (never on the test set).
    comparison = pd.DataFrame({n: m for n, (_, m) in candidates.items()}).T.round(3)
    log.info(
        "CV comparison (train split, lower MAE/RMSE is better):\n%s",
        comparison.to_string(),
    )
    winner = comparison["mae"].idxmin()
    log.info("Selected model: %s", winner)

    model = candidates[winner][0]
    if winner != "xgboost_tuned":  # the search already refit this one
        model.fit(X_train, y_train)

    # 4. Touch the test set once.
    test_pred = model.predict(X_test)
    test_metrics = regression_metrics(y_test, test_pred)
    log.info(
        "Test  MAE: %.3f min | RMSE: %.3f | R2: %.3f",
        test_metrics["mae"],
        test_metrics["rmse"],
        test_metrics["r2"],
    )
    error_breakdown(X_test, y_test, test_pred)

    if refit_full:
        log.info("Refitting selected model on train + test for the shipped artifact")
        model.fit(X, y)

    # 5. One artifact: the whole pipeline plus provenance.
    artifact = {
        "pipeline": model,
        "features": FEATURES,
        "raw_columns": RAW_FEATURE_COLUMNS,
        "selected_model": winner,
        "cv_comparison": comparison.to_dict(),
        "test_metrics": test_metrics,
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split": split,
            "refit_full": refit_full,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)
    log.info("Saved model artifact to %s", model_path)
    return artifact


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def load_artifact(model_path: Path) -> dict:
    artifact = joblib.load(model_path)
    trained_with = artifact["meta"]["sklearn"]
    if trained_with != sklearn.__version__:
        log.warning(
            "Model was trained with scikit-learn %s but %s is installed",
            trained_with,
            sklearn.__version__,
        )
    return artifact


def predict_dataframe(artifact: dict, raw: pd.DataFrame) -> np.ndarray:
    """Predict delivery minutes for raw rows (same columns as the training CSV)."""
    X = prepare_features(raw)
    invalid = int(X["distance_km"].isna().sum())
    if invalid:
        log.warning("%d rows have invalid coordinates; distance imputed", invalid)
    return artifact["pipeline"].predict(X)


def predict_file(model_path: Path, input_path: Path, output_path: Path) -> pd.DataFrame:
    artifact = load_artifact(model_path)
    raw = pd.read_csv(input_path)
    out = pd.DataFrame(
        {"predicted_minutes": np.round(predict_dataframe(artifact, raw), 1)}
    )
    if "ID" in raw.columns:
        out.insert(0, "ID", raw["ID"].to_numpy())
    out.to_csv(output_path, index=False)
    log.info("Wrote %d predictions to %s", len(out), output_path)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train, compare, tune and save a model")
    t.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    t.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    t.add_argument("--split", choices=["time", "random"], default="time")
    t.add_argument("--n-iter", type=int, default=25, help="XGBoost search candidates")
    t.add_argument("--eda", action="store_true", help="Save EDA figures to reports/")
    t.add_argument(
        "--refit-full", action="store_true", help="Refit on train+test before saving"
    )

    p = sub.add_parser("predict", help="Predict delivery time for raw order rows")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("predictions.csv"))

    return parser.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    if args.command == "train":
        train(args.data, args.model, args.split, args.n_iter, args.eda, args.refit_full)
    else:
        predict_file(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
