import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from core.config import settings
from core.state import GraphState

logger = logging.getLogger("mlpilot")


def looks_like_datetime(series: pd.Series) -> bool:
    """True if a column's values look like dates (2023-01-15, 15/03/2022...)."""
    sample = series.dropna().astype(str).head(20)
    if sample.empty or sample.str.contains(r"[-/]").mean() < 0.8:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() >= 0.8
    except Exception:
        return False


def data_preparation_agent(state: GraphState):

    print("\n" + "=" * 60)
    print("DATA PREPARATION AGENT")
    print("=" * 60)

    df = state["df"].copy()
    target_column = state["target_column"]
    problem_type = state["problem_type"]

    # ---- 1. Expand date columns FIRST (before ID detection) ----
    # A date column is unique-per-row, so the ID heuristic would swallow it.
    # Expanding first turns it into low-cardinality numeric features
    # (year/month/dayofweek) that safely pass the ID check afterwards.
    date_columns = [
        col for col in df.columns
        if col != target_column and looks_like_datetime(df[col])
    ]
    for col in date_columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = parsed.dt.year
        df[f"{col}_month"] = parsed.dt.month
        df[f"{col}_dayofweek"] = parsed.dt.dayofweek
        df = df.drop(columns=[col])
    if date_columns:
        print(f"Expanded date columns: {date_columns}")

    # ---- 2. Drop ID-like columns (unique per row = pure noise for models) ----
    id_columns = [
        col for col in df.columns
        if col != target_column and (
            df[col].nunique() == len(df)
            or str(col).strip().lower() in {"id", "index", "unnamed: 0"}
            or str(col).strip().lower().endswith("_id")
        )
    ]
    if id_columns:
        df = df.drop(columns=id_columns)
        print(f"Dropped ID-like columns: {id_columns}")

    # ---- 3. Separate features and target ----
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Recompute feature types AFTER dropping/expanding (state lists are stale now)
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # ---- 4. Preprocessor (unchanged logic) ----
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    # ---- 5. Safe stratify (crashes if any class has <2 samples) ----
    stratify = None
    if problem_type == "classification":
        counts = y.value_counts()
        if (counts >= 2).all():
            stratify = y
        else:
            print("Skipping stratify (some class has fewer than 2 samples)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.test_size,       # was hardcoded 0.2
        random_state=settings.random_state, # was hardcoded 42
        stratify=stratify,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Numerical features: {len(numerical_features)} | Categorical features: {len(categorical_features)}")
    print("Preprocessing pipeline created.")

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
    }