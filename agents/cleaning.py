import logging
import pandas as pd

from core.config import settings

logger = logging.getLogger("mlpilot")


def data_cleaning_agent(state: dict) -> dict:
    """Fixes problems that imputation CANNOT fix:
    - rows whose target (answer) is missing
    - columns that are mostly empty (imputing them = inventing data)
    - duplicate rows
    NOTE: it does NOT impute missing values — that happens later inside
    the sklearn preprocessor, which prevents data leakage during CV.
    """

    df = state["df"].copy()
    target = state["target_column"]
    dropped_columns = []

    # ---- 1. Drop rows where the target itself is missing ----
    if target in df.columns and df[target].isnull().any():
        before = len(df)
        df = df.dropna(subset=[target])
        logger.info(f"Dropped {before - len(df)} rows with missing target '{target}'")

    # ---- 2. Drop columns that are mostly missing (except the target) ----
    missing_ratio = df.isnull().mean()
    hopeless = [
        col
        for col in missing_ratio[missing_ratio > settings.column_missing_limit].index
        if col != target
    ]
    if hopeless:
        df = df.drop(columns=hopeless)
        dropped_columns.extend(hopeless)
        logger.warning(f"Dropped columns (>{settings.column_missing_limit:.0%} missing): {hopeless}")

    # ---- 3. Drop duplicate rows ----
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        logger.info(f"Dropped {before - len(df)} duplicate rows")

    # ---- Safety net: did we clean the dataset into oblivion? ----
    if df.empty:
        raise ValueError("Dataset is empty after cleaning. Check column_missing_limit in config.yaml")

    logger.info(f"Cleaning done. Data is now {df.shape[0]} rows x {df.shape[1]} columns")

    return {
        "df": df,
        "rows": df.shape[0],
        "columns": df.shape[1],
        # recompute missing stats on the CLEANED data
        "missing_values": df.isnull().sum()[lambda s: s > 0].to_dict(),
        "dropped_columns": dropped_columns,
        "cleaned": True,   # <- the loop guard flag. Gates look for this!
    }