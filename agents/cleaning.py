import logging
import pandas as pd

from core.config import settings

logger = logging.getLogger("mlpilot")


# ============================================================
# PLAN BUILDER — diagnostics -> deterministic STRUCTURAL plan
# ============================================================
def _build_cleaning_plan(diagnostics: dict, target: str | None) -> dict | None:
    """Turns the profiler's facts into a fixed list of actions.

    Pure function: same diagnostics -> same plan. No LLM, no heuristics
    beyond thresholds. Returns None when there are no usable diagnostics,
    so the caller falls back to the legacy heuristics.

    STRUCTURAL actions ONLY (drop columns / drop duplicate rows).
    Imputation and outlier capping are deliberately NOT in this plan:
    doing them here on the full dataset would fit statistics (medians,
    IQR bounds) on CV test folds — leakage. They stay inside the sklearn
    preprocessor, which is fit on training folds only. The profiler's
    skew/outlier facts still inform that preprocessor via prep upgrades.
    """
    plan = {"drop_columns": [], "drop_duplicate_rows": False, "reasons": []}
    if not diagnostics or diagnostics.get("error"):
        return None

    n = max(int(diagnostics.get("n_rows", 0)), 1)
    limit = settings.column_missing_limit

    def drop(col: str, reason: str):
        # the target is NEVER a cleaning candidate
        if col == target or col in plan["drop_columns"]:
            return
        plan["drop_columns"].append(col)
        plan["reasons"].append({"action": "drop_column", "column": col, "reason": reason})

    for col, ci in diagnostics.get("columns", {}).items():
        ctype = ci.get("type")
        if ctype == "constant":
            drop(col, "single observed value — carries no information")
        elif ctype == "id_like":
            drop(col, "near-unique identifier, not a feature")
        elif ci.get("missing_count", 0) / n > limit:
            drop(col, f"{ci.get('missing_count', 0) / n:.0%} missing > {limit:.0%} limit — imputing would invent data")

    for col in diagnostics.get("duplicate_columns", []):
        drop(col, "exact duplicate of another column")

    for issue in diagnostics.get("issues", []):
        if issue.get("type") == "target_leakage" and issue.get("column"):
            drop(issue["column"], "identical to the target (leakage)")

    dup_rows = int(diagnostics.get("duplicate_rows", 0))
    if dup_rows > 0:
        plan["drop_duplicate_rows"] = True
        plan["reasons"].append(
            {"action": "drop_duplicate_rows", "column": None,
             "reason": f"{dup_rows} duplicate row(s) found"}
        )
    return plan


# ============================================================
# CLEANING AGENT  (v2 — plan-driven, legacy fallback kept)
# ============================================================
def data_cleaning_agent(state: dict) -> dict:
    """Fixes problems that imputation CANNOT fix:
    - rows whose target (answer) is missing
    - columns that are mostly empty (imputing them = inventing data)
    - zero-variance, ID-like, duplicated and target-leaking columns (v2)
    - duplicate rows (v2: only when the profiler actually found some)
    NOTE: it does NOT impute missing values — that happens later inside
    the sklearn preprocessor, which prevents data leakage during CV.
    Same is true for outlier capping — also deferred to the preprocessor.

    v2: when diagnostics are available, the actions come from
    _build_cleaning_plan() (deterministic, every action has a reason).
    An explicit `cleaning_plan` in state takes precedence; without any
    diagnostics the original heuristics run unchanged.
    """

    df = state["df"].copy()
    target = state["target_column"]
    dropped_columns = []

    # ---- 1. Drop rows where the target itself is missing (always) ----
    if target in df.columns and df[target].isnull().any():
        before = len(df)
        df = df.dropna(subset=[target])
        logger.info(f"Dropped {before - len(df)} rows with missing target '{target}'")

    # ---- 2. Column drops + duplicate rows ----
    diagnostics = state.get("diagnostics")
    plan = state.get("cleaning_plan")
    if plan is None and diagnostics and not diagnostics.get("error"):
        plan = _build_cleaning_plan(diagnostics, target)

    if plan is not None:
        # -- 2a. PLAN PATH: profiler-derived structural actions --
        to_drop = [c for c in plan.get("drop_columns", []) if c in df.columns and c != target]
        if to_drop:
            df = df.drop(columns=to_drop)
            dropped_columns.extend(to_drop)
            logger.warning(f"Dropped columns per cleaning plan: {to_drop}")
        if plan.get("drop_duplicate_rows"):
            before = len(df)
            df = df.drop_duplicates()
            logger.info(f"Dropped {before - len(df)} duplicate rows (plan)")
        for r in plan.get("reasons", []):
            where = f" [{r['column']}]" if r.get("column") else ""
            logger.info(f"plan:{r['action']}{where} — {r['reason']}")
    else:
        # -- 2b. LEGACY PATH: original heuristics, unchanged --
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