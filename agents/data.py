import numpy as np
import pandas as pd

from core.state import GraphState
from core.diagnostics import profile_dataset, diagnostics_to_text


# ============================================================
# 1. DATA UPLOAD AGENT
# ============================================================
def data_upload_agent(state: GraphState):
    print("\n" + "=" * 60)
    print("DATA UPLOAD AGENT")
    print("=" * 60)
    file_path = state["file_path"]
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("Dataset is empty.")
        print("Dataset loaded successfully.")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")
        return {
            "df": df,
            "rows": df.shape[0],
            "columns": df.shape[1],
        }
    except Exception as e:
        raise ValueError(f"Could not load dataset: {e}")


# ============================================================
# 2. DATA ANALYSIS AGENT  (v2 - diagnostics-driven)
# ============================================================
def data_analysis_agent(state: GraphState):
    """Profiles the dataset via core/diagnostics.py.

    Facts only - no decisions, no LLM (narration stays with the insight
    agent). Stores the full diagnostics dict + a compact digest in state,
    and keeps the legacy keys (missing_values, numerical_columns,
    categorical_columns) backward-compatible for downstream agents.
    """
    print("\n" + "=" * 60)
    print("DATA ANALYSIS AGENT")
    print("=" * 60)
    df = state["df"]
    target = state.get("target_column")

    diagnostics = profile_dataset(df, target=target)
    if diagnostics.get("error"):
        raise ValueError(f"Diagnostics failed: {diagnostics['error']}")

    # ---- backward-compatible keys (downstream agents depend on these) ----
    # missing_values now comes from the NORMALIZED view: sentinel strings
    # like '?' or 'NA' are counted as missing, not smuggled in as data.
    missing_values = {
        col: c["missing_count"]
        for col, c in diagnostics["columns"].items()
        if c["missing_count"] > 0
    }
    if diagnostics["target"].get("missing"):
        missing_values[diagnostics["target"]["name"]] = diagnostics["target"]["missing"]

    # dtype-based split kept EXACTLY as before, so preparation / model
    # selection behave identically until their own upgrade files land
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number, "datetime", "datetimetz"]   # ← datetime excluded
).columns.tolist()
    # ---- console report ----
    summary = diagnostics["summary"]
    tgt = diagnostics["target"]
    print(f"Rows: {diagnostics['n_rows']}   Columns: {diagnostics['n_columns']}")
    print(f"Column types: {diagnostics['overview']['column_types']}")
    print(
        f"Missing cells: {diagnostics['missing']['total_missing_cells']} "
        f"in {len(missing_values)} column(s)"
    )
    if tgt.get("kind_hint") == "classification":
        print(
            f"Target '{tgt['name']}': classification hint, "
            f"{tgt.get('n_classes')} classes, imbalance ratio {tgt.get('imbalance_ratio')}"
        )
    elif tgt.get("kind_hint") == "regression":
        print(
            f"Target '{tgt['name']}': regression hint, "
            f"range {tgt.get('min')}..{tgt.get('max')}, skew {tgt.get('skew')}"
        )
    converted = diagnostics["overview"].get("numeric_stored_as_text", [])
    if converted:
        print(f"Numbers stored as text (converted for profiling): {converted}")

    print(
        f"Issues found: {summary['total']} "
        f"(high: {summary['by_severity']['high']}, "
        f"medium: {summary['by_severity']['medium']}, "
        f"low: {summary['by_severity']['low']})"
    )
    for issue in diagnostics["issues"][:8]:
        col = f" [{issue['column']}]" if issue.get("column") is not None else ""
        print(f"  ({issue['severity']}) {issue['type']}{col}: {issue['detail']}")

    return {
        "diagnostics": diagnostics,
        "analysis_digest": diagnostics_to_text(diagnostics),
        "missing_values": missing_values,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
    }