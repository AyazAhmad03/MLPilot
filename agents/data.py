import pandas as pd
import numpy as np

from core.state import GraphState


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
            "columns": df.shape[1]
        }

    except Exception as e:

        raise ValueError(
            f"Could not load dataset: {e}"
        )


# ============================================================
# 2. DATA ANALYSIS AGENT
# ============================================================

def data_analysis_agent(state: GraphState):

    print("\n" + "=" * 60)
    print("DATA ANALYSIS AGENT")
    print("=" * 60)

    df = state["df"]

    # Missing values

    missing = (
        df.isnull()
        .sum()
    )

    missing = missing[
        missing > 0
    ].to_dict()

    # Numerical columns

    numerical_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    # Categorical columns

    categorical_columns = (
        df.select_dtypes(
            exclude=np.number
        )
        .columns
        .tolist()
    )

    print(
        f"Numerical columns: "
        f"{len(numerical_columns)}"
    )

    print(
        f"Categorical columns: "
        f"{len(categorical_columns)}"
    )

    print(
        f"Columns with missing values: "
        f"{len(missing)}"
    )

    if missing:

        print("\nMissing values:")

        for column, count in missing.items():

            print(
                f"  {column}: {count}"
            )

    return {
        "missing_values": missing,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns
    }