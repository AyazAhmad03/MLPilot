from typing import TypedDict, Any
import pandas as pd
import numpy as np

from langgraph.graph import StateGraph, START, END

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.svm import (
    SVC,
    SVR
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# STATE
# ============================================================

class GraphState(TypedDict, total=False):

    # Input
    file_path: str
    target_column: str

    # Dataset
    df: Any
    rows: int
    columns: int

    # Analysis
    missing_values: dict
    numerical_columns: list
    categorical_columns: list
    problem_type: str

    # ML data
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any

    # Models
    candidate_models: dict
    cv_results: dict
    best_model_name: str
    best_model: Any

    # Final model
    trained_pipeline: Any

    # Results
    metrics: dict
    report: str


# ============================================================
# 1. DATA UPLOAD AGENT
# ============================================================

def data_upload_agent(state: GraphState):

    print("\n" + "=" * 60)
    print("DATA UPLOAD AGENT")
    print("=" * 60)

    file_path = state["file_path"]

    try:

        df = pd.read_csv('customer_dataset.csv')

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
            print(f"  {column}: {count}")

    return {

        "missing_values": missing,

        "numerical_columns":
            numerical_columns,

        "categorical_columns":
            categorical_columns
    }


# ============================================================
# CREATING LANGGRAPH
# ============================================================

def create_graph():

    graph = StateGraph(
        GraphState
    )

    # Nodes

    graph.add_node(
        "data_upload",
        data_upload_agent
    )

    graph.add_node(
        "data_analysis",
        data_analysis_agent
    )

    
    # Edges

    graph.add_edge(
        START,
        "data_upload"
    )

    graph.add_edge(
        "data_upload",
        "data_analysis"
    )

    graph.add_edge(
        "data_analysis",
        "problem_detection"
    )

    