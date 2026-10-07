from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from core.state import GraphState


# ============================================================
# 3. PROBLEM DETECTION AGENT
# ============================================================

def problem_detection_agent(state: GraphState):

    print("\n" + "=" * 60)
    print("PROBLEM DETECTION AGENT")
    print("=" * 60)

    df = state["df"]

    target_column = state["target_column"]

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            f"not found in dataset."
        )

    target = df[target_column]

    unique_values = target.nunique()

    # Classification detection

    if (
        target.dtype == "object"
        or str(target.dtype) == "category"
        or target.dtype == "bool"
        or unique_values <= 10
    ):

        problem_type = "classification"

    else:

        problem_type = "regression"

    print(
        f"Target column: {target_column}"
    )

    print(
        f"Unique target values: "
        f"{unique_values}"
    )

    print(
        f"Detected problem type: "
        f"{problem_type}"
    )

    return {
        "problem_type": problem_type
    }


# ============================================================
# 4. DATA PREPARATION AGENT
# ============================================================

def data_preparation_agent(state: GraphState):

    print("\n" + "=" * 60)
    print("DATA PREPARATION AGENT")
    print("=" * 60)

    df = state["df"]

    target_column = state["target_column"]

    numerical_columns = (
        state["numerical_columns"]
    )

    categorical_columns = (
        state["categorical_columns"]
    )

    problem_type = state["problem_type"]

    # Separate X and y

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # Remove target column from feature lists

    numerical_features = [
        col
        for col in numerical_columns
        if col != target_column
    ]

    categorical_features = [
        col
        for col in categorical_columns
        if col != target_column
    ]

    # Numerical preprocessing

    numerical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing

    categorical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Complete preprocessing

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # Train-test split

    if problem_type == "classification":

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
        )

    else:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )
        )

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    print("Preprocessing pipeline created.")

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor
    }