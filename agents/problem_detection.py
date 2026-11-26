import logging

from core.state import GraphState

logger = logging.getLogger("mlpilot")

# A diagnostics-reported distinct count above this is treated as regression
# even when kind_hint is missing or ambiguous.
_MAX_CLASS_CARDINALITY = 50

# Fallback heuristic (original df-based rule): a numeric target with at most
# this many distinct values is still classification.
_DF_UNIQUE_CLASS_MAX = 10

# Candidate keys under which diagnostics may store per-column stats
# (same fallback chain as core/gates.py).
_COLUMNS_KEYS = ("columns", "per_column", "column_stats", "cols")


def _num(value):
    """Best-effort float conversion; None for non-numeric input."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def problem_detection_agent(state: GraphState):

    print("\n" + "=" * 60)
    print("PROBLEM DETECTION AGENT")
    print("=" * 60)

    diagnostics = state.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    hints = diagnostics.get("target")
    hints = hints if isinstance(hints, dict) else {}

    df = state.get("df")
    target_column = state.get("target_column")

    # Original safety check, kept: validate the target whenever we can.
    if df is not None and target_column is not None and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # ------------------------------------------------------------------
    # Class-count hint from diagnostics (used for task + subtask)
    # ------------------------------------------------------------------
    n_classes = None
    for key in ("n_classes", "n_unique", "class_count"):
        n = _num(hints.get(key))
        if n is not None and n >= 2:
            n_classes = n
            break
    unique_count = int(n_classes) if n_classes is not None else None

    # ------------------------------------------------------------------
    # 1) decide the task: diagnostics hint -> class count -> df heuristic
    # ------------------------------------------------------------------
    problem_type = None
    why = ""

    kind_hint = str(hints.get("kind_hint") or "").strip().lower()
    if any(k in kind_hint for k in ("class", "binary", "categorical")):
        problem_type = "classification"
        why = f"diagnostics kind_hint = '{kind_hint}'"
    elif "regress" in kind_hint:
        problem_type = "regression"
        why = f"diagnostics kind_hint = '{kind_hint}'"

    if problem_type is None and n_classes is not None:
        if n_classes <= _MAX_CLASS_CARDINALITY:
            problem_type = "classification"
        else:
            problem_type = "regression"
        why = f"diagnostics target has {int(n_classes)} distinct value(s)"

    if problem_type is None and df is not None and target_column is not None:
        # Original heuristic, kept verbatim as the fallback.
        target = df[target_column]
        unique_values = target.nunique()
        unique_count = int(unique_values)
        if (
            target.dtype == "object"
            or str(target.dtype) == "category"
            or target.dtype == "bool"
            or unique_values <= _DF_UNIQUE_CLASS_MAX
        ):
            problem_type = "classification"
        else:
            problem_type = "regression"
        why = f"df heuristic: dtype={target.dtype}, unique={unique_values}"

    if problem_type is None:
        raise ValueError(
            "problem_detection_agent could not detect the problem type: "
            "no diagnostics target hints and no dataframe fallback available."
        )

    # ------------------------------------------------------------------
    # 2) subtask (binary / multiclass)
    # ------------------------------------------------------------------
    subtask = None
    if problem_type == "classification":
        if unique_count == 2:
            subtask = "binary"
        elif unique_count is not None and unique_count <= _MAX_CLASS_CARDINALITY:
            subtask = "multiclass"
        elif "binary" in kind_hint:
            subtask = "binary"
        else:
            print("Class count unknown; subtask left to the training agent.")

    # ------------------------------------------------------------------
    # 3) class balance (classification only)
    # ------------------------------------------------------------------
    balance, balance_detail, balance_hint = "n/a", "", None
    if problem_type == "classification":
        imb = _num(hints.get("imbalance_ratio"))
        if imb is not None and imb > 0:
            if imb > 1.0:  # majority:minority form, e.g. 9.0 == 9:1
                balance = "balanced" if imb <= 3.0 else ("imbalanced" if imb <= 10.0 else "severe")
                balance_detail = f"imbalance ratio {imb:.1f}:1"
            else:          # minority-fraction form, e.g. 0.08
                balance = "balanced" if imb >= 0.20 else ("imbalanced" if imb >= 0.05 else "severe")
                balance_detail = f"minority fraction = {imb:.3f}"
            balance_hint = {"imbalanced": "use_class_weights",
                            "severe": "use_class_weights+consider_resampling"}.get(balance)
        else:
            mcc = _num(hints.get("min_class_count"))
            if mcc is not None:
                balance_detail = f"smallest class has {int(mcc)} row(s)"
                if mcc < 10:
                    balance, balance_hint = "severe", "use_class_weights+consider_resampling"
                elif mcc < 50:
                    balance, balance_hint = "imbalanced", "use_class_weights"
                else:
                    balance = "balanced"
            else:
                balance, balance_detail = "unknown", "no imbalance stats in diagnostics"

    # ------------------------------------------------------------------
    # 4) recommended metrics
    # ------------------------------------------------------------------
    if problem_type == "classification":
        if balance in ("imbalanced", "severe"):
            metrics, metric_why = ["f1_macro", "balanced_accuracy"], "imbalanced target -> macro/rank-aware metrics"
        elif subtask == "binary":
            metrics, metric_why = ["roc_auc", "f1_macro"], "binary target -> roc_auc primary"
        else:
            metrics, metric_why = ["accuracy", "f1_macro"], "balanced classification -> accuracy primary"
    else:
        skew = str(hints.get("skew_level") or "").lower()
        if skew == "high":
            metrics, metric_why = ["mae", "r2"], "skewed target -> mae is robust to outliers"
        else:
            metrics, metric_why = ["rmse", "r2"], "regression default"

    # ------------------------------------------------------------------
    # 5) feature exclusions, prep hints, notes
    # ------------------------------------------------------------------
    cols = {}
    for key in _COLUMNS_KEYS:
        candidate = diagnostics.get(key)
        if isinstance(candidate, dict) and candidate:
            cols = candidate
            break
    exclude = [
        name for name, info in cols.items()
        if isinstance(info, dict) and info.get("type") == "id_like"
    ]

    prep_hints = []
    if balance_hint:
        prep_hints.extend(part for part in balance_hint.split("+") if part)
    if problem_type == "regression" and str(hints.get("skew_level") or "").lower() == "high":
        prep_hints.append("consider_log_transform_target")

    notes = []
    if exclude:
        listed = ", ".join(exclude[:5]) + ("..." if len(exclude) > 5 else "")
        notes.append(f"{len(exclude)} id_like column(s) excluded from features: {listed}")
    if hints.get("missing"):
        notes.append("diagnostics reported missing target values; confirm cleaning handled them")

    # ------------------------------------------------------------------
    # Console report (original style, kept)
    # ------------------------------------------------------------------
    print(f"Target column: {target_column}")
    print(f"Unique target values: {unique_count if unique_count is not None else 'n/a'}")
    print(f"Detected problem type: {problem_type}")
    print(f"Decision basis: {why}")
    if subtask:
        print(f"Subtask: {subtask}")
    if problem_type == "classification":
        suffix = f" ({balance_detail})" if balance_detail else ""
        print(f"Class balance: {balance}{suffix}")
    print(f"Recommended metrics: {', '.join(metrics)}")
    if prep_hints:
        print(f"Prep hints: {', '.join(prep_hints)}")
    logger.info("problem detection: %s (%s)", problem_type, why)

    spec = {
        "task": problem_type,
        "subtask": subtask,
        "decided_by": why,
        "n_classes": int(unique_count) if problem_type == "classification" and unique_count else None,
        "balance": {"status": balance, "detail": balance_detail},
        "metrics": metrics,
        "metric_why": metric_why,
        "exclude_from_features": exclude,
        "prep_hints": prep_hints,
        "notes": notes,
    }
    return {
        "problem_type": problem_type,
        "problem_spec": spec,
    }