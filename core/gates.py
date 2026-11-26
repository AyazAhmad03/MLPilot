from core.config import settings


# ============================================================
# GATE 1 — DATA QUALITY GATE  (v2: diagnostics-driven)
# ============================================================

# Severity weights for the profiler's issue levels.
# Must stay in sync with the severities used in core/diagnostics.py.
_SEVERITY_WEIGHT = {"high": 1.0, "medium": 0.5, "low": 0.1}

# Issues the cleaning agent CANNOT fix — they must not trigger a cleaning
# detour (they'd cause: gate -> cleaning -> nothing fixed -> wasted cycle).
# Class imbalance is preparation's job (SMOTE / class_weight), not cleaning's.
_NOT_CLEANING_FIXABLE = {"class_imbalance"}


def data_quality_gate(state: dict) -> str:
    """Stands after data_analysis.
    Decides: is the dataset clean enough to continue, or should
    the cleaning agent run first?

    v2 — severity-weighted, deterministic:

        issue_score = sum(weight of every cleaning-fixable issue) / n_columns

      high=1.0, medium=0.5, low=0.1. "ready" requires BOTH:
        - missing_ratio <= settings.missing_ratio_limit   (legacy floor)
        - issue_score  <= issue limit                     (new)

    Falls back to the legacy missing-ratio-only rule when diagnostics is
    absent from state, so an older analysis agent still routes correctly.
    """

    # Loop guard: if we already cleaned once, don't clean again.
    # Without this check -> analysis -> cleaning -> analysis -> cleaning ... forever.
    if state.get("cleaned"):
        return "ready"

    df = state["df"]
    total_cells = df.shape[0] * df.shape[1]

    # if the dataset has no cells at all, treat it as ready
    if total_cells == 0:
        return "ready"

    total_missing = sum(state.get("missing_values", {}).values())
    missing_ratio = total_missing / total_cells

    # Legacy behavior preserved verbatim as the FLOOR: too much missing
    # data always means cleaning, no matter what the issue score says.
    if missing_ratio > settings.missing_ratio_limit:
        return "needs_cleaning"

    diagnostics = state.get("diagnostics")
    if not diagnostics or diagnostics.get("error"):
        # no usable diagnostics (older analysis version in the graph)
        # -> legacy missing-ratio rule only
        return "ready"

    # Optional settings key: works without touching config.py, but is picked
    # up automatically if you later add gate_score_limit to Settings.
    score_limit = getattr(settings, "gate_score_limit", 0.35)

    n_cols = max(int(diagnostics.get("n_columns", df.shape[1])), 1)
    issues = [
        i for i in diagnostics.get("issues", [])
        if i.get("type") not in _NOT_CLEANING_FIXABLE
    ]
    penalty = sum(_SEVERITY_WEIGHT.get(i.get("severity"), 0.5) for i in issues) / n_cols

    print(
        f"GATE 1 (data quality)  "
        f"missing_ratio={missing_ratio:.3f} / {settings.missing_ratio_limit}   "
        f"issue_score={penalty:.3f} / {score_limit}   "
        f"({len(issues)} cleaning-fixable issues)"
    )

    if penalty > score_limit:
        return "needs_cleaning"
    return "ready"


# ============================================================
# GATE 2 — PERFORMANCE GATE  (unchanged)
# ============================================================
def performance_gate(state: dict) -> str:
    """Stands after evaluation.
    Decides: is the model good enough, or should the tuning
    agent try to improve it first?"""

    # Loop guard: we only allow ONE tuning attempt.
    # Without this -> tuning -> training -> evaluation -> tuning ... forever.
    if not settings.enable_tuning or state.get("tuning_attempted"):
        return "report"

    metrics = state["metrics"]

    # classification report has "f1_score", regression report has "r2_score"
    threshold = (
        settings.classification_threshold
        if state["problem_type"] == "classification"
        else settings.regression_threshold
    )
    score = metrics.get("f1_score", metrics.get("r2_score", 0))

    if score < threshold:
        return "tune"
    return "report"