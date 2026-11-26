
from core.state import GraphState


# ============================================================
# 9. AUTOMATED REPORT AGENT
# ============================================================

def report_agent(state: GraphState):

    print("\n" + "=" * 60)
    print("REPORT AGENT")
    print("=" * 60)

    problem_type = state.get("problem_type", "Unknown")
    problem_spec = state.get("problem_spec", {})
    best_model = state.get("best_model_name", "Unknown")
    rows = state.get("rows", 0)
    columns = state.get("columns", 0)
    metrics = state.get("metrics", {})
    cv_results = state.get("cv_results", {})

    # Get existing diagnostic results from graph state
    diagnostics = state.get("diagnostics", {}) or {}

    summary = diagnostics.get("summary", {}) or {}
    issues = diagnostics.get("issues", []) or []
    target = diagnostics.get("target", {}) or {}
    missing = diagnostics.get("missing", {}) or {}
    correlations = diagnostics.get("correlation_pairs", []) or []

    # --------------------------------------------------------
    # DATA QUALITY SUMMARY
    # --------------------------------------------------------

    total_issues = summary.get("total", len(issues))
    severity = summary.get("by_severity", {}) or {}

    missing_cells = 0

    if isinstance(missing, dict):
        missing_cells = missing.get(
            "total_missing",
            missing.get("total", missing.get("count", 0))
        )

    if not missing_cells:
        missing_values = state.get("missing_values", 0)

        if isinstance(missing_values, (int, float)):
            missing_cells = missing_values
        elif isinstance(missing_values, dict):
            missing_cells = sum(
                v for v in missing_values.values()
                if isinstance(v, (int, float))
            )

    # --------------------------------------------------------
    # BUILD REPORT
    # --------------------------------------------------------

    report = f"""
============================================================
                    MLPilot REPORT
============================================================

DATASET
------------------------------------------------------------
Rows              : {rows}
Columns           : {columns}

DATA QUALITY ANALYSIS
------------------------------------------------------------
Missing Cells     : {missing_cells}
Total Issues      : {total_issues}
High Severity     : {severity.get('high', 0)}
Medium Severity   : {severity.get('medium', 0)}
Low Severity      : {severity.get('low', 0)}

DETECTED DATA QUALITY ISSUES
------------------------------------------------------------
"""

    if issues:
        for index, issue in enumerate(issues, start=1):
            issue_type = issue.get("type", "Unknown")
            column = issue.get("column")

            if not column:
                column = ", ".join(issue.get("columns", []))

            severity_level = issue.get("severity", "Unknown")
            detail = issue.get("detail", "No additional details")

            report += (
                f"\n{index}. {issue_type}\n"
                f"   Column(s) : {column or 'N/A'}\n"
                f"   Severity  : {severity_level}\n"
                f"   Details   : {detail}\n"
            )
    else:
        report += "No data quality issues detected.\n"

    # --------------------------------------------------------
    # CORRELATION ANALYSIS
    # --------------------------------------------------------

    report += """
CORRELATION ANALYSIS
------------------------------------------------------------
"""

    if correlations:
        for pair in correlations:
            col1 = pair.get("column_1", pair.get("feature_1", ""))
            col2 = pair.get("column_2", pair.get("feature_2", ""))
            correlation = pair.get(
                "correlation",
                pair.get("correlation_value", pair.get("r"))
            )

            if col1 or col2:
                report += (
                    f"{col1} <-> {col2}: "
                    f"{correlation}\n"
                )
            else:
                report += f"{pair}\n"
    else:
        report += "No correlation findings available.\n"

    # --------------------------------------------------------
    # TARGET PROFILE
    # --------------------------------------------------------

    report += """
TARGET PROFILE
------------------------------------------------------------
"""

    if target:
        for key, value in target.items():
            report += f"{key}: {value}\n"
    else:
        report += "Target profile not available.\n"

    # --------------------------------------------------------
    # PROBLEM INFORMATION
    # --------------------------------------------------------

    report += f"""
PROBLEM
------------------------------------------------------------
Problem Type      : {problem_type}
"""

    if problem_spec:
        for key, value in problem_spec.items():
            report += f"{key}: {value}\n"

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    report += f"""
MODEL SELECTION
------------------------------------------------------------
Best Model        : {best_model}

CROSS VALIDATION
------------------------------------------------------------
"""

    for model_name, result in cv_results.items():
        score = result.get("mean_score", 0)
        report += f"{model_name}: {score:.4f}\n"

    # --------------------------------------------------------
    # FINAL PERFORMANCE
    # --------------------------------------------------------

    report += """
FINAL TEST PERFORMANCE
------------------------------------------------------------
"""

    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            report += f"{metric}: {value:.4f}\n"
        else:
            report += f"{metric}: {value}\n"

    report += """
============================================================
                  END OF REPORT
============================================================
"""

    return {
        "report": report
    }
