"""core/diagnostics.py - deterministic dataset profiler for MLPilot.

Facts, not decisions:  profile_dataset(df, target=None, thresholds=None) -> dict

* Output is plain JSON-safe Python (no numpy types).
* The input DataFrame is never mutated; no LLM calls.
* `issues` is the flat, severity-tagged list consumed by core/gates.py (Gate 1)
  and the cleaning-plan rule engine.
* Call normalize_frame() in the real cleaning step too, so the pipeline sees the
  same data that was profiled (text numbers, blank strings, bool columns).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["profile_dataset", "diagnostics_to_text", "normalize_frame", "DEFAULT_THRESHOLDS"]

DEFAULT_THRESHOLDS: dict[str, float] = {      # mirrors config.yaml keys
    "missing_drop_ratio": 0.50,    # share missing -> drop candidate
    "missing_flag_ratio": 0.05,    # share missing -> flag
    "skew_moderate": 0.50, "skew_high": 1.00, "kurtosis_high": 3.00,
    "outlier_ratio_high": 0.05,    # IQR-outlier share -> flag
    "corr_high": 0.95,             # |pearson| -> redundant pair
    "rare_label_ratio": 0.01, "high_cardinality": 50,
    "id_like_unique_ratio": 0.95, "min_rows_id_like": 50,
    "imbalance_ratio": 4.00,       # majority/minority -> imbalanced
    "min_class_count": 5,          # smallest class needed for stratified k-fold
    "max_class_values": 20,        # integer target uniques <= this -> classification
    "discrete_max_unique": 10,     # skip skew/outlier checks at or below this
    "normality_min_samples": 20,
    "numeric_text_ratio": 0.95,    # share of text values that parse as numbers
    "duplicate_scan_max_columns": 100,
}
_MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "?", "-"}


def _f(v: Any, digits: int = 4) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return round(x, digits) if np.isfinite(x) else None


def _pct(x: float | None) -> str:
    return f"{(x or 0.0) * 100:.1f}%"


def _base(s: pd.Series) -> dict[str, Any]:
    total, n = len(s), int(s.notna().sum())
    return {"count": n, "missing_count": total - n,
            "missing_ratio": _f((total - n) / total) if total else None}


# ---------------------------------------------------------------- normalise
def normalize_frame(df: pd.DataFrame, numeric_ratio: float = 0.95) -> tuple[pd.DataFrame, list[str]]:
    """Copy of df with str column names, bool->int, stripped text, sentinel
    strings ('', '?', 'NA', ...) -> NaN, and numbers-stored-as-text -> numeric.
    Returns (frame, columns converted from text to numeric)."""
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    converted: list[str] = []
    for c in out.columns:
        s = out[c]
        if s.dtype == bool:
            out[c] = s.astype(int)
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            t = s.astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)
            t = t.mask(t.map(lambda v: isinstance(v, str) and v.lower() in _MISSING_TOKENS))
            ok = int(t.notna().sum())
            num = pd.to_numeric(t, errors="coerce")
            if ok and num.notna().sum() / ok >= numeric_ratio:
                out[c], _ = num, converted.append(c)
            else:
                out[c] = t
    return out, converted


# ------------------------------------------------------------------ typing
def _col_type(s: pd.Series, is_target: bool, n_rows: int, th: dict[str, float]) -> str:
    if is_target:
        return "target"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    nu = int(s.nunique(dropna=True))
    if nu <= 1:
        return "constant"
    if pd.api.types.is_numeric_dtype(s):
        if (nu > 2 and n_rows >= th["min_rows_id_like"]
                and nu / n_rows >= th["id_like_unique_ratio"]
                and bool(np.all(np.mod(s.dropna().astype(float), 1) == 0))):
            return "id_like"             # integer-valued near-unique column
        return "numeric"
    if nu > th["high_cardinality"] and nu / n_rows >= th["id_like_unique_ratio"]:
        return "id_like"
    return "categorical"


# --------------------------------------------------------- column profilers
def _numeric(s: pd.Series, th: dict[str, float]) -> dict[str, Any]:
    c = s.dropna()
    nu = int(c.nunique()) if len(c) else 0
    info = _base(s)
    info.update(zero_variance=nu <= 1, skew=None, kurtosis=None, skew_level="unknown",
                heavy_tails=False, outlier_count=0, outlier_ratio=0.0, outlier_bounds=None)
    if not len(c):
        return info
    info.update(mean=_f(c.mean()), std=_f(c.std()), min=_f(c.min()), max=_f(c.max()))
    if nu <= th["discrete_max_unique"]:      # flags / small counts: skew & IQR are meaningless
        info["skew_level"] = "n/a"
        return info
    if len(c) >= th["normality_min_samples"]:
        sk, ku = _f(c.skew()), _f(c.kurt())
        a = None if sk is None else abs(sk)
        info.update(skew=sk, kurtosis=ku, heavy_tails=bool((ku or 0) > th["kurtosis_high"]),
                    skew_level="unknown" if a is None else "high" if a >= th["skew_high"]
                    else "moderate" if a >= th["skew_moderate"] else "none")
    q1, q3 = c.quantile(0.25), c.quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((c < lo) | (c > hi)).sum())
        info.update(outlier_count=n, outlier_ratio=_f(n / len(c)),
                    outlier_bounds=[_f(lo, 6), _f(hi, 6)])
    return info


def _categorical(s: pd.Series, th: dict[str, float]) -> dict[str, Any]:
    c = s.dropna().astype(str)
    vc = c.value_counts()
    info = _base(s)
    info.update(n_unique=len(vc), zero_variance=len(vc) <= 1,
                top_values=[{"label": str(k), "count": int(v), "share": _f(v / len(c))}
                            for k, v in vc.head(5).items()])
    rare = vc[vc < th["rare_label_ratio"] * len(c)] if len(c) else vc.iloc[0:0]
    info.update(rare_label_count=len(rare),
                rare_label_share=_f(rare.sum() / len(c)) if len(rare) else None)
    return info


def _datetime(s: pd.Series) -> dict[str, Any]:
    c = s.dropna()
    info = _base(s)
    info["n_unique"] = int(c.nunique())
    if len(c):
        info.update(min=str(c.min()), max=str(c.max()),
                    span_days=_f((c.max() - c.min()).total_seconds() / 86400))
    return info


# ------------------------------------------------------ target / dataset facts
def _target(df: pd.DataFrame, target: str | None, th: dict[str, float]) -> dict[str, Any]:
    """Facts only; kind_hint is a hint (problem_detection owns the decision)."""
    if target is None:
        return {"name": None, "kind_hint": "unknown", "missing": 0}
    s = df[target]
    c = s.dropna()
    info: dict[str, Any] = {"name": target, "missing": int(s.isna().sum()),
                            "n_unique": int(c.nunique()), "kind_hint": "unknown"}
    if c.empty:
        info["note"] = "target has no observed values"
        return info
    is_num = pd.api.types.is_numeric_dtype(c) and not pd.api.types.is_bool_dtype(c)
    integral = is_num and bool(np.all(np.mod(c.astype(float), 1) == 0))
    if not is_num or (integral and info["n_unique"] <= th["max_class_values"]):
        vc = c.astype(str).value_counts()
        info.update(kind_hint="classification", n_classes=len(vc), min_class_count=int(vc.iloc[-1]),
                    class_distribution={str(k): int(v) for k, v in vc.head(20).items()},
                    imbalance_ratio=_f(vc.iloc[0] / vc.iloc[-1]) if len(vc) >= 2 else None)
        if len(vc) >= 2:
            info.update(majority_class=str(vc.index[0]), minority_class=str(vc.index[-1]))
    else:
        info.update(kind_hint="regression", mean=_f(c.mean()), std=_f(c.std()),
                    min=_f(c.min()), max=_f(c.max()),
                    skew=_f(c.skew()) if len(c) >= th["normality_min_samples"] else None)
    return info


def _corr_pairs(df, types, target, th) -> list[dict[str, Any]]:
    """Feature-feature redundancy only (the target is never included)."""
    cols = [c for c in df.columns if types[c] == "numeric" and c != target]
    if len(cols) < 2:
        return []
    with np.errstate(all="ignore"):
        m = df[cols].corr().to_numpy()
    i, j = np.triu_indices(len(cols), 1)
    pairs = [{"columns": [cols[a], cols[b]], "pearson": _f(m[a, b])}
             for a, b in zip(i, j) if np.isfinite(m[a, b]) and abs(m[a, b]) >= th["corr_high"]]
    return sorted(pairs, key=lambda p: abs(p["pearson"]), reverse=True)


def _duplicate_columns(df, target, th) -> tuple[list[str], list[str], bool]:
    """(exact duplicate features, features identical to the target, scan skipped)."""
    cols = [c for c in df.columns if c != target]
    if len(cols) > int(th["duplicate_scan_max_columns"]):
        return [], [], True
    seen: dict[tuple, str] = {}
    dups: list[str] = []
    for c in cols:
        key = (str(df[c].dtype), int(pd.util.hash_pandas_object(df[c], index=False).sum()))
        if key in seen and df[c].equals(df[seen[key]]):
            dups.append(c)
        else:
            seen.setdefault(key, c)
    leaks = [c for c in cols if target is not None and df[c].equals(df[target])]
    return dups, leaks, False


# ------------------------------------------------------------------- issues
def _issues(types, cols, tgt, pairs, dups, leaks, dup_rows, converted, th) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(kind, col, sev, detail, **extra):
        out.append({"type": kind, "column": col, "severity": sev, "detail": detail, **extra})

    for col, c in cols.items():
        r = c.get("missing_ratio") or 0.0
        if r >= th["missing_drop_ratio"]:
            add("missing_high", col, "high", f"{_pct(r)} of values missing - candidate for dropping")
        elif r >= th["missing_flag_ratio"]:
            add("missing", col, "medium", f"{_pct(r)} of values missing")
        if c.get("zero_variance"):
            add("zero_variance", col, "high", "single observed value - carries no information")
        if c.get("skew_level") == "high":
            add("skew_high", col, "medium", f"highly skewed (skew={c['skew']})")
        elif c.get("skew_level") == "moderate":
            add("skew_moderate", col, "low", f"moderately skewed (skew={c['skew']})")
        if (c.get("outlier_ratio") or 0.0) >= th["outlier_ratio_high"]:
            add("outliers", col, "medium",
                f"{_pct(c['outlier_ratio'])} IQR outliers (bounds {c['outlier_bounds']})")
        if types[col] == "id_like":
            add("id_like", col, "medium", "near-unique values - identifier, drop before encoding")
        if types[col] == "categorical":
            if (c.get("n_unique") or 0) >= th["high_cardinality"]:
                add("high_cardinality", col, "medium", f"{c['n_unique']} distinct labels")
            if c.get("rare_label_count"):
                add("rare_labels", col, "low",
                    f"{c['rare_label_count']} rare labels ({_pct(c.get('rare_label_share'))} of rows)")
    for col in converted:
        add("numeric_stored_as_text", col, "low", "numbers stored as text (converted for profiling)")
    for col in dups:
        add("duplicate_column", col, "high", "exact copy of another column - redundant")
    for col in leaks:
        add("target_leakage", col, "high", "identical to the target column")
    for p in pairs:
        a, b = p["columns"]
        add("high_correlation", f"{a}|{b}", "medium", f"|r|={abs(p['pearson']):.3f} - redundant pair",
            columns=[a, b])
    if dup_rows:
        add("duplicate_rows", None, "medium", f"{dup_rows} duplicated row(s)")

    name = tgt.get("name")
    if name:
        n, ratio, mc = tgt.get("n_classes"), tgt.get("imbalance_ratio"), tgt.get("min_class_count")
        if tgt.get("missing"):
            add("target_missing", name, "high", f"{tgt['missing']} missing value(s) in the target")
        if n == 1:
            add("single_class_target", name, "high", "only one class present")
        if n and n > th["max_class_values"]:
            add("many_classes", name, "medium", f"{n} distinct classes in the target")
        if ratio is not None and ratio >= th["imbalance_ratio"]:
            add("class_imbalance", name, "high", f"majority/minority ratio {ratio:.1f}x")
        if n and n >= 2 and mc is not None and mc < th["min_class_count"]:
            add("rare_class", name, "high",
                f"smallest class has {mc} row(s) - stratified split / "
                f"{int(th['min_class_count'])}-fold CV may fail")
    return out


# --------------------------------------------------------------- entry point
def profile_dataset(df: pd.DataFrame, target: str | None = None,
                    thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Profile a DataFrame and return JSON-safe diagnostics. Read-only."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    res: dict[str, Any] = {"n_rows": 0, "n_columns": 0, "overview": {}, "missing": {}, "target": {},
                           "columns": {}, "correlation_pairs": [], "duplicate_columns": [],
                           "duplicate_rows": 0, "issues": [], "summary": {}}
    if df is None or len(df.columns) == 0 or len(df) == 0:
        res["error"] = "empty dataset - no rows or columns to profile"
        return res
    names = [str(c) for c in df.columns]
    if len(set(names)) != len(names):
        res["error"] = "duplicate column names"
        return res
    target = None if target is None else str(target)
    if target is not None and target not in names:
        res["error"] = f"target column '{target}' not found"
        return res

    df, converted = normalize_frame(df, th["numeric_text_ratio"])
    n = len(df)
    types = {c: _col_type(df[c], c == target, n, th) for c in df.columns}
    cols: dict[str, Any] = {}
    for c in df.columns:
        t = types[c]
        if t == "target":
            continue
        if t == "datetime":
            cols[c] = _datetime(df[c])
        elif t != "categorical" and pd.api.types.is_numeric_dtype(df[c]):
            cols[c] = _numeric(df[c], th)
        else:
            cols[c] = _categorical(df[c], th)
        cols[c]["type"] = t

    na = df.isna()
    miss_rows = int(na.any(axis=1).sum())
    miss_cells = int(na.to_numpy().sum())
    counts: dict[str, int] = {}
    for c, t in types.items():
        if t != "target":
            counts[t] = counts.get(t, 0) + 1
    try:
        dup_rows = int(df.duplicated().sum())
    except TypeError:                     # unhashable cells (lists/dicts)
        dup_rows = 0
    dups, leaks, skipped = _duplicate_columns(df, target, th)
    pairs = _corr_pairs(df, types, target, th)
    tgt = _target(df, target, th)
    issues = _issues(types, cols, tgt, pairs, dups, leaks, dup_rows, converted, th)
    sev = {"high": 0, "medium": 0, "low": 0}
    by_type: dict[str, int] = {}
    for i in issues:
        sev[i["severity"]] += 1
        by_type[i["type"]] = by_type.get(i["type"], 0) + 1

    res.update(
        n_rows=n, n_columns=len(df.columns), columns=cols, target=tgt, correlation_pairs=pairs,
        duplicate_columns=dups, duplicate_rows=dup_rows, issues=issues,
        overview={"column_types": counts, "numeric_stored_as_text": converted,
                  "duplicate_scan_skipped": skipped,
                  "memory_mb": _f(df.memory_usage(deep=True).sum() / 1e6, 2)},
        missing={"total_missing_cells": miss_cells, "rows_with_missing": miss_rows,
                 "rows_with_missing_ratio": _f(miss_rows / n),
                 "cells_missing_ratio": _f(miss_cells / df.size),
                 "columns_with_missing": [c for c in df.columns if na[c].any()]},
        summary={"total": len(issues), "by_severity": sev, "by_type": by_type},
    )
    return res


# ------------------------------------------------------------ LLM digest
def diagnostics_to_text(d: dict[str, Any], max_items: int = 12) -> str:
    """Compact human-readable digest - the LLM narration input."""
    if d.get("error"):
        return f"DIAGNOSTICS ERROR: {d['error']}"
    types = ", ".join(f"{k}: {v}" for k, v in sorted(d["overview"].get("column_types", {}).items()))
    m, t, s = d["missing"], d["target"], d["summary"]
    lines = [f"DATASET: {d['n_rows']} rows x {d['n_columns']} columns ({types or 'none'})",
             f"MISSING: {m['total_missing_cells']} cells in {len(m['columns_with_missing'])} column(s); "
             f"{m['rows_with_missing']} rows affected"]
    if t.get("kind_hint") == "classification":
        dist = ", ".join(f"{k}: {v}" for k, v in list(t["class_distribution"].items())[:6])
        lines.append(f"TARGET '{t['name']}': classification hint, {t['n_classes']} classes [{dist}], "
                     f"imbalance ratio {t.get('imbalance_ratio')}")
    elif t.get("kind_hint") == "regression":
        lines.append(f"TARGET '{t['name']}': regression hint, range {t['min']}..{t['max']}, "
                     f"skew {t.get('skew')}")
    if t.get("missing"):
        lines.append(f"TARGET WARNING: {t['missing']} missing target values")
    if d["correlation_pairs"]:
        ps = "; ".join(f"{p['columns'][0]}~{p['columns'][1]} (r={p['pearson']})"
                       for p in d["correlation_pairs"][:5])
        lines.append(f"REDUNDANCY: {len(d['correlation_pairs'])} high-correlation pair(s): {ps}")
    if d["duplicate_rows"]:
        lines.append(f"DUPLICATE ROWS: {d['duplicate_rows']}")
    sev = s["by_severity"]
    lines.append(f"ISSUES: {s['total']} total (high: {sev['high']}, medium: {sev['medium']}, "
                 f"low: {sev['low']})")
    shown = d["issues"][:max_items * 2]
    lines += [f"  ({i['severity']}) {i['type']} [{i['column']}]: {i['detail']}" for i in shown]
    if len(d["issues"]) > len(shown):
        lines.append(f"  (+{len(d['issues']) - len(shown)} more)")
    return "\n".join(lines)