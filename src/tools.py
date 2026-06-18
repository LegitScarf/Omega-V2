import os
import json
import logging
import math
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from typing import Dict, List, Any, Optional
from crewai.tools import tool

from .utils import get_output_path, load_json_file

logger = logging.getLogger("Omega.Tools")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

# ── In-memory dataset registry ─────────────────────────────────────────────────
# The uploaded dataframe is registered here by app.py before the crew kicks off.
# All tools read from this registry instead of re-reading from disk.
# Key: session_id (str), Value: pd.DataFrame

_dataset_registry: Dict[str, pd.DataFrame] = {}
_active_session_id: Optional[str] = None


def register_dataset(session_id: str, df: pd.DataFrame) -> None:
    """
    Called by app.py after file upload.
    Stores the dataframe and marks it as the active session.
    """
    global _active_session_id
    _dataset_registry[session_id] = df.copy()
    _active_session_id = session_id
    logger.info(f"Dataset registered — session={session_id}, "
                f"rows={len(df)}, cols={len(df.columns)}")


def _get_active_df() -> Optional[pd.DataFrame]:
    """Return the active session's dataframe, or None if not registered."""
    if _active_session_id and _active_session_id in _dataset_registry:
        return _dataset_registry[_active_session_id]
    return None


# ── File I/O helpers ───────────────────────────────────────────────────────────

def _write_output_json(filename: str, payload: Dict[str, Any]) -> None:
    output_path = get_output_path(filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_serialise)


def _load_output_json(filename: str) -> Dict[str, Any]:
    output_path = get_output_path(filename)
    if not os.path.exists(output_path):
        return {}
    try:
        return load_json_file(output_path)
    except Exception as exc:
        logger.warning("Could not load %s: %s", filename, exc)
        return {}


def _json_serialise(obj: Any) -> Any:
    """
    JSON serialiser for types that are not natively serialisable.
    Handles numpy scalars, NaN/Inf, and pandas NA.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.NA.__class__):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ── Private helpers ────────────────────────────────────────────────────────────

def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _get_categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _parse_column_list(raw: Any) -> List[str]:
    """
    Safely parse target_columns from the inputs dict.
    Handles JSON strings, Python lists, and empty values.
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(c).strip() for c in raw if str(c).strip()]
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed if str(c).strip()]
        except (json.JSONDecodeError, ValueError):
            pass
        # Comma-separated fallback
        return [c.strip() for c in stripped.split(",") if c.strip()]
    return []


def _resolve_columns(df: pd.DataFrame, target_columns: List[str]) -> tuple[List[str], List[str]]:
    """
    Split target_columns into found (present in df) and missing.
    Returns (found_columns, missing_columns).
    """
    found = [c for c in target_columns if c in df.columns]
    missing = [c for c in target_columns if c not in df.columns]
    return found, missing


def _iqr_outlier_bounds(series: pd.Series) -> tuple[float, float]:
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    return (q1 - 1.5 * iqr), (q3 + 1.5 * iqr)


def _infer_chart_type(intent_type: str, desired_output: List[str],
                      target_columns: List[str], df: Optional[pd.DataFrame] = None) -> str:
    """
    Determine the best Plotly chart type from intent and column types.

    Priority:
      1. If desired_output explicitly names a chart type, use it.
      2. Otherwise derive from intent_type and column dtypes.
    """
    chart_map = {
        "bar_chart": "bar",
        "scatter":   "scatter",
        "histogram": "histogram",
        "line_chart": "line",
        "box_plot":  "box",
        "pie_chart": "pie",
    }
    for key, chart in chart_map.items():
        if key in desired_output:
            return chart

    intent_defaults = {
        "aggregation":   "bar",
        "ranking":       "bar",
        "comparison":    "bar",
        "correlation":   "scatter",
        "distribution":  "histogram",
        "trend":         "line",
        "descriptive":   "histogram",
        "filter":        "bar",
    }
    chart = intent_defaults.get(intent_type, "bar")

    # Refine: if both columns are numeric and intent is correlation → scatter
    if df is not None and len(target_columns) >= 2:
        numeric_cols = _get_numeric_columns(df)
        if all(c in numeric_cols for c in target_columns[:2]) and intent_type == "correlation":
            chart = "scatter"

    return chart


def _humanise_column(col: str) -> str:
    """Convert snake_case or camelCase column names to readable labels."""
    import re
    # camelCase → spaces
    col = re.sub(r"([a-z])([A-Z])", r"\1 \2", col)
    # underscores → spaces
    col = col.replace("_", " ").strip().title()
    return col


# ── EDA Tools ─────────────────────────────────────────────────────────────────

@tool("Compute EDA Statistics")
def compute_eda_stats(target_columns: str) -> Dict[str, Any]:
    """
    Compute descriptive statistics for the specified columns of the active dataset.
    Pass target_columns as a JSON array string or comma-separated column names.
    Pass an empty string to analyse all columns.
    """
    try:
        df = _get_active_df()
        if df is None or df.empty:
            return {
                "status": "failed",
                "error": "no_dataset",
                "message": "No dataset is registered. Upload a file first.",
            }

        cols = _parse_column_list(target_columns)
        if not cols:
            cols = df.columns.tolist()

        found_cols, missing_cols = _resolve_columns(df, cols)
        if not found_cols:
            return {
                "status": "failed",
                "error": "no_valid_columns",
                "message": f"None of the requested columns exist in the dataset.",
                "missing_columns": missing_cols,
            }

        working_df = df[found_cols]
        numeric_cols = _get_numeric_columns(working_df)
        categorical_cols = _get_categorical_columns(working_df)

        # Summary statistics for numeric columns
        summary_stats: Dict[str, Any] = {}
        for col in numeric_cols:
            series = working_df[col].dropna()
            if series.empty:
                summary_stats[col] = {"error": "all_null"}
                continue
            summary_stats[col] = {
                "mean":   round(_coerce_float(series.mean()), 4),
                "median": round(_coerce_float(series.median()), 4),
                "std":    round(_coerce_float(series.std()), 4),
                "min":    round(_coerce_float(series.min()), 4),
                "max":    round(_coerce_float(series.max()), 4),
                "q1":     round(_coerce_float(series.quantile(0.25)), 4),
                "q3":     round(_coerce_float(series.quantile(0.75)), 4),
                "count":  int(series.count()),
            }

        # Cardinality for categorical columns
        for col in categorical_cols:
            series = working_df[col]
            top_values = series.value_counts().head(5).to_dict()
            summary_stats[col] = {
                "unique_count": int(series.nunique()),
                "top_values":   {str(k): int(v) for k, v in top_values.items()},
                "count":        int(series.count()),
            }

        # Null report
        null_report = {
            col: round(float(working_df[col].isna().mean() * 100), 2)
            for col in found_cols
        }

        # Data types
        data_types = {
            col: str(working_df[col].dtype)
            for col in found_cols
        }

        # Skewness for numeric columns
        skewness = {}
        for col in numeric_cols:
            series = working_df[col].dropna()
            if len(series) >= 3:
                skewness[col] = round(_coerce_float(series.skew()), 4)

        # Plain-English observations (max 5)
        observations = []
        for col in numeric_cols[:3]:
            s = summary_stats.get(col, {})
            if "mean" in s and "std" in s:
                observations.append(
                    f"{_humanise_column(col)} has a mean of {s['mean']} "
                    f"with a standard deviation of {s['std']}."
                )
        for col, null_pct in null_report.items():
            if null_pct > 10 and len(observations) < 5:
                observations.append(
                    f"{_humanise_column(col)} has {null_pct}% missing values, "
                    f"which may affect analysis accuracy."
                )
        for col, sk in skewness.items():
            if abs(sk) > 1.0 and len(observations) < 5:
                direction = "right" if sk > 0 else "left"
                observations.append(
                    f"{_humanise_column(col)} is strongly skewed to the {direction} "
                    f"(skewness={sk}), suggesting outliers may be present."
                )

        result = {
            "status":        "success",
            "summary_stats": summary_stats,
            "null_report":   null_report,
            "data_types":    data_types,
            "skewness":      skewness,
            "missing_columns": missing_cols,
            "observations":  observations[:5],
            "error":         None,
        }

        _write_output_json("eda_result.json", result)
        logger.info(f"EDA complete — {len(found_cols)} columns analysed")
        return result

    except Exception as e:
        logger.exception(f"EDA Exception: {e}")
        return {"status": "failed", "error": "exception", "message": str(e)}


@tool("Detect Outliers")
def detect_outliers(target_columns: str) -> Dict[str, Any]:
    """
    Detect outliers in numeric columns using the IQR method.
    Pass target_columns as a JSON array string or comma-separated column names.
    Pass an empty string to check all numeric columns.
    """
    try:
        df = _get_active_df()
        if df is None or df.empty:
            return {
                "status": "failed",
                "error": "no_dataset",
                "message": "No dataset is registered.",
            }

        cols = _parse_column_list(target_columns)
        if not cols:
            cols = _get_numeric_columns(df)
        else:
            found, _ = _resolve_columns(df, cols)
            cols = [c for c in found if c in _get_numeric_columns(df)]

        if not cols:
            return {
                "status": "failed",
                "error": "no_numeric_columns",
                "message": "No numeric columns found in the requested set.",
            }

        outlier_flags = []
        for col in cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            lower, upper = _iqr_outlier_bounds(series)
            mask = (series < lower) | (series > upper)
            count = int(mask.sum())
            outlier_flags.append({
                "column":        col,
                "outlier_count": count,
                "lower_bound":   round(lower, 4),
                "upper_bound":   round(upper, 4),
                "pct_affected":  round(float(count / len(series) * 100), 2),
            })

        result = {
            "status":        "success",
            "outlier_flags": outlier_flags,
            "columns_checked": cols,
            "error":         None,
        }

        logger.info(f"Outlier detection complete — {len(cols)} columns checked")
        return result

    except Exception as e:
        logger.exception(f"Outlier Detection Exception: {e}")
        return {"status": "failed", "error": "exception", "message": str(e)}


# ── Query Tools ────────────────────────────────────────────────────────────────

@tool("Generate SQL Query")
def generate_sql(user_query: str, dataset_schema: str, intent_type: str,
                 target_columns: str, filters: str) -> Dict[str, Any]:
    """
    Generate a DuckDB SQL query from the parsed intent fields.
    Validates that all referenced columns exist in the schema before returning.
    target_columns and filters must be JSON-encoded strings.
    """
    try:
        df = _get_active_df()
        if df is None:
            return {"status": "failed", "error": "no_dataset",
                    "message": "No dataset registered."}

        cols = _parse_column_list(target_columns)
        try:
            filter_dict: Dict[str, Any] = json.loads(filters) if filters.strip() else {}
        except (json.JSONDecodeError, AttributeError):
            filter_dict = {}

        valid_cols = df.columns.tolist()
        invalid_cols = [c for c in cols if c not in valid_cols]
        if invalid_cols:
            return {
                "status":         "failed",
                "error":          "invalid_columns",
                "message":        f"Columns not in dataset: {invalid_cols}",
                "invalid_columns": invalid_cols,
            }

        # Build SQL based on intent_type
        where_clauses = []
        for col, val in filter_dict.items():
            if col not in valid_cols:
                logger.warning(f"Filter column '{col}' not in dataset — skipping")
                continue
            if isinstance(val, str):
                where_clauses.append(f'"{col}" = \'{val}\'')
            elif isinstance(val, (int, float)):
                where_clauses.append(f'"{col}" = {val}')

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        assumptions = []

        if intent_type in ("aggregation", "ranking", "comparison"):
            # Separate numeric and categorical columns to prevent grouping/aggregating incorrectly
            numeric_valid = _get_numeric_columns(df)
            categorical_valid = _get_categorical_columns(df)
            
            num_cols = [c for c in cols if c in numeric_valid]
            cat_cols = [c for c in cols if c in categorical_valid]
            
            # Fallback handling if column categorization yields empty subsets
            if not cat_cols and cols:
                cat_cols = [cols[0]]
                num_cols = [c for c in cols[1:] if c in numeric_valid]
            elif not cat_cols:
                cat_cols = [valid_cols[0]]
                
            agg_fn = "AVG" if intent_type == "comparison" else "SUM"
            order = "DESC" if intent_type == "ranking" else ""
            limit = "LIMIT 20" if intent_type == "ranking" else ""
            
            select_items = [f'"{c}"' for c in cat_cols]
            group_items = [f'"{c}"' for c in cat_cols]
            
            if num_cols:
                for nc in num_cols:
                    select_items.append(f'{agg_fn}("{nc}") AS "{nc.lower()}_total"')
                order_by_col = f'"{num_cols[0].lower()}_total"'
            else:
                select_items.append('COUNT(*) AS count')
                order_by_col = '"count"'
                assumptions.append("No numeric metric column specified — defaulting to COUNT(*)")
                
            select_sql = ", ".join(select_items)
            group_sql = f"GROUP BY {', '.join(group_items)}"
            order_sql = f"ORDER BY {order_by_col} {order}".strip()
            
            sql = f"SELECT {select_sql} FROM dataset {where_sql} {group_sql} {order_sql} {limit}".strip()

        elif intent_type == "correlation":
            if len(cols) < 2:
                numeric = _get_numeric_columns(df)
                cols = numeric[:2]
                assumptions.append(f"Defaulting to first two numeric columns: {cols}")
            sql = f'SELECT "{cols[0]}", "{cols[1]}" FROM dataset {where_sql} LIMIT 500'

        elif intent_type == "distribution":
            col = cols[0] if cols else _get_numeric_columns(df)[0]
            sql = f'SELECT "{col}" FROM dataset {where_sql} LIMIT 1000'

        elif intent_type == "trend":
            # Trends should also group and aggregate by date cleanly
            numeric_valid = _get_numeric_columns(df)
            categorical_valid = _get_categorical_columns(df)
            
            time_cols = [c for c in cols if c not in numeric_valid]
            num_cols = [c for c in cols if c in numeric_valid]
            
            if not time_cols:
                time_col = cols[0] if cols else valid_cols[0]
            else:
                time_col = time_cols[0]
                
            if not num_cols:
                select_sql = f'"{time_col}", COUNT(*) AS count'
                group_sql = f'GROUP BY "{time_col}"'
                order_sql = f'ORDER BY "{time_col}"'
            else:
                metric_col = num_cols[0]
                select_sql = f'"{time_col}", SUM("{metric_col}") AS "{metric_col.lower()}_total"'
                group_sql = f'GROUP BY "{time_col}"'
                order_sql = f'ORDER BY "{time_col}"'
                
            sql = f"SELECT {select_sql} FROM dataset {where_sql} {group_sql} {order_sql} LIMIT 500"

        elif intent_type == "filter":
            select_cols = ", ".join(f'"{c}"' for c in cols) if cols else "*"
            sql = f"SELECT {select_cols} FROM dataset {where_sql} LIMIT 500"

        else:
            # descriptive — return full dataset sample
            sql = f"SELECT * FROM dataset {where_sql} LIMIT 200"
            assumptions.append("Descriptive intent — returning a full dataset sample (max 200 rows)")

        result = {
            "status":      "success",
            "sql_query":   sql.strip(),
            "intent_type": intent_type,
            "assumptions": assumptions,
            "error":       None,
        }

        logger.info(f"SQL generated — intent={intent_type}, query={sql.strip()[:80]}")
        return result

    except Exception as e:
        logger.exception(f"SQL Generation Exception: {e}")
        return {"status": "failed", "error": "exception", "message": str(e)}


@tool("Execute SQL Query")
def execute_sql(sql_query: str) -> Dict[str, Any]:
    """
    Execute a DuckDB SQL query against the active dataset.
    The dataset is available as the table name 'dataset'.
    Returns up to 500 rows. Attempts one auto-correction on failure.
    """
    try:
        df = _get_active_df()
        if df is None:
            return {"status": "failed", "error": "no_dataset",
                    "message": "No dataset registered."}

        def _run(query: str, dataframe: pd.DataFrame) -> pd.DataFrame:
            con = duckdb.connect()
            con.register("dataset", dataframe)
            return con.execute(query).df()

        try:
            result_df = _run(sql_query, df)
        except Exception as first_err:
            logger.warning(f"SQL execution failed: {first_err} — attempting auto-correction")
            # Auto-correction: strip any trailing semicolons and reattempt
            corrected = sql_query.rstrip(";").strip()
            try:
                result_df = _run(corrected, df)
                sql_query = corrected
            except Exception as second_err:
                logger.exception(f"SQL auto-correction also failed: {second_err}")
                return {
                    "status":     "failed",
                    "error":      "execution_failed",
                    "message":    str(second_err),
                    "sql_query":  sql_query,
                    "result_rows": [],
                    "row_count":  0,
                    "truncated":  False,
                }

        truncated = len(result_df) > 500
        result_df = result_df.head(500)

        # Replace NaN/Inf with None for JSON safety
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        result_df = result_df.where(pd.notnull(result_df), None)

        result_rows = result_df.to_dict(orient="records")

        result = {
            "status":      "success",
            "sql_query":   sql_query,
            "result_rows": result_rows,
            "row_count":   len(result_rows),
            "truncated":   truncated,
            "error":       None,
        }

        _write_output_json("query_result.json", result)
        logger.info(f"SQL executed — {len(result_rows)} rows returned, truncated={truncated}")
        return result

    except Exception as e:
        logger.exception(f"SQL Execution Exception: {e}")
        return {"status": "failed", "error": "exception", "message": str(e)}


# ── Visualisation Tools ────────────────────────────────────────────────────────

@tool("Select Chart Type")
def select_chart_type(intent_type: str, desired_output: str,
                      target_columns: str) -> Dict[str, Any]:
    """
    Determine the most appropriate Plotly chart type based on intent and columns.
    desired_output and target_columns must be JSON-encoded strings.
    """
    try:
        df = _get_active_df()
        cols = _parse_column_list(target_columns)

        try:
            desired = json.loads(desired_output) if desired_output.strip() else []
        except (json.JSONDecodeError, AttributeError):
            desired = []

        chart_type = _infer_chart_type(intent_type, desired, cols, df)

        result = {
            "status":     "success",
            "chart_type": chart_type,
            "reasoning":  f"intent_type='{intent_type}' with desired_output={desired} → '{chart_type}'",
        }

        logger.info(f"Chart type selected — {chart_type}")
        return result

    except Exception as e:
        logger.exception(f"Chart Selection Exception: {e}")
        return {"status": "failed", "error": "exception", "message": str(e)}


@tool("Build Plotly Chart Specification")
def build_plotly_spec(spec_json: str) -> Dict[str, Any]:
    """
    Build a full Plotly figure specification from a JSON string.
    spec_json must be a JSON-encoded object with keys:
      chart_type (str), result_rows (list of dicts),
      x_column (str), y_column (str), title (str).
    template is optional and defaults to plotly_white.
    Example: {"chart_type":"bar","result_rows":[...],"x_column":"region","y_column":"sales","title":"Sales by Region"}
    """
    try:
        try:
            spec = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
        except (json.JSONDecodeError, TypeError):
            return {
                "status":          "failed",
                "error":           "invalid_spec_json",
                "message":         "spec_json could not be parsed as JSON.",
                "chart_generated": False,
                "render_note":     "spec_json was not valid JSON.",
            }

        chart_type = spec.get("chart_type", "bar")
        x_column   = spec.get("x_column", "")
        y_column   = spec.get("y_column", "")
        title      = spec.get("title", "")
        template   = spec.get("template", "plotly_white")
        rows_raw   = spec.get("result_rows", [])

        try:
            rows: List[Dict] = json.loads(rows_raw) if isinstance(rows_raw, str) else rows_raw
        except (json.JSONDecodeError, TypeError):
            rows = []

        if not rows:
            return {
                "status":          "success",
                "chart_generated": False,
                "render_note":     "No data rows — chart not generated.",
                "plotly_spec":     None,
                "chart_type":      chart_type,
                "chart_title":     title,
                "axes":            {"x_label": x_column, "y_label": y_column},
            }

        result_df = pd.DataFrame(rows)

        # Validate columns exist in result
        if x_column not in result_df.columns:
            x_column = result_df.columns[0]
        if y_column not in result_df.columns and len(result_df.columns) > 1:
            y_column = result_df.columns[1]
        elif y_column not in result_df.columns:
            y_column = x_column

        # Smart clean hierarchical category strings (e.g. "Root|Sub|Leaf" -> "Root > Leaf")
        def _clean_category_label(val):
            if isinstance(val, str) and "|" in val:
                parts = [p.strip() for p in val.split("|") if p.strip()]
                if len(parts) >= 2:
                    return f"{parts[0]} > {parts[-1]}"
                elif parts:
                    return parts[0]
            return val

        if x_column in result_df.columns:
            result_df[x_column] = result_df[x_column].apply(_clean_category_label)
            # If duplicates occur after label cleaning, group and aggregate
            if y_column in result_df.columns and y_column != x_column:
                try:
                    result_df[y_column] = pd.to_numeric(result_df[y_column], errors="coerce")
                    result_df = result_df.groupby(x_column, as_index=False).agg({y_column: "sum"})
                    result_df = result_df.sort_values(by=y_column, ascending=False)
                except Exception:
                    pass

        x_label = _humanise_column(x_column)
        y_label = _humanise_column(y_column)

        # Build Plotly figure
        if chart_type == "bar":
            # Cap bar charts at 15 items for readability and bundle rest into "Other"
            if result_df[x_column].nunique() > 15:
                top_n = result_df.head(15)
                other_sum = result_df.iloc[15:][y_column].sum()
                other_row = pd.DataFrame([{x_column: "Other", y_column: other_sum}])
                result_df = pd.concat([top_n, other_row], ignore_index=True)
            
            fig = go.Figure(
                go.Bar(x=result_df[x_column], y=result_df[y_column],
                       marker_color="#4f86c6")
            )
        elif chart_type == "scatter":
            trace = go.Scatter(
                x=result_df[x_column], y=result_df[y_column],
                mode="markers", marker=dict(color="#4f86c6", size=7, opacity=0.75)
            )
            fig = go.Figure(trace)
            # Add OLS trendline if both columns are numeric
            try:
                x_num = pd.to_numeric(result_df[x_column], errors="coerce").dropna()
                y_num = pd.to_numeric(result_df[y_column], errors="coerce").dropna()
                if len(x_num) > 2:
                    slope, intercept, _, _, _ = stats.linregress(x_num, y_num)
                    x_range = [float(x_num.min()), float(x_num.max())]
                    y_range = [slope * x + intercept for x in x_range]
                    fig.add_trace(go.Scatter(
                        x=x_range, y=y_range, mode="lines",
                        line=dict(color="#e07b39", dash="dash", width=1.5),
                        name="Trend"
                    ))
            except Exception:
                pass

        elif chart_type == "histogram":
            fig = go.Figure(
                go.Histogram(x=result_df[x_column], marker_color="#4f86c6",
                             nbinsx=30, opacity=0.85)
            )
            y_label = "Count"

        elif chart_type == "line":
            fig = go.Figure(
                go.Scatter(x=result_df[x_column], y=result_df[y_column],
                           mode="lines+markers", line=dict(color="#4f86c6", width=2))
            )

        elif chart_type == "box":
            fig = go.Figure(
                go.Box(y=result_df[y_column], name=y_label,
                       marker_color="#4f86c6", boxmean=True)
            )

        elif chart_type == "pie":
            # Cap pie slices at 7 items for readability and bundle rest into "Other"
            if result_df[x_column].nunique() > 7:
                top_n = result_df.head(7)
                other_sum = result_df.iloc[7:][y_column].sum()
                other_row = pd.DataFrame([{x_column: "Other", y_column: other_sum}])
                result_df = pd.concat([top_n, other_row], ignore_index=True)
            
            fig = go.Figure(
                go.Pie(labels=result_df[x_column], values=result_df[y_column],
                       hole=0.3)
            )

        else:
            # Fallback to bar
            chart_type = "bar"
            if result_df[x_column].nunique() > 15:
                top_n = result_df.head(15)
                other_sum = result_df.iloc[15:][y_column].sum()
                other_row = pd.DataFrame([{x_column: "Other", y_column: other_sum}])
                result_df = pd.concat([top_n, other_row], ignore_index=True)
            
            fig = go.Figure(
                go.Bar(x=result_df[x_column], y=result_df[y_column],
                       marker_color="#4f86c6")
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=template,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=chart_type == "scatter",
        )

        plotly_spec = json.loads(fig.to_json())

        result = {
            "status":          "success",
            "chart_type":      chart_type,
            "plotly_spec":     plotly_spec,
            "chart_title":     title,
            "axes":            {"x_label": x_label, "y_label": y_label},
            "chart_generated": True,
            "render_note":     None,
        }

        _write_output_json("chart.json", result)
        logger.info(f"Chart spec built — type={chart_type}, rows={len(rows)}")
        return result

    except Exception as e:
        logger.exception(f"Chart Build Exception: {e}")
        return {"status": "failed", "error": "exception",
                "message": str(e), "chart_generated": False}


# ── Insight Tools ──────────────────────────────────────────────────────────────

_JARGON_TERMS = [
    "dataframe", "sql", "query", "aggregation", "coefficient",
    "dtype", "pandas", "numpy", "duckdb", "boolean", "null",
    "schema", "index", "correlation coefficient", "p-value",
]


def _strip_jargon(text: str) -> bool:
    """Return True if the text contains technical jargon."""
    lower = text.lower()
    return any(term in lower for term in _JARGON_TERMS)


@tool("Compose Business Insight")
def compose_insight(user_query: str, eda_result: str,
                    query_result: str, chart_meta: str) -> Dict[str, Any]:
    """
    Compose a plain-English 3-sentence business insight from analytical results.
    eda_result, query_result, and chart_meta must be JSON-encoded strings.
    """
    try:
        # Parse inputs
        try:
            eda  = json.loads(eda_result)  if isinstance(eda_result,  str) else (eda_result  or {})
            qr   = json.loads(query_result) if isinstance(query_result, str) else (query_result or {})
            chart = json.loads(chart_meta)  if isinstance(chart_meta,  str) else (chart_meta  or {})
        except (json.JSONDecodeError, TypeError):
            eda, qr, chart = {}, {}, {}

        rows    = qr.get("result_rows", [])
        obs     = eda.get("observations", [])
        summary = eda.get("summary_stats", {})
        row_count = qr.get("row_count", 0)

        if not rows and not obs:
            return {
                "status":               "success",
                "insight_text":         "The analysis did not return any results for your question. "
                                        "This may be because no data matched your filters. "
                                        "Try broadening your search or checking if the column names are correct.",
                "key_metric":           "No data found",
                "follow_up_suggestions": [
                    "Can you show me what values exist in this column?",
                    "What does the overall dataset look like?",
                ],
                "error": None,
            }

        # Derive key metric from top query row
        key_metric = "See results"
        if rows:
            first_row = rows[0]
            values = list(first_row.values())
            if len(values) >= 2:
                label = str(values[0])
                value = values[1]
                try:
                    value = f"{float(value):,.2f}"
                except (TypeError, ValueError):
                    value = str(value)
                key_metric = f"{label}: {value}"
            elif values:
                key_metric = str(values[0])

        # Build 3 sentences
        sentences = []

        # Sentence 1 — most important finding
        if rows and len(rows) > 0:
            first_row  = rows[0]
            keys = list(first_row.keys())
            if len(keys) >= 2:
                dim_col, metric_col = keys[0], keys[1]
                top_val = first_row.get(dim_col, "")
                top_metric = first_row.get(metric_col, "")
                try:
                    top_metric = f"{float(top_metric):,.2f}"
                except (TypeError, ValueError):
                    top_metric = str(top_metric)
                sentences.append(
                    f"The highest {_humanise_column(metric_col)} is {top_metric}, "
                    f"recorded for {top_val}."
                )
            else:
                sentences.append(obs[0] if obs else f"Your query returned {row_count} results.")
        elif obs:
            sentences.append(obs[0])

        # Sentence 2 — context or comparison
        if len(rows) >= 2:
            second_row = rows[1]
            keys = list(second_row.keys())
            if len(keys) >= 2:
                dim_col, metric_col = keys[0], keys[1]
                second_val = second_row.get(dim_col, "")
                second_metric = second_row.get(metric_col, "")
                try:
                    second_metric = f"{float(second_metric):,.2f}"
                except (TypeError, ValueError):
                    second_metric = str(second_metric)
                sentences.append(
                    f"The next highest is {second_val} at {second_metric}, "
                    f"which accounts for the second largest share in this group."
                )
        elif obs and len(obs) > 1:
            sentences.append(obs[1])
        elif summary:
            # Fallback: use summary stat observation
            col = list(summary.keys())[0]
            s = summary[col]
            if isinstance(s, dict) and "mean" in s:
                sentences.append(
                    f"Overall, {_humanise_column(col)} averages {s['mean']:,} "
                    f"across the dataset."
                )

        # Sentence 3 — actionable observation
        actionable = ""
        null_issues = {
            col: pct for col, pct in eda.get("null_report", {}).items()
            if pct > 20
        }
        outlier_issues = [
            f for f in eda.get("outlier_flags", [])
            if isinstance(f, dict) and f.get("outlier_count", 0) > 0
        ]
        if null_issues:
            col = list(null_issues.keys())[0]
            pct = null_issues[col]
            actionable = (
                f"Note that {_humanise_column(col)} has {pct}% missing values — "
                f"consider reviewing data collection for this field before drawing firm conclusions."
            )
        elif outlier_issues:
            oi = outlier_issues[0]
            actionable = (
                f"{_humanise_column(oi['column'])} contains {oi['outlier_count']} unusual values "
                f"that fall outside the expected range — these may be worth investigating further."
            )
        elif len(rows) > 5:
            actionable = (
                f"With {row_count} data points in total, there is enough information here "
                f"to explore further breakdowns by other dimensions in your dataset."
            )
        else:
            actionable = (
                "Consider filtering the data further or adding more records "
                "to get a more reliable view of this trend."
            )
        sentences.append(actionable)

        insight_text = " ".join(sentences[:3])

        # Jargon check
        if _strip_jargon(insight_text):
            logger.warning("Jargon detected in insight — applying cleanup")
            for term in _JARGON_TERMS:
                insight_text = insight_text.replace(term, "")

        # Follow-up suggestions
        follow_ups = [
            f"Can you break this down by a different category?",
            f"Are there any trends over time in this data?",
        ]
        if len(rows) > 1:
            keys = list(rows[0].keys())
            if keys:
                follow_ups[0] = f"Which {_humanise_column(keys[0])} has the lowest value?"
        if obs:
            follow_ups[1] = "What does the full distribution of this data look like?"

        result = {
            "status":                "success",
            "insight_text":          insight_text,
            "key_metric":            key_metric,
            "follow_up_suggestions": follow_ups[:2],
            "error":                 None,
        }

        _write_output_json("insight.json", result)
        logger.info("Insight composed successfully")
        return result

    except Exception as e:
        logger.exception(f"Insight Composition Exception: {e}")
        return {"status": "failed", "error": "exception", "message": str(e)}


@tool("Run Hypothesis Test")
def run_hypothesis_test(target_columns: str) -> Dict[str, Any]:
    """
    Run an automated statistical hypothesis test (correlation, t-test, ANOVA, chi-square)
    on the specified target columns in the dataset.
    Pass target_columns as a JSON array string or comma-separated column names.
    """
    try:
        df = _get_active_df()
        if df is None or df.empty:
            res = {
                "status": "failed",
                "error": "no_dataset",
                "message": "No dataset registered.",
            }
            _write_output_json("hypothesis_test.json", res)
            return res
        
        cols = _parse_column_list(target_columns)
        found_cols, missing_cols = _resolve_columns(df, cols)
        
        if len(found_cols) < 2:
            # Try to auto-detect numeric columns or categorical columns to run a test on
            numeric = _get_numeric_columns(df)
            categorical = _get_categorical_columns(df)
            if len(numeric) >= 2:
                found_cols = numeric[:2]
            elif len(numeric) >= 1 and len(categorical) >= 1:
                found_cols = [categorical[0], numeric[0]]
            elif len(categorical) >= 2:
                found_cols = categorical[:2]
            else:
                res = {
                    "status": "failed",
                    "error": "insufficient_columns",
                    "message": "At least 2 valid columns are required to run a statistical test.",
                }
                _write_output_json("hypothesis_test.json", res)
                return res
        
        col1, col2 = found_cols[0], found_cols[1]
        
        # Smart coercion: if a column is object type but mostly numeric digits (with some empty/question marks)
        # coerce it to numeric so statistical tests work correctly.
        df_working = df.copy()
        for col in [col1, col2]:
            if df_working[col].dtype == "object":
                # Convert to numeric, turn errors to NaN
                coerced = pd.to_numeric(df_working[col], errors='coerce')
                # If at least 40% of the non-null entries were successfully converted, use it
                if coerced.notna().sum() > (df_working[col].notna().sum() * 0.4):
                    df_working[col] = coerced
                    logger.info(f"Coerced object column '{col}' to numeric for statistical analysis.")
        
        # Determine data types
        is_num1 = pd.api.types.is_numeric_dtype(df_working[col1])
        is_num2 = pd.api.types.is_numeric_dtype(df_working[col2])
        
        cleaned_df = df_working[[col1, col2]].dropna()
        if len(cleaned_df) < 10:
            res = {
                "status": "failed",
                "error": "insufficient_data",
                "message": f"Insufficient data (only {len(cleaned_df)} non-null rows) to perform a statistical test.",
            }
            _write_output_json("hypothesis_test.json", res)
            return res
            
        test_name = ""
        statistic_name = ""
        statistic_val = 0.0
        p_val = 1.0
        h0 = ""
        h1 = ""
        interpretation = ""
        sig_threshold = 0.05
        is_significant = False
        
        # Scenario 1: Numeric vs Numeric -> Correlation Test
        if is_num1 and is_num2:
            stat, pval = stats.pearsonr(cleaned_df[col1], cleaned_df[col2])
            test_name = "Pearson Correlation Coefficient Test"
            statistic_name = "Correlation Coefficient (r)"
            statistic_val = float(stat)
            p_val = float(pval)
            h0 = f"There is no linear correlation between {_humanise_column(col1)} and {_humanise_column(col2)} (correlation = 0)."
            h1 = f"There is a significant linear correlation between {_humanise_column(col1)} and {_humanise_column(col2)} (correlation != 0)."
            is_significant = p_val < sig_threshold
            
            strength = "weak"
            abs_stat = abs(statistic_val)
            if abs_stat > 0.7:
                strength = "strong"
            elif abs_stat > 0.4:
                strength = "moderate"
                
            direction = "positive" if statistic_val > 0 else "negative"
            
            if is_significant:
                interpretation = (
                    f"We reject the null hypothesis. There is strong statistical evidence "
                    f"(p-value = {p_val:.2e}) of a {strength} {direction} correlation between "
                    f"{_humanise_column(col1)} and {_humanise_column(col2)}."
                )
            else:
                interpretation = (
                    f"We fail to reject the null hypothesis. There is no statistically significant correlation "
                    f"(p-value = {p_val:.3f}) between {_humanise_column(col1)} and {_humanise_column(col2)}."
                )

        # Scenario 2: One Numeric, One Categorical -> Group Differences (T-test or ANOVA)
        elif (is_num1 and not is_num2) or (not is_num1 and is_num2):
            num_col = col1 if is_num1 else col2
            cat_col = col2 if is_num1 else col1
            
            unique_cats = cleaned_df[cat_col].unique()
            groups = [cleaned_df[cleaned_df[cat_col] == cat][num_col].values for cat in unique_cats]
            groups = [g for g in groups if len(g) >= 3]
            
            if len(groups) < 2:
                res = {
                    "status": "failed",
                    "error": "insufficient_groups",
                    "message": f"Column '{cat_col}' does not have enough distinct categories with sufficient data.",
                }
                _write_output_json("hypothesis_test.json", res)
                return res
            
            if len(groups) == 2:
                stat, pval = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                test_name = "Welch's Two-Sample T-Test"
                statistic_name = "t-statistic"
                statistic_val = float(stat)
                p_val = float(pval)
                h0 = f"The mean of {_humanise_column(num_col)} is equal across both groups of {_humanise_column(cat_col)}."
                h1 = f"The mean of {_humanise_column(num_col)} is significantly different between the two groups of {_humanise_column(cat_col)}."
                is_significant = p_val < sig_threshold
                
                if is_significant:
                    interpretation = (
                        f"We reject the null hypothesis. There is a statistically significant difference "
                        f"(p-value = {p_val:.2e}) in the average {_humanise_column(num_col)} across the groups."
                    )
                else:
                    interpretation = (
                        f"We fail to reject the null hypothesis. There is no statistically significant difference "
                        f"(p-value = {p_val:.3f}) in the average {_humanise_column(num_col)} across the groups."
                    )
            else:
                stat, pval = stats.f_oneway(*groups)
                test_name = "One-Way ANOVA (Analysis of Variance)"
                statistic_name = "F-statistic"
                statistic_val = float(stat)
                p_val = float(pval)
                h0 = f"The average of {_humanise_column(num_col)} is the same across all categories of {_humanise_column(cat_col)}."
                h1 = f"At least one category of {_humanise_column(cat_col)} has a significantly different average {_humanise_column(num_col)}."
                is_significant = p_val < sig_threshold
                
                if is_significant:
                    interpretation = (
                        f"We reject the null hypothesis. At least one category of {_humanise_column(cat_col)} "
                        f"shows a statistically significant difference (p-value = {p_val:.2e}) in average {_humanise_column(num_col)}."
                    )
                else:
                    interpretation = (
                        f"We fail to reject the null hypothesis. No statistically significant differences "
                        f"(p-value = {p_val:.3f}) were found in average {_humanise_column(num_col)} across the categories."
                    )

        # Scenario 3: Categorical vs Categorical -> Chi-Square Test
        else:
            contingency_table = pd.crosstab(cleaned_df[col1], cleaned_df[col2])
            if contingency_table.size < 4:
                res = {
                    "status": "failed",
                    "error": "small_contingency_table",
                    "message": "Categorical combinations must form at least a 2x2 grid.",
                }
                _write_output_json("hypothesis_test.json", res)
                return res
            stat, pval, dof, expected = stats.chi2_contingency(contingency_table)
            test_name = "Chi-Square Test of Independence"
            statistic_name = "Chi-Square statistic"
            statistic_val = float(stat)
            p_val = float(pval)
            h0 = f"{_humanise_column(col1)} and {_humanise_column(col2)} are independent of each other."
            h1 = f"There is a significant association/dependence between {_humanise_column(col1)} and {_humanise_column(col2)}."
            is_significant = p_val < sig_threshold
            
            if is_significant:
                interpretation = (
                    f"We reject the null hypothesis. There is strong statistical evidence "
                    f"(p-value = {p_val:.2e}) of a significant association or dependency between "
                    f"{_humanise_column(col1)} and {_humanise_column(col2)}."
                )
            else:
                interpretation = (
                    f"We fail to reject the null hypothesis. There is no statistically significant association "
                    f"(p-value = {p_val:.3f}) between {_humanise_column(col1)} and {_humanise_column(col2)}."
                )
        
        result = {
            "status": "success",
            "test_name": test_name,
            "statistic_name": statistic_name,
            "statistic_value": round(statistic_val, 4),
            "p_value": p_val,
            "null_hypothesis": h0,
            "alternative_hypothesis": h1,
            "interpretation": interpretation,
            "is_significant": is_significant,
            "columns": [col1, col2],
            "error": None
        }
        
        _write_output_json("hypothesis_test.json", result)
        logger.info(f"Hypothesis test completed: {test_name} — p-value={p_val}")
        return result
        
    except Exception as e:
        logger.exception(f"Hypothesis Test Exception: {e}")
        err_res = {"status": "failed", "error": "exception", "message": str(e)}
        _write_output_json("hypothesis_test.json", err_res)
        return err_res