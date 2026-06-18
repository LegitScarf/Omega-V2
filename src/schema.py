import logging
import math
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger("Omega.Schema")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


# ── Private helpers ────────────────────────────────────────────────────────────

def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _infer_semantic_type(col: str, series: pd.Series) -> str:
    """
    Infer a human-readable semantic type beyond pandas dtype.
    Helps agents understand what a column represents without seeing the data.

    Returns one of:
        numeric_continuous, numeric_integer, categorical_low_cardinality,
        categorical_high_cardinality, datetime, boolean, text, identifier
    """
    dtype = series.dtype
    col_lower = col.lower()
    n_unique = series.nunique()
    n_total  = len(series.dropna())

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"

    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"

    if pd.api.types.is_float_dtype(dtype):
        return "numeric_continuous"

    if pd.api.types.is_integer_dtype(dtype):
        # Heuristic: low unique count integers are likely categorical codes
        if n_total > 0 and n_unique / n_total < 0.05 and n_unique <= 20:
            return "categorical_low_cardinality"
        return "numeric_integer"

    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        # Try parsing as datetime
        sample = series.dropna().head(10)
        try:
            pd.to_datetime(sample, infer_datetime_format=True)
            return "datetime"
        except Exception:
            pass

        # Identifier heuristic: very high cardinality + id/key/code in name
        id_signals = ("id", "key", "code", "uuid", "ref", "token", "hash")
        if any(sig in col_lower for sig in id_signals) and n_unique > 100:
            return "identifier"

        # Cardinality-based split
        if n_total > 0:
            ratio = n_unique / n_total
            if ratio < 0.10 or n_unique <= 15:
                return "categorical_low_cardinality"
            if n_unique > 100 and ratio > 0.50:
                return "categorical_high_cardinality"

        return "text"

    return "unknown"


def _safe_sample_values(series: pd.Series, n: int = 3) -> List[Any]:
    """
    Return up to n non-null sample values from a series, safely serialised.
    NaN, Inf, and numpy scalars are normalised to Python native types.
    """
    samples = series.dropna().head(n * 3).tolist()
    result = []
    seen = set()
    for val in samples:
        if len(result) >= n:
            break
        try:
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                continue
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = round(float(val), 4)
            elif isinstance(val, str):
                val = val.strip()
                if not val:
                    continue
            key = str(val)
            if key not in seen:
                seen.add(key)
                result.append(val)
        except Exception:
            continue
    return result


def _column_profile(col: str, series: pd.Series) -> Dict[str, Any]:
    """
    Build a compact profile dict for a single column.
    This is the per-column object that goes into the full schema dict
    and is also rendered into the schema string for agent context.
    """
    dtype_str     = str(series.dtype)
    semantic_type = _infer_semantic_type(col, series)
    n_total       = len(series)
    n_null        = int(series.isna().sum())
    null_pct      = round(float(n_null / n_total * 100), 1) if n_total > 0 else 0.0
    n_unique      = int(series.nunique())
    samples       = _safe_sample_values(series, n=3)

    profile: Dict[str, Any] = {
        "dtype":         dtype_str,
        "semantic_type": semantic_type,
        "null_pct":      null_pct,
        "n_unique":      n_unique,
        "sample_values": samples,
    }

    # Add numeric stats if applicable
    if semantic_type in ("numeric_continuous", "numeric_integer"):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if not numeric.empty:
            profile["min"]  = round(_coerce_float(numeric.min()), 4)
            profile["max"]  = round(_coerce_float(numeric.max()), 4)
            profile["mean"] = round(_coerce_float(numeric.mean()), 4)

    # Add top categories for low-cardinality columns
    if semantic_type == "categorical_low_cardinality":
        top = series.value_counts().head(5)
        profile["top_categories"] = {
            str(k): int(v) for k, v in top.items()
        }

    return profile


# ── Public API ─────────────────────────────────────────────────────────────────

def build_schema_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a full schema dictionary from the uploaded dataframe.

    Returns a dict with:
        - row_count: int
        - col_count: int
        - columns: dict of column_name -> column_profile
        - numeric_columns: list of numeric column names
        - categorical_columns: list of categorical column names
        - datetime_columns: list of datetime column names
        - identifier_columns: list of likely identifier column names
    """
    if df is None or df.empty:
        logger.warning("build_schema_dict called with empty dataframe")
        return {
            "row_count":            0,
            "col_count":            0,
            "columns":              {},
            "numeric_columns":      [],
            "categorical_columns":  [],
            "datetime_columns":     [],
            "identifier_columns":   [],
        }

    columns: Dict[str, Any] = {}
    numeric_cols:     List[str] = []
    categorical_cols: List[str] = []
    datetime_cols:    List[str] = []
    identifier_cols:  List[str] = []

    for col in df.columns:
        try:
            profile = _column_profile(col, df[col])
            columns[col] = profile
            stype = profile["semantic_type"]

            if stype in ("numeric_continuous", "numeric_integer"):
                numeric_cols.append(col)
            elif stype in ("categorical_low_cardinality", "categorical_high_cardinality", "text", "boolean"):
                categorical_cols.append(col)
            elif stype == "datetime":
                datetime_cols.append(col)
            elif stype == "identifier":
                identifier_cols.append(col)

        except Exception as e:
            logger.warning(f"Column profiling failed for '{col}': {e}")
            columns[col] = {
                "dtype":         str(df[col].dtype),
                "semantic_type": "unknown",
                "null_pct":      0.0,
                "n_unique":      0,
                "sample_values": [],
            }

    schema = {
        "row_count":           len(df),
        "col_count":           len(df.columns),
        "columns":             columns,
        "numeric_columns":     numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns":    datetime_cols,
        "identifier_columns":  identifier_cols,
    }

    logger.info(
        f"Schema built — {len(df)} rows, {len(df.columns)} cols "
        f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical, "
        f"{len(datetime_cols)} datetime)"
    )
    return schema


def build_schema_string(df: pd.DataFrame) -> str:
    """
    Build a compact, token-efficient schema string for injection into
    agent context via the {schema} placeholder in tasks.yaml.

    Format per column:
        column_name | semantic_type | dtype | null%: X% | unique: N | samples: [a, b, c]

    Numeric columns also get:
        range: [min → max] | mean: X

    Low-cardinality categoricals also get:
        categories: [A, B, C]

    The string is deliberately concise — agents need to know what columns
    exist and what kind of data they hold, not a full statistical profile.
    That's what the EDA agent is for.
    """
    schema_dict = build_schema_dict(df)

    if not schema_dict["columns"]:
        return "Schema: empty dataset — no columns found."

    lines = [
        f"Dataset: {schema_dict['row_count']:,} rows × {schema_dict['col_count']} columns",
        f"Numeric columns:     {schema_dict['numeric_columns']}",
        f"Categorical columns: {schema_dict['categorical_columns']}",
        f"Datetime columns:    {schema_dict['datetime_columns']}",
        "",
        "Column definitions:",
        "─" * 60,
    ]

    for col, profile in schema_dict["columns"].items():
        stype   = profile.get("semantic_type", "unknown")
        dtype   = profile.get("dtype", "unknown")
        null_pct = profile.get("null_pct", 0.0)
        n_unique = profile.get("n_unique", 0)
        samples  = profile.get("sample_values", [])

        # Base line
        line = (
            f"  {col} | {stype} | dtype={dtype} | "
            f"null%={null_pct}% | unique={n_unique} | "
            f"samples={samples}"
        )

        # Append numeric range
        if "min" in profile and "max" in profile:
            line += f" | range=[{profile['min']} → {profile['max']}] | mean={profile['mean']}"

        # Append top categories
        if "top_categories" in profile:
            cats = list(profile["top_categories"].keys())
            line += f" | categories={cats}"

        lines.append(line)

    lines.append("─" * 60)
    schema_str = "\n".join(lines)

    logger.info(f"Schema string built — ~{len(schema_str)} chars")
    return schema_str


def get_column_names(df: pd.DataFrame) -> List[str]:
    """Return all column names from the dataframe."""
    return df.columns.tolist() if df is not None else []


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return only numeric column names."""
    return build_schema_dict(df)["numeric_columns"]


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Return only categorical column names."""
    return build_schema_dict(df)["categorical_columns"]


def get_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Return only datetime column names."""
    return build_schema_dict(df)["datetime_columns"]