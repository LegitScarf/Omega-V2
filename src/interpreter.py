import io
import sys
import traceback
import contextlib
import pandas as pd
import numpy as np
import scipy
try:
    import statsmodels
except ImportError:
    statsmodels = None
import plotly
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any

def execute_code(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes the given Python code in a sandboxed local environment.
    Exposes the dataframe `df` as a global variable.
    Captures stdout, stderr, and any exceptions raised.
    """
    from .utils import get_output_path
    from .predictive import (
        fit_regression_model,
        fit_classification_model,
        fit_kmeans_clustering,
        forecast_time_series
    )
    
    def serialize_safe(obj):
        if isinstance(obj, dict):
            return {str(k): serialize_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [serialize_safe(x) for x in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return serialize_safe(obj.tolist())
        elif isinstance(obj, pd.Index):
            return serialize_safe(obj.tolist())
        elif isinstance(obj, (pd.Interval, pd.Timestamp)):
            return str(obj)
        try:
            import json
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

    def write_output_json(filename: str, data: Dict[str, Any]) -> None:
        import json
        safe_data = serialize_safe(data)
        with open(get_output_path(filename), "w", encoding="utf-8") as f:
            json.dump(safe_data, f, indent=2)

    # Prepare global execution context
    exec_globals = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "scipy": scipy,
        "statsmodels": statsmodels,
        "plotly": plotly,
        "go": go,
        "px": px,
        "get_output_path": get_output_path,
        "write_output_json": write_output_json,
        "fit_regression_model": fit_regression_model,
        "fit_classification_model": fit_classification_model,
        "fit_kmeans_clustering": fit_kmeans_clustering,
        "forecast_time_series": forecast_time_series,
    }
    exec_locals = {}

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    success = True
    error_message = None

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            # Compile first to catch syntax errors cleanly
            compiled_code = compile(code, "<sandbox>", "exec")
            exec(compiled_code, exec_globals, exec_locals)
    except Exception as e:
        success = False
        tb = traceback.format_exc()
        error_message = f"{str(e)}\n\nTraceback:\n{tb}"

    stdout_val = stdout_buffer.getvalue()
    stderr_val = stderr_buffer.getvalue()

    return {
        "success": success,
        "stdout": stdout_val,
        "stderr": stderr_val,
        "error": error_message,
        "locals": exec_locals,
    }
