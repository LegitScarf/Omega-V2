import os
import json
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional
from .utils import get_output_path

logger = logging.getLogger("Omega.Predictive")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


def _write_output_json(filename: str, payload: Dict[str, Any]) -> None:
    output_path = get_output_path(filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def forecast_time_series(
    df: pd.DataFrame,
    time_col: str,
    metric_col: str,
    horizon: int = 6
) -> Dict[str, Any]:
    """
    Fit a seasonal linear regression trend on metric_col grouped by time_col,
    and forecast the next `horizon` periods with 95% confidence intervals.
    """
    try:
        if df is None or df.empty:
            res = {"status": "failed", "error": "no_dataset", "message": "No dataset registered."}
            _write_output_json("prediction.json", res)
            return res

        # Ensure columns exist
        if time_col not in df.columns or metric_col not in df.columns:
            res = {
                "status": "failed",
                "error": "missing_columns",
                "message": f"Time column '{time_col}' or metric column '{metric_col}' not found in dataset."
            }
            _write_output_json("prediction.json", res)
            return res

        # Prepare dataset: convert and drop NaNs
        working_df = df[[time_col, metric_col]].copy()
        working_df[metric_col] = pd.to_numeric(working_df[metric_col], errors='coerce')
        working_df = working_df.dropna()

        if len(working_df) < 5:
            res = {
                "status": "failed",
                "error": "insufficient_data",
                "message": f"Not enough numeric data points (only {len(working_df)}) to compute a forecast."
            }
            _write_output_json("prediction.json", res)
            return res

        # Try to parse datetime columns robustly
        is_datetime = False
        if pd.api.types.is_datetime64_any_dtype(working_df[time_col]):
            is_datetime = True
        else:
            try:
                parsed = pd.to_datetime(working_df[time_col], errors='coerce')
                # If at least half of the values are valid datetimes, treat it as a datetime column
                if parsed.notna().sum() > 0.5 * len(working_df):
                    working_df[time_col] = parsed
                    is_datetime = True
            except Exception:
                is_datetime = False

        if is_datetime:
            working_df = working_df.dropna(subset=[time_col])
            # Sort chronologically and aggregate duplicates
            aggregated = working_df.groupby(time_col)[metric_col].mean().sort_index().reset_index()
            dates = aggregated[time_col]
            values = aggregated[metric_col].values
            
            # Detect frequency to generate future dates
            if len(dates) >= 2:
                diffs = dates.diff().dropna()
                median_diff = diffs.median()
                
                # Check frequency based on median time delta
                if median_diff.days >= 360:
                    freq = 'YS'
                elif median_diff.days >= 28:
                    freq = 'MS'
                elif median_diff.days >= 7:
                    freq = 'W'
                else:
                    freq = 'D'
            else:
                freq = 'D'
                
            future_dates = pd.date_range(start=dates.max(), periods=horizon + 1, freq=freq)[1:]
            hist_dates_str = dates.dt.strftime('%Y-%m-%d').tolist()
            future_dates_str = future_dates.strftime('%Y-%m-%d').tolist()
        else:
            # Fallback for numerical/ordinal time column (e.g. Years like 2006, 2007)
            # Try to convert to float/int
            working_df[time_col] = pd.to_numeric(working_df[time_col], errors='coerce')
            working_df = working_df.dropna(subset=[time_col])
            
            if len(working_df) < 5:
                res = {
                    "status": "failed",
                    "error": "insufficient_data",
                    "message": "Time column contains non-numeric strings and is not parseable as a date."
                }
                _write_output_json("prediction.json", res)
                return res
                
            aggregated = working_df.groupby(time_col)[metric_col].mean().sort_index().reset_index()
            time_vals = aggregated[time_col].values
            values = aggregated[metric_col].values
            
            if len(time_vals) >= 2:
                step = np.median(np.diff(time_vals))
            else:
                step = 1
                
            future_times = [float(time_vals[-1] + (i * step)) for i in range(1, horizon + 1)]
            
            hist_dates_str = [str(x) for x in time_vals]
            future_dates_str = [str(x) for x in future_times]

        n = len(values)
        x = np.arange(n)
        
        # Fit linear trend model
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Dynamic seasonality detection based on date frequency
        seasonality = np.zeros(n)
        has_seasonality = False
        period = 0
        
        if is_datetime and len(dates) >= 2:
            if freq == 'MS': # Monthly data
                period = 12
            elif freq == 'W': # Weekly data
                period = 52
            elif freq == 'D': # Daily data
                period = 7
        
        # Fallback to defaults if not datetime or dataset too small
        if period == 0 or n < period * 2:
            if n >= 24:
                period = 12
            elif n >= 8:
                period = 4
                
        if period > 0 and n >= period * 2:
            has_seasonality = True
            
        seasonal_factors = {}
        if has_seasonality:
            # De-trend historical values to extract seasonal components
            trend = intercept + slope * x
            residuals = values - trend
            # Calculate average residual per seasonal index
            for i in range(period):
                seasonal_factors[i] = np.mean(residuals[i::period])
            # Apply adjustments to trend
            for i in range(n):
                seasonality[i] = seasonal_factors[i % period]
        
        # Historical fitted values
        fitted = intercept + slope * x + seasonality
        residuals = values - fitted
        std_dev = np.std(residuals) if len(residuals) > 1 else 1.0
        
        # Forecast future points
        forecast_x = np.arange(n, n + horizon)
        forecast_trend = intercept + slope * forecast_x
        
        forecast_seasonal = np.zeros(horizon)
        if has_seasonality:
            for idx, i in enumerate(forecast_x):
                forecast_seasonal[idx] = seasonal_factors[i % period]
                
        forecast_values = forecast_trend + forecast_seasonal
        
        # Standard error of prediction for CI bounds: std_dev * sqrt(1 + 1/n + (x_f - x_mean)^2 / sum((x_i - x_mean)^2))
        x_mean = np.mean(x)
        x_var_sum = np.sum((x - x_mean)**2) if n > 1 else 1.0
        
        lower_bound = []
        upper_bound = []
        t_crit = 1.96 # 95% confidence multiplier
        
        for x_f in forecast_x:
            se_pred = std_dev * np.sqrt(1 + (1 / n) + ((x_f - x_mean)**2 / x_var_sum))
            margin = t_crit * se_pred
            lower_bound.append(max(0.0, float(forecast_values[x_f - n] - margin)))
            upper_bound.append(float(forecast_values[x_f - n] + margin))

        result = {
            "status": "success",
            "time_column": time_col,
            "metric_column": metric_col,
            "historical_dates": hist_dates_str,
            "historical_values": [float(v) for v in values],
            "forecast_dates": future_dates_str,
            "forecast_values": [float(v) for v in forecast_values],
            "lower_bound": [float(v) for v in lower_bound],
            "upper_bound": [float(v) for v in upper_bound],
            "model_metrics": {
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "slope": float(slope),
                "std_err": float(std_dev)
            },
            "error": None
        }
        
        _write_output_json("prediction.json", result)
        logger.info(f"Time-series forecast completed successfully for {metric_col} over {time_col}.")
        return result

    except Exception as e:
        logger.exception(f"Forecast Exception: {e}")
        err_res = {"status": "failed", "error": "exception", "message": str(e)}
        _write_output_json("prediction.json", err_res)
        return err_res


def fit_regression_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Fits a Multiple Linear Regression model supporting both numeric and categorical features.
    For categorical features, runs one-hot encoding dynamically and returns category metadata.
    """
    try:
        if df is None or df.empty:
            res = {"status": "failed", "error": "no_dataset", "message": "No dataset registered."}
            _write_output_json("prediction.json", res)
            return res

        all_cols = [target_col] + feature_cols
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            res = {
                "status": "failed",
                "error": "missing_columns",
                "message": f"Required columns not found in dataset: {missing}"
            }
            _write_output_json("prediction.json", res)
            return res

        # Prepare regression dataframe
        working_df = df[all_cols].copy()
        
        # Coerce target column to numeric
        working_df[target_col] = pd.to_numeric(working_df[target_col], errors='coerce')
        
        # Identify types of features
        numeric_features = []
        categorical_features = []
        features_meta = []
        
        for col in feature_cols:
            is_numeric = False
            if pd.api.types.is_numeric_dtype(working_df[col]):
                is_numeric = True
            else:
                coerced = pd.to_numeric(working_df[col], errors='coerce')
                if coerced.notna().sum() > (working_df[col].notna().sum() * 0.4):
                    working_df[col] = coerced
                    is_numeric = True
            
            if is_numeric:
                working_df[col] = pd.to_numeric(working_df[col], errors='coerce')
                numeric_features.append(col)
            else:
                working_df[col] = working_df[col].astype(str)
                # Keep top 15 categories to prevent massive dummy explosion
                unique_vals = working_df[col].value_counts()
                if len(unique_vals) > 15:
                    top_cats = unique_vals.head(14).index.tolist()
                    working_df[col] = working_df[col].apply(lambda x: x if x in top_cats else "Other")
                categorical_features.append(col)

        working_df = working_df.dropna(subset=[target_col] + numeric_features)
        
        # Downsample large datasets to 50k rows for low-latency fitting
        if len(working_df) > 50000:
            logger.info("Downsampling dataset to 50,000 rows for regression fitting.")
            working_df = working_df.sample(n=50000, random_state=42)
            
        if len(working_df) < 10:
            res = {
                "status": "failed",
                "error": "insufficient_data",
                "message": f"Insufficient numeric rows (only {len(working_df)} non-null rows) to fit a regression model."
            }
            _write_output_json("prediction.json", res)
            return res

        # Collect metadata for UI controls and compute standardization scales
        for col in feature_cols:
            if col in numeric_features:
                series = working_df[col].dropna()
                mean_val = float(series.mean()) if not series.empty else 0.0
                std_val = float(series.std()) if not series.empty and series.std() > 0 else 1.0
                features_meta.append({
                    "name": col,
                    "type": "numeric",
                    "mean": mean_val,
                    "std": std_val,
                    "min": float(series.min()) if not series.empty else 0.0,
                    "max": float(series.max()) if not series.empty else 0.0
                })
                # Standardize in-place for training stability
                working_df[col] = (working_df[col] - mean_val) / std_val
            else:
                cats = sorted(working_df[col].unique().tolist())
                features_meta.append({
                    "name": col,
                    "type": "categorical",
                    "categories": cats,
                    "default": cats[0] if cats else ""
                })

        # Process X design matrix with dummy encoding
        X_df = pd.DataFrame(index=working_df.index)
        for col in numeric_features:
            X_df[col] = working_df[col]
            
        dummy_mappings = {}
        for col in categorical_features:
            cats = sorted(working_df[col].unique().tolist())
            dummy_mappings[col] = {}
            # cats[0] is baseline category (dropped to prevent dummy trap)
            for cat in cats[1:]:
                col_name = f"{col}_{cat}"
                X_df[col_name] = (working_df[col] == cat).astype(float)
                dummy_mappings[col][cat] = col_name

        n, k = X_df.shape
        Y = working_df[target_col].values
        X = X_df.values

        # Design matrix (prepend intercept column of 1s)
        A = np.hstack([np.ones((n, 1)), X])

        beta, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)
        intercept = float(beta[0])
        
        coefficients = {}
        for idx, col_name in enumerate(X_df.columns):
            coefficients[col_name] = float(beta[idx + 1])

        # R-squared
        Y_mean = np.mean(Y)
        tss = np.sum((Y - Y_mean)**2)
        rss = np.sum((Y - (A @ beta))**2)
        r_squared = 1.0 - (rss / tss) if tss > 0 else 0.0
        std_err = np.sqrt(rss / (n - k - 1)) if n > k + 1 else 0.0

        result = {
            "status": "regression",
            "target_column": target_col,
            "intercept": intercept,
            "coefficients": coefficients,
            "dummy_mappings": dummy_mappings,
            "features": features_meta,
            "model_metrics": {
                "r_squared": r_squared,
                "std_err": std_err,
                "sample_size": n
            },
            "error": None
        }

        _write_output_json("prediction.json", result)
        logger.info(f"Multiple linear regression fitted successfully for {target_col} based on {feature_cols}.")
        return result

    except Exception as e:
        logger.exception(f"Regression Exception: {e}")
        err_res = {"status": "failed", "error": "exception", "message": str(e)}
        _write_output_json("prediction.json", err_res)
        return err_res


def fit_classification_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Fits a binary logistic regression model supporting both numeric and categorical features.
    Casts target to binary mapping, one-hot encodes categorical predictors, and fits using
    scipy.optimize.minimize.
    """
    try:
        if df is None or df.empty:
            res = {"status": "failed", "error": "no_dataset", "message": "No dataset registered."}
            _write_output_json("prediction.json", res)
            return res

        all_cols = [target_col] + feature_cols
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            res = {
                "status": "failed",
                "error": "missing_columns",
                "message": f"Required columns not found in dataset: {missing}"
            }
            _write_output_json("prediction.json", res)
            return res

        # Prepare classification dataframe
        working_df = df[all_cols].copy()
        
        # Identify types of features
        numeric_features = []
        categorical_features = []
        features_meta = []
        
        for col in feature_cols:
            is_numeric = False
            if pd.api.types.is_numeric_dtype(working_df[col]):
                is_numeric = True
            else:
                coerced = pd.to_numeric(working_df[col], errors='coerce')
                if coerced.notna().sum() > (working_df[col].notna().sum() * 0.4):
                    working_df[col] = coerced
                    is_numeric = True
            
            if is_numeric:
                working_df[col] = pd.to_numeric(working_df[col], errors='coerce')
                numeric_features.append(col)
            else:
                working_df[col] = working_df[col].astype(str)
                # Keep top 15 categories to prevent massive dummy explosion
                unique_vals = working_df[col].value_counts()
                if len(unique_vals) > 15:
                    top_cats = unique_vals.head(14).index.tolist()
                    working_df[col] = working_df[col].apply(lambda x: x if x in top_cats else "Other")
                categorical_features.append(col)

        # Drop missing features and target
        working_df = working_df.dropna(subset=[target_col] + numeric_features)
        
        # Downsample large datasets to 50k rows for low-latency fitting
        if len(working_df) > 50000:
            logger.info("Downsampling dataset to 50,000 rows for classification fitting.")
            working_df = working_df.sample(n=50000, random_state=42)
            
        if len(working_df) < 10:
            res = {
                "status": "failed",
                "error": "insufficient_data",
                "message": f"Insufficient numeric rows (only {len(working_df)} non-null rows) to fit a classification model."
            }
            _write_output_json("prediction.json", res)
            return res

        # Standardize and map target (binary or multi-class)
        # Always convert classification targets to strings to avoid mixed-type sorting issues (e.g. when replacing with 'Other')
        working_df[target_col] = working_df[target_col].astype(str)
        
        unique_targets = sorted(working_df[target_col].unique().tolist())
        is_multiclass = len(unique_targets) > 2
        
        # Limit to top 10 categories for multiclass classification to avoid huge models
        if is_multiclass and len(unique_targets) > 10:
            top_targets = working_df[target_col].value_counts().head(9).index.tolist()
            working_df[target_col] = working_df[target_col].apply(lambda x: x if x in top_targets else "Other")
            unique_targets = sorted(working_df[target_col].unique().tolist())

        # Collect metadata for UI controls and compute standardization scales
        for col in feature_cols:
            if col in numeric_features:
                series = working_df[col].dropna()
                mean_val = float(series.mean()) if not series.empty else 0.0
                std_val = float(series.std()) if not series.empty and series.std() > 0 else 1.0
                features_meta.append({
                    "name": col,
                    "type": "numeric",
                    "mean": mean_val,
                    "std": std_val,
                    "min": float(series.min()) if not series.empty else 0.0,
                    "max": float(series.max()) if not series.empty else 0.0
                })
                # Standardize in-place for training stability
                working_df[col] = (working_df[col] - mean_val) / std_val
            else:
                cats = sorted(working_df[col].unique().tolist())
                features_meta.append({
                    "name": col,
                    "type": "categorical",
                    "categories": cats,
                    "default": cats[0] if cats else ""
                })

        # Process X design matrix with dummy encoding
        X_df = pd.DataFrame(index=working_df.index)
        for col in numeric_features:
            X_df[col] = working_df[col]
            
        dummy_mappings = {}
        for col in categorical_features:
            cats = sorted(working_df[col].unique().tolist())
            dummy_mappings[col] = {}
            for cat in cats[1:]:
                col_name = f"{col}_{cat}"
                X_df[col_name] = (working_df[col] == cat).astype(float)
                dummy_mappings[col][cat] = col_name

        n, k = X_df.shape
        X = X_df.values
        A = np.hstack([np.ones((n, 1)), X])

        # Stable cross entropy with L2 regularization
        def loss_fn(beta, A, Y):
            z = A @ beta
            loss = -Y * z + np.logaddexp(0, z)
            return np.sum(loss) + 0.05 * np.sum(beta[1:]**2)

        def grad_fn(beta, A, Y):
            z = A @ beta
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
            grad = A.T @ (p - Y)
            reg = 0.1 * beta
            reg[0] = 0.0
            return grad + reg

        from scipy.optimize import minimize

        if not is_multiclass:
            # Binary classification
            class_0, class_1 = unique_targets[0], unique_targets[1]
            lower_vals = [str(x).lower().strip() for x in unique_targets]
            pos_indicators = ['true', 'yes', '1', 'y', 't', 'success', 'churn', 'high', 'nintendo']
            if any(val in pos_indicators for val in lower_vals):
                idx_1 = 0
                for i, val in enumerate(lower_vals):
                    if val in pos_indicators:
                        idx_1 = i
                        break
                class_1 = unique_targets[idx_1]
                class_0 = unique_targets[1 - idx_1]
            
            Y = (working_df[target_col] == class_1).astype(float).values
            res = minimize(fun=loss_fn, x0=np.zeros(A.shape[1]), jac=grad_fn, args=(A, Y), method='BFGS')
            beta = res.x
            
            intercept = float(beta[0])
            coefficients = {}
            for idx, col_name in enumerate(X_df.columns):
                coefficients[col_name] = float(beta[idx + 1])

            z = A @ beta
            probs = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
            preds = (probs >= 0.5).astype(float)

            tp = np.sum((Y == 1.0) & (preds == 1.0))
            tn = np.sum((Y == 0.0) & (preds == 0.0))
            fp = np.sum((Y == 0.0) & (preds == 1.0))
            fn = np.sum((Y == 1.0) & (preds == 0.0))

            accuracy = float((tp + tn) / n) if n > 0 else 0.0
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            result = {
                "status": "classification",
                "model_mode": "binary",
                "target_column": target_col,
                "target_label": str(target_col),
                "class_0_label": str(class_0),
                "class_1_label": str(class_1),
                "intercept": intercept,
                "coefficients": coefficients,
                "dummy_mappings": dummy_mappings,
                "features": features_meta,
                "model_metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "sample_size": n
                },
                "error": None
            }
        else:
            # Multi-class OvR Classification
            intercepts = {}
            coefficients = {}
            
            # Predict values for each class
            all_probs = {}
            for target_val in unique_targets:
                Y_c = (working_df[target_col] == target_val).astype(float).values
                res_c = minimize(fun=loss_fn, x0=np.zeros(A.shape[1]), jac=grad_fn, args=(A, Y_c), method='BFGS')
                beta_c = res_c.x
                
                intercepts[str(target_val)] = float(beta_c[0])
                coef_dict = {}
                for idx, col_name in enumerate(X_df.columns):
                    coef_dict[col_name] = float(beta_c[idx + 1])
                coefficients[str(target_val)] = coef_dict
                
                # Calculate raw odds
                z_c = A @ beta_c
                all_probs[target_val] = np.exp(np.clip(z_c, -20.0, 20.0))
            
            # Softmax to get predictions and evaluate metrics
            sum_odds = np.zeros(n)
            for target_val in unique_targets:
                sum_odds += all_probs[target_val]
            
            pred_indices = []
            for i in range(n):
                sample_odds = [all_probs[val][i] for val in unique_targets]
                pred_indices.append(np.argmax(sample_odds))
                
            y_pred = [unique_targets[idx] for idx in pred_indices]
            y_true = working_df[target_col].tolist()
            
            # Compute macro metrics
            correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            accuracy = float(correct / n) if n > 0 else 0.0
            
            # Macro precision / recall
            precisions = []
            recalls = []
            for val in unique_targets:
                tp = sum(1 for t, p in zip(y_true, y_pred) if t == val and p == val)
                fp = sum(1 for t, p in zip(y_true, y_pred) if t != val and p == val)
                fn = sum(1 for t, p in zip(y_true, y_pred) if t == val and p != val)
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                precisions.append(prec)
                recalls.append(rec)
                
            macro_precision = float(np.mean(precisions))
            macro_recall = float(np.mean(recalls))
            f1_score = float(2 * macro_precision * macro_recall / (macro_precision + macro_recall)) if (macro_precision + macro_recall) > 0 else 0.0

            result = {
                "status": "classification",
                "model_mode": "multiclass",
                "target_column": target_col,
                "classes": [str(x) for x in unique_targets],
                "intercepts": intercepts,
                "coefficients": coefficients,
                "dummy_mappings": dummy_mappings,
                "features": features_meta,
                "model_metrics": {
                    "accuracy": accuracy,
                    "precision": macro_precision,
                    "recall": macro_recall,
                    "f1_score": f1_score,
                    "sample_size": n
                },
                "error": None
            }

        _write_output_json("prediction.json", result)
        logger.info(f"Classification regression fitted successfully for {target_col} based on {feature_cols}.")
        return result

    except Exception as e:
        logger.exception(f"Classification Exception: {e}")
        err_res = {"status": "failed", "error": "exception", "message": str(e)}
        _write_output_json("prediction.json", err_res)
        return err_res


def fit_kmeans_clustering(
    df: pd.DataFrame,
    feature_cols: List[str],
    k: int = 3
) -> Dict[str, Any]:
    """
    Fits a K-Means clustering model. Standardizes numerical data, one-hot encodes
    categorical data, finds centroid clusters in pure NumPy, and runs PCA
    via SVD to project the high-dimensional clusters onto PC1, PC2, PC3 coordinates for 3D visual plotting.
    """
    try:
        if df is None or df.empty:
            res = {"status": "failed", "error": "no_dataset", "message": "No dataset registered."}
            _write_output_json("prediction.json", res)
            return res

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            res = {
                "status": "failed",
                "error": "missing_columns",
                "message": f"Required columns not found in dataset: {missing}"
            }
            _write_output_json("prediction.json", res)
            return res

        k = max(2, min(k, 8))

        working_df = df[feature_cols].copy()
        
        numeric_features = []
        categorical_features = []
        features_meta = []
        
        for col in feature_cols:
            is_numeric = False
            if pd.api.types.is_numeric_dtype(working_df[col]):
                is_numeric = True
            else:
                coerced = pd.to_numeric(working_df[col], errors='coerce')
                if coerced.notna().sum() > (working_df[col].notna().sum() * 0.4):
                    working_df[col] = coerced
                    is_numeric = True
            
            if is_numeric:
                working_df[col] = pd.to_numeric(working_df[col], errors='coerce')
                numeric_features.append(col)
            else:
                working_df[col] = working_df[col].astype(str)
                unique_vals = working_df[col].value_counts()
                if len(unique_vals) > 15:
                    top_cats = unique_vals.head(14).index.tolist()
                    working_df[col] = working_df[col].apply(lambda x: x if x in top_cats else "Other")
                categorical_features.append(col)

        working_df = working_df.dropna(subset=numeric_features)
        
        # Downsample large datasets to 50k rows for low-latency fitting
        if len(working_df) > 50000:
            logger.info("Downsampling dataset to 50,000 rows for clustering fitting.")
            working_df = working_df.sample(n=50000, random_state=42)
            
        if len(working_df) < k * 5:
            res = {
                "status": "failed",
                "error": "insufficient_data",
                "message": f"Insufficient non-null rows (only {len(working_df)} rows) to partition into {k} clusters."
            }
            _write_output_json("prediction.json", res)
            return res

        for col in feature_cols:
            if col in numeric_features:
                series = working_df[col]
                features_meta.append({
                    "name": col,
                    "type": "numeric",
                    "mean": float(series.mean()) if not series.empty else 0.0,
                    "min": float(series.min()) if not series.empty else 0.0,
                    "max": float(series.max()) if not series.empty else 0.0
                })
            else:
                cats = sorted(working_df[col].unique().tolist())
                features_meta.append({
                    "name": col,
                    "type": "categorical",
                    "categories": cats,
                    "default": cats[0] if cats else ""
                })

        X_df = pd.DataFrame(index=working_df.index)
        for col in numeric_features:
            X_df[col] = working_df[col]
            
        dummy_mappings = {}
        for col in categorical_features:
            cats = sorted(working_df[col].unique().tolist())
            dummy_mappings[col] = {}
            for cat in cats[1:]:
                col_name = f"{col}_{cat}"
                X_df[col_name] = (working_df[col] == cat).astype(float)
                dummy_mappings[col][cat] = col_name

        n_samples, n_features = X_df.shape
        
        means = X_df.mean()
        stds = X_df.std().replace(0, 1.0)
        X_std = (X_df - means) / stds
        X_matrix = X_std.values

        np.random.seed(42)
        centroids = np.zeros((k, n_features))
        centroids[0] = X_matrix[np.random.choice(n_samples)]
        
        for c_idx in range(1, k):
            dists = np.min([np.sum((X_matrix - centroids[i])**2, axis=1) for i in range(c_idx)], axis=0)
            probs = dists / (np.sum(dists) + 1e-10)
            centroids[c_idx] = X_matrix[np.random.choice(n_samples, p=probs)]

        labels = np.zeros(n_samples)
        for iteration in range(100):
            dists = np.zeros((n_samples, k))
            for j in range(k):
                dists[:, j] = np.sum((X_matrix - centroids[j])**2, axis=1)
            new_labels = np.argmin(dists, axis=1)
            
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            
            for j in range(k):
                pts = X_matrix[labels == j]
                if len(pts) > 0:
                    centroids[j] = np.mean(pts, axis=0)

        cluster_summaries = []
        for j in range(k):
            indices = np.where(labels == j)[0]
            c_size = len(indices)
            pct = float(c_size / n_samples * 100)
            
            pts_df = X_df.iloc[indices]
            deviations = []
            for col in X_df.columns:
                col_mean = pts_df[col].mean()
                glob_mean = X_df[col].mean()
                glob_std = X_df[col].std()
                if glob_std > 0:
                    z_score = (col_mean - glob_mean) / glob_std
                else:
                    z_score = 0.0
                deviations.append((col, z_score, col_mean))
                
            deviations.sort(key=lambda val: abs(val[1]), reverse=True)
            top_devs = deviations[:2]
            
            characteristics = []
            for col, z, mean_val in top_devs:
                readable_col = col.replace("_", " ").title()
                if z > 0.5:
                    characteristics.append(f"High {readable_col}")
                elif z < -0.5:
                    characteristics.append(f"Low {readable_col}")
            
            char_str = ", ".join(characteristics) if characteristics else "Average profile"
            cluster_summaries.append({
                "cluster_id": j,
                "size": int(c_size),
                "percentage": pct,
                "characteristics": char_str,
                "centroid": [float(val) for val in centroids[j]]
            })

        if n_features == 1:
            pc_coords = np.zeros((n_samples, 3))
            pc_coords[:, 0] = X_matrix[:, 0]
        elif n_features == 2:
            pc_coords = np.zeros((n_samples, 3))
            pc_coords[:, 0] = X_matrix[:, 0]
            pc_coords[:, 1] = X_matrix[:, 1]
        else:
            X_centered = X_matrix - np.mean(X_matrix, axis=0)
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            pc_coords = X_centered @ Vt.T[:, :3]

        wcss = 0.0
        for j in range(k):
            pts = X_matrix[labels == j]
            if len(pts) > 0:
                wcss += np.sum((pts - centroids[j])**2)

        result = {
            "status": "clustering",
            "features": features_meta,
            "feature_names_internal": list(X_df.columns),
            "dummy_mappings": dummy_mappings,
            "means": {col: float(val) for col, val in means.items()},
            "stds": {col: float(val) for col, val in stds.items()},
            "clusters": cluster_summaries,
            "sample_size": n_samples,
            "pc_coords": [[float(val) for val in row] for row in pc_coords],
            "labels": [int(l) for l in labels],
            "model_metrics": {
                "wcss": float(wcss),
                "clusters_count": k
            },
            "error": None
        }

        _write_output_json("prediction.json", result)
        logger.info(f"K-Means clustering completed successfully for {feature_cols} with k={k}.")
        return result

    except Exception as e:
        logger.exception(f"Clustering Exception: {e}")
        err_res = {"status": "failed", "error": "exception", "message": str(e)}
        _write_output_json("prediction.json", err_res)
        return err_res
