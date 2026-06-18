import os
import json
import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from openai import OpenAI

from .tools import (
    compute_eda_stats,
    detect_outliers,
    generate_sql,
    execute_sql,
    select_chart_type,
    build_plotly_spec,
    run_hypothesis_test,
)
from .predictive import forecast_time_series, fit_regression_model, fit_classification_model, fit_kmeans_clustering
from .schema import build_schema_string
from .utils import get_output_path, load_json_file

logger = logging.getLogger("Omega.Crew")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

# ── Intent parser ──────────────────────────────────────────────────────────────
# Classified preprocessing step
_INTENT_SYSTEM_PROMPT = """
You are a data analytics intent classifier. Given a user query and a dataset
    schema, extract the analytical intent as a structured JSON object.

Return ONLY a valid JSON object with exactly these keys:

{
  "intent_type": one of ["descriptive", "aggregation", "comparison",
                          "correlation", "distribution", "trend",
                          "ranking", "filter", "conversational", "forecast", "regression", "classification", "prescriptive", "clustering"],
  "target_columns": list of column names from the schema most relevant
                    to the query (use exact names from schema),
  "filters": dict of {column: value} pairs representing any filtering
             conditions mentioned in the query (empty dict if none),
  "desired_output": list containing one or more of ["table", "bar_chart",
                    "scatter", "histogram", "line_chart", "box_plot", "pie_chart"]
}

Rules:
- Only reference column names that exist in the provided schema.
- If the query is ambiguous, choose the most conservative intent.
- Use "conversational" if the query is an advisory, Q&A, or follow-up question (e.g., asking for advice on missing data, explanation of analytical results, or general data science recommendations). Do NOT use this for business strategy requests, next steps, action plans, or optimization recommendations.
- Use "forecast" if the query explicitly asks to predict, project, forecast, or estimate future values over a temporal (date, time, year) column.
- Use "regression" if the query asks to predict, estimate, or calculate one numerical target column based on one or more independent predictor/feature columns. You MUST extract both the target column and all predictor/feature columns in the query.
- Use "classification" if the query asks to predict, classify, or estimate a categorical/discrete target column (e.g. true/false, high/low, yes/no, specific category) based on one or more independent predictor/feature columns. You MUST extract both the target column and all predictor/feature columns in the query.
- Use "prescriptive" if the query asks for strategies, recommendations, next steps, optimization plans, action items, or business decisions (e.g. "what strategies can we implement...", "how can we enhance...", "what are the next steps...", "action plan to improve...").
- Use "clustering" if the query asks to cluster, segment, group, partition, or categorize records/entries/customers based on attributes, features, or metrics (e.g. "cluster records into 3 groups based on sales and year", "segment games based on platforms").
- When intent_type is "regression" or "classification", target_columns MUST contain the target column as the first element, followed by all predictor/feature columns (e.g., for 'Predict sales based on genre and year', target_columns must be ['sales', 'genre', 'year']). For "clustering", target_columns contains the list of feature columns to be used for finding segments.
- Never return keys outside the four listed above.
- Never return markdown, code fences, or explanatory text — JSON only.
"""

def _parse_intent(user_query: str, schema: str) -> Dict[str, Any]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    user_message = (
        f"User query: {user_query}\n\n"
        f"Dataset schema:\n{schema}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM_PROMPT.strip()},
                {"role": "user",   "content": user_message},
            ],
            temperature=0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)

        # Validate required keys are present
        required = {"intent_type", "target_columns", "filters", "desired_output"}
        if not required.issubset(parsed.keys()):
            raise ValueError(f"Missing keys in intent response: {required - parsed.keys()}")

        logger.info(f"Intent parsed — type={parsed['intent_type']} "
                    f"columns={parsed['target_columns']}")
        return parsed

    except Exception as e:
        logger.warning(f"Intent parsing failed: {e} — falling back to defaults")
        return {
            "intent_type": "descriptive",
            "target_columns": [],
            "filters": {},
            "desired_output": ["table"],
        }


# ── Structured Insight Generation ──────────────────────────────────────────────
_DESCRIPTIVE_INSIGHT_PROMPT = """
You are a senior data analyst. Your job is to translate descriptive statistics and data profiling results into a clear, jargon-free data health and completeness overview for a non-technical user.

You must return ONLY a valid JSON object with exactly these four keys:
- "insight_text": A string containing exactly 3 to 5 sentences summarizing the dataset:
    * Summarize the scale and scope of the dataset (e.g. number of rows, features, time range covered).
    * Outline the general completeness of the records, highlighting any columns with notable missing values (nulls) or data quality flags.
    * Mention if any major outliers or structural irregularities were detected that the user should be aware of.
- "key_metric": A short, clean string representing the overall scale of the data (e.g. "16,598 Game Records", "205 Vehicle Entries").
- "follow_up_suggestions": A list of exactly 2 plain-English follow-up questions that the user might naturally want to ask next about the dataset structure or contents.
- "error": null (or a plain-English error explanation if the results are completely empty or corrupt).

Rules:
1. STRICTLY avoid technical terms like SQL, dataframe, aggregation, correlation coefficient, p-value, etc. Use business-friendly synonyms (e.g. "records" or "entries" instead of "rows", "attributes" or "characteristics" instead of "columns").
2. Do not recommend business or sales strategies (e.g. do not say "consider focusing on marketing strategies..."). This is a structural descriptive profiling task; describe the data as it is.
3. Keep the tone professional, objective, and analytical.
"""

_ANALYTICAL_INSIGHT_PROMPT = """
You are a senior business analyst. Your job is to translate statistical data, queries, and analytical findings into a clear, jargon-free business insight for a non-technical user.

You must return ONLY a valid JSON object with exactly these four keys:
- "insight_text": A string containing exactly 3 to 5 sentences of business insight:
    * State the single most important analytical finding or takeaway directly (e.g. top performers, strong correlations, distinct trends).
    * Provide concrete context or comparison using specific values or proportions from the results.
    * State one concrete, actionable business recommendation based on this finding.
- "key_metric": A short string representing the single most important number or finding (e.g., "Top Genre: Action (3.2M)", "Correlation: Strong Positive").
- "follow_up_suggestions": A list of exactly 2 plain-English, conversational follow-up questions that the user might naturally want to ask next to explore this finding further.
- "error": null (or a plain-English error explanation if the results are completely empty or insufficient to make sense of).

Rules:
1. STRICTLY avoid technical database or programming terms (e.g. do not say "SQL", "dataframe", "aggregation", "query", "DuckDB", "coefficient", "null", "columns", "rows", "pandas", etc.). Use business-friendly synonyms (e.g. "totals" instead of "aggregations", "records" instead of "rows").
2. Only reference numbers and categories present in the provided analytical context. Do not invent any statistics.
3. Keep the tone conversational, professional, and action-oriented.
"""

_CONVERSATIONAL_INSIGHT_PROMPT = """
You are the user's personal data analyst, named Omega.
Your goal is to conversationally, clearly, and directly answer the user's advisory, Q&A, or follow-up question using the provided context of their dataset.

You must return ONLY a valid JSON object with exactly these four keys:
- "insight_text": A highly detailed, thorough string containing a comprehensive, clear, and actionable response explaining your data analysis, observations, and advice (typically 2 to 3 detailed paragraphs with deep analytical coverage).
- "key_metric": A short, clean string representing the advice topic (e.g. "Advice: Impute Nulls", "Data Transformation").
- "follow_up_suggestions": A list of exactly 2 conversational follow-up questions the user might naturally want to ask you next.
- "error": null (or a plain-English error explanation if you cannot answer the question).

Rules:
1. Use professional, analytical language. You may reference standard statistical, analytical, or modeling concepts (like standard errors, correlation coefficients, R-squared, data types, missing values, or regression parameters) where appropriate to make your findings mathematically sound, but present them in a clear, business-friendly context.
2. Ground your advice directly in the provided dataset context. If the context contains descriptive stats (like missing values or outliers), refer to the actual numbers and columns.
3. Be helpful, clean, and direct. If the query discusses system logs, security events, or technical data (e.g., failed logins, attack types, cyber threats), include specific, domain-relevant security observations.
"""

_PRESCRIPTIVE_INSIGHT_PROMPT = """
You are the user's senior business consultant and prescriptive analytics advisor, named Omega.
Your goal is to propose a concrete, actionable, and data-grounded strategy plan to address the user's query.

You must return ONLY a valid JSON object with exactly these seven keys:
- "insight_text": A detailed, thorough string containing a comprehensive, clear strategic analysis summarizing the direction (typically 2 detailed paragraphs outlining the operational context).
- "key_metric": A short topic label (e.g., "Market Entry Plan", "Optimization Framework").
- "strategies": A list of 3 to 5 specific, highly detailed, data-grounded strategies (plain-English sentences describing specific programs or technical controls).
- "priority_matrix": A list of 3 to 5 objects: [{"action": "Action description", "impact": "High" or "Medium" or "Low", "effort": "High" or "Medium" or "Low"}] representing the proposed activities.
- "risks": A list of 2 to 4 potential risks or data quality concerns (e.g., missing records, high residuals, or sample limitations).
- "follow_up_suggestions": A list of exactly 2 plain-English, conversational follow-up questions.
- "error": null (or a plain-English error explanation if you cannot answer the question).

Rules:
1. Ground every strategy and matrix action in the provided dataset schema, stats, and previous results.
2. Use professional, analytical language. You may reference standard statistical, analytical, or modeling concepts (like standard errors, correlation coefficients, R-squared, data types, missing values, or regression parameters) where appropriate to make your findings mathematically sound, but present them in a clear, business-friendly context.
3. You MUST generate at least 3-5 high-fidelity strategy recommendations. Ensure they are concrete, specific actions. If the query covers operational issues or security events (like brute force attacks, failed logins, system performance), customize the matrix actions and strategies to map directly to technical controls (e.g., credential rate limiting, multi-factor authentication, account locking).
"""

def _load_json_safely(filename: str) -> Dict[str, Any]:
    try:
        path = Path(get_output_path(filename))
        if path.exists():
            return load_json_file(path)
    except Exception:
        pass
    return {}

def _generate_insight_via_llm(
    user_query: str,
    intent_type: str,
    eda_data: Dict[str, Any],
    query_data: Dict[str, Any],
    chart_data: Dict[str, Any],
    hypothesis_data: Optional[Dict[str, Any]] = None,
    prediction_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # Load past data profile from disk if available to make Q&A context-aware
    if intent_type in ["conversational", "prescriptive"]:
        if not eda_data or eda_data.get("status") == "skipped":
            eda_data = _load_json_safely("eda_result.json")
        if not query_data or query_data.get("status") == "skipped":
            query_data = _load_json_safely("query_result.json")
        if not prediction_data or prediction_data.get("status") == "skipped":
            prediction_data = _load_json_safely("prediction.json")

    context = f"User Query: {user_query}\n"
    context += f"Intent Type: {intent_type}\n\n"
    
    if eda_data and eda_data.get("status") != "skipped":
        context += "Descriptive Statistics / EDA Summary:\n"
        context += json.dumps({
            "summary_stats": eda_data.get("summary_stats"),
            "observations": eda_data.get("observations"),
            "outlier_flags": eda_data.get("outlier_flags")
        }, indent=2) + "\n\n"
        
    if query_data and query_data.get("status") != "skipped":
        context += "Executed SQL Query:\n"
        context += f"{query_data.get('sql_query')}\n\n"
        context += "Query Results (first 10 rows shown):\n"
        context += json.dumps(query_data.get("result_rows", [])[:10], indent=2) + "\n\n"
        
    if chart_data and chart_data.get("status") != "skipped":
        context += "Visualisation Config:\n"
        context += f"Chart Type: {chart_data.get('chart_type')}\n"
        context += f"Chart Title: {chart_data.get('chart_title')}\n\n"

    if hypothesis_data and hypothesis_data.get("status") != "skipped":
        context += "Statistical Hypothesis Test Results:\n"
        context += json.dumps(hypothesis_data, indent=2) + "\n\n"

    if prediction_data and prediction_data.get("status") != "skipped":
        if prediction_data.get("status") == "regression":
            context += "Fitted Multiple Linear Regression Model:\n"
            context += json.dumps({
                "target_column": prediction_data.get("target_column"),
                "intercept": prediction_data.get("intercept"),
                "coefficients": prediction_data.get("coefficients"),
                "model_metrics": prediction_data.get("model_metrics")
            }, indent=2) + "\n\n"
        else:
            context += "Future Forecast Projections:\n"
            context += json.dumps({
                "time_column": prediction_data.get("time_column"),
                "metric_column": prediction_data.get("metric_column"),
                "forecast_dates": prediction_data.get("forecast_dates"),
                "forecast_values": prediction_data.get("forecast_values"),
                "model_metrics": prediction_data.get("model_metrics")
            }, indent=2) + "\n\n"

    # Select system prompt dynamically based on the intent
    if intent_type == "descriptive":
        system_prompt = _DESCRIPTIVE_INSIGHT_PROMPT.strip()
    elif intent_type == "conversational":
        system_prompt = _CONVERSATIONAL_INSIGHT_PROMPT.strip()
    elif intent_type == "prescriptive":
        system_prompt = _PRESCRIPTIVE_INSIGHT_PROMPT.strip()
    else:
        system_prompt = _ANALYTICAL_INSIGHT_PROMPT.strip()

    try:
        completion_kwargs = {
            "model": "gpt-4o-mini",
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analytical Context:\n{context}"}
            ],
            "temperature": 0.2,
        }
        if intent_type not in ["conversational", "prescriptive"]:
            completion_kwargs["max_tokens"] = 2048

        response = client.chat.completions.create(**completion_kwargs)
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        
        # Ensure fallback defaults are met
        required = {"insight_text", "key_metric", "follow_up_suggestions", "error"}
        for k in required:
            if k not in parsed:
                if k == "follow_up_suggestions":
                    parsed[k] = []
                elif k == "error":
                    parsed[k] = None
                else:
                    parsed[k] = ""
                    
        # Optional prescriptive keys
        if "strategies" not in parsed:
            parsed["strategies"] = []
        if "priority_matrix" not in parsed:
            parsed["priority_matrix"] = []
        if "risks" not in parsed:
            parsed["risks"] = []
                    
        parsed["intent_type"] = intent_type
        return parsed
        
    except Exception as exc:
        logger.exception(f"Failed to generate structured insight: {exc}")
        return {
            "insight_text": "Failed to generate business insights due to an internal error.",
            "key_metric": "Error",
            "follow_up_suggestions": [
                "Would you like to try executing the query again?",
                "Can you check if the dataset has columns matching your query?"
            ],
            "error": str(exc)
        }


# ── Helpers for skipped files & callbacks ──────────────────────────────────────────

def _write_json(filename: str, data: Dict[str, Any]) -> None:
    path = Path(get_output_path(filename))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class MockTaskOutput:
    """Mock class that mimics CrewAI TaskOutput object so progress tracking still works."""
    def __init__(self, name: str):
        self.name = name


# ── Task Routing Constants ───────────────────────────────────────────────────

# Intents that need EDA (descriptive stats pass)
_EDA_INTENTS = {"descriptive"}

# Intents that need a SQL query executed
_QUERY_INTENTS = {
    "aggregation", "comparison", "correlation",
    "distribution", "trend", "ranking", "filter",
}

# Intents that produce a chart
_VIZ_INTENTS = {
    "aggregation", "comparison", "correlation",
    "distribution", "trend", "ranking",
}


# ── Public Entry Point ────────────────────────────────────────────────────────

def run_omega(
    user_query:    str,
    dataframe,
    step_callback: Optional[Callable] = None,
    task_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    High-performance hybrid data analysis pipeline.
    Bypasses high-latency sequential CrewAI agent loops and executes
    deterministic local python analysis tools directly.
    Generates final business insight with a single structured LLM completion.
    """
    # Step 1 — Build schema
    schema = build_schema_string(dataframe)
    logger.info(
        f"Schema built — {len(dataframe.columns)} columns, {len(dataframe)} rows"
    )

    # Step 2 — Intent preprocessing (low-latency gpt-4o-mini completion)
    logger.info("Parsing intent...")
    intent = _parse_intent(user_query, schema)
    intent_type = intent["intent_type"]
    target_columns_str = json.dumps(intent["target_columns"])
    filters_str = json.dumps(intent["filters"])
    desired_output_str = json.dumps(intent["desired_output"])

    logger.info(
        f"Inputs parsed — intent_type={intent_type}, "
        f"target_columns={intent['target_columns']}"
    )

    # ── Task Routing ──────────────────────────────────────────────────────────
    # descriptive  → EDA + insight            (no SQL, no chart)
    # filter       → query + insight          (no EDA, no chart)
    # distribution → EDA + query + viz + insight
    # other queries → query + viz + insight    (no EDA)
    # unknown      → all four (safe fallback)
    
    do_eda   = False
    do_query = False
    do_viz   = False
    
    if intent_type == "descriptive":
        do_eda = True
    elif intent_type in ["conversational", "prescriptive"]:
        # Conversational Q&A / prescriptive strategy queries bypass code execution
        do_eda   = False
        do_query = False
        do_viz   = False
    elif intent_type == "filter":
        do_query = True
    elif intent_type == "distribution":
        do_eda = True
        do_query = True
        do_viz = True
    elif intent_type in _QUERY_INTENTS:
        do_query = True
        do_viz = True
    else:
        # Fallback
        do_eda = True
        do_query = True
        do_viz = True

    # ── Step 1: Descriptive Stats (EDA) ───────────────────────────────────────
    eda_result = {"status": "skipped", "observations": [], "summary_stats": {}, "outlier_flags": []}
    if do_eda:
        logger.info("Executing EDA tools directly...")
        # Invoke compute_eda_stats and detect_outliers via .func to bypass Tool wrapper
        eda_stats = compute_eda_stats.func(target_columns_str)
        outliers = detect_outliers.func(target_columns_str)
        
        # Merge results (compute_eda_stats already writes eda_result.json)
        eda_result = {**eda_stats, "outlier_flags": outliers.get("outlier_flags", [])}
        _write_json("eda_result.json", eda_result)
        
        if task_callback:
            task_callback(MockTaskOutput("run_eda"))
    else:
        # Prevent Streamlit wait timeout on skipped files
        _write_json("eda_result.json", eda_result)

    # ── Step 2: SQL Query Execution ───────────────────────────────────────────
    query_result = {"status": "skipped", "result_rows": [], "row_count": 0}
    if do_query:
        logger.info("Executing SQL tools directly...")
        sql_info = generate_sql.func(
            user_query=user_query,
            dataset_schema=schema,
            intent_type=intent_type,
            target_columns=target_columns_str,
            filters=filters_str
        )
        if sql_info.get("status") == "success":
            query_result = execute_sql.func(sql_info["sql_query"])
        else:
            query_result = {"status": "failed", "error": sql_info.get("error"), "result_rows": [], "row_count": 0}
            _write_json("query_result.json", query_result)
            
        if task_callback:
            task_callback(MockTaskOutput("run_query"))
    else:
        # Prevent Streamlit wait timeout on skipped files
        _write_json("query_result.json", query_result)

    # ── Step 3: Plotly Spec Visualization ─────────────────────────────────────
    chart_result = {"status": "skipped", "plotly_spec": None, "chart_generated": False}
    if do_viz:
        logger.info("Executing Visualization tools directly...")
        chart_type_info = select_chart_type.func(
            intent_type=intent_type,
            desired_output=desired_output_str,
            target_columns=target_columns_str
        )
        if chart_type_info.get("status") == "success":
            spec_payload = {
                "chart_type": chart_type_info["chart_type"],
                "result_rows": query_result.get("result_rows", []),
                "x_column": intent["target_columns"][0] if intent["target_columns"] else "",
                "y_column": intent["target_columns"][1] if len(intent["target_columns"]) > 1 else "",
                "title": f"Chart for: {user_query}"
            }
            # build_plotly_spec already writes chart.json
            chart_result = build_plotly_spec.func(json.dumps(spec_payload))
        else:
            chart_result = {"status": "failed", "chart_generated": False, "plotly_spec": None}
            _write_json("chart.json", chart_result)
            
        if task_callback:
            task_callback(MockTaskOutput("render_chart"))
    else:
        # Prevent Streamlit wait timeout on skipped files
        _write_json("chart.json", chart_result)

    # ── Step 3.5: Statistical Hypothesis Testing ──────────────────────────────
    hypothesis_result = {"status": "skipped"}
    test_keywords = ["test", "hypothesis", "significant", "significance", "difference", "correlation test", "p-value", "confirm", "confirming"]
    if intent_type in ["correlation", "comparison"] or any(kw in user_query.lower() for kw in test_keywords):
        logger.info("Executing Hypothesis Test tool directly...")
        hypothesis_result = run_hypothesis_test.func(target_columns_str)
    else:
        _write_json("hypothesis_test.json", hypothesis_result)

    # ── Step 3.7: Time-Series Forecasting / Regression / Classification / Clustering (Phase 1, 2, 3 & 4) ──
    prediction_result = {"status": "skipped"}
    forecast_keywords = ["forecast", "project", "estimate future", "projection"]
    regression_keywords = ["predict", "regression", "fit model", "estimate based on"]
    classification_keywords = ["classify", "classification", "probability of", "likelihood of", "probability", "logistic"]
    clustering_keywords = ["cluster", "segment", "group", "partition", "categorize"]
    
    is_forecast = (intent_type == "forecast") or any(kw in user_query.lower() for kw in forecast_keywords)
    is_classification = (intent_type == "classification") or (
        any(kw in user_query.lower() for kw in classification_keywords) and not is_forecast
    )
    is_clustering = (intent_type == "clustering") or (
        any(kw in user_query.lower() for kw in clustering_keywords) 
        and "group by" not in user_query.lower() 
        and "grouped by" not in user_query.lower()
        and not is_forecast 
        and not is_classification
    )
    is_regression = (intent_type == "regression") or (
        any(kw in user_query.lower() for kw in regression_keywords) and not is_forecast and not is_classification and not is_clustering
    )
    
    import pandas as pd
    import numpy as np
    
    if is_clustering:
        logger.info("Executing Clustering engine...")
        cols = intent.get("target_columns", [])
        if len(cols) >= 1:
            import re
            k_val = 3
            match = re.search(r"\b([2-8])\b\s*(?:clusters|groups|segments)", user_query.lower())
            if match:
                k_val = int(match.group(1))
            else:
                match_digit = re.search(r"cluster.*?(\b[2-8]\b)", user_query.lower())
                if match_digit:
                    k_val = int(match_digit.group(1))
                    
            prediction_result = fit_kmeans_clustering(dataframe, cols, k=k_val)
        else:
            prediction_result = {
                "status": "failed",
                "error": "insufficient_columns",
                "message": "To run clustering, specify at least one feature column."
            }
            _write_json("prediction.json", prediction_result)
            
    elif is_classification:
        logger.info("Executing Classification engine...")
        cols = intent.get("target_columns", [])
        if len(cols) >= 2:
            target = cols[0]
            features = cols[1:]
            prediction_result = fit_classification_model(dataframe, target, features)
        else:
            prediction_result = {
                "status": "failed",
                "error": "insufficient_columns",
                "message": "To run a classification model, specify the target column first followed by one or more feature columns."
            }
            _write_json("prediction.json", prediction_result)
            
    elif is_regression:
        logger.info("Executing Regression engine...")
        cols = intent.get("target_columns", [])
        if len(cols) >= 2:
            target = cols[0]
            features = cols[1:]
            prediction_result = fit_regression_model(dataframe, target, features)
        else:
            prediction_result = {
                "status": "failed",
                "error": "insufficient_columns",
                "message": "To run a regression model, specify the target column first followed by one or more feature columns."
            }
            _write_json("prediction.json", prediction_result)
            
    elif is_forecast:
        logger.info("Executing Forecasting engine...")
        time_col = intent["target_columns"][0] if intent["target_columns"] else ""
        metric_col = intent["target_columns"][1] if len(intent["target_columns"]) > 1 else ""
        # Swap date vs metric if needed
        if time_col and metric_col:
            is_num_metric = pd.api.types.is_numeric_dtype(dataframe[metric_col])
            # Swap if metric_col is date-like and time_col is not
            if any(x in str(dataframe[metric_col].dtype).lower() for x in ['datetime', 'period']) or any(x in metric_col.lower() for x in ['year', 'date', 'time', 'month']):
                time_col, metric_col = metric_col, time_col
        elif time_col and not metric_col:
            numeric_cols = [c for c in dataframe.select_dtypes(include=[np.number]).columns if c != time_col]
            if numeric_cols:
                metric_col = numeric_cols[0]
                
        if time_col and metric_col:
            prediction_result = forecast_time_series(dataframe, time_col, metric_col)
        else:
            prediction_result = {
                "status": "failed",
                "error": "insufficient_columns",
                "message": "To run a forecast, please specify a time column and a numeric metric column."
            }
            _write_json("prediction.json", prediction_result)
    else:
        _write_json("prediction.json", prediction_result)

    # ── Step 4: Structured Insight LLM Call ──────────────────────────────────
    logger.info("Generating final structured business insight...")
    insight_result = _generate_insight_via_llm(
        user_query=user_query,
        intent_type=intent_type,
        eda_data=eda_result,
        query_data=query_result,
        chart_data=chart_result,
        hypothesis_data=hypothesis_result,
        prediction_data=prediction_result
    )
    _write_json("insight.json", insight_result)
    
    if task_callback:
        task_callback(MockTaskOutput("generate_insight"))

    logger.info("Omega pipeline execution completed successfully.")
    return insight_result