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
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
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
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
    
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

# Intents that need EDA (descripti# ── Public Entry Point ────────────────────────────────────────────────────────

def run_omega(
    user_query:    str,
    dataframe,
    step_callback: Optional[Callable] = None,
    task_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Agentic Reasoning Platform (Omega V3).
    Formulates an analytical plan, writes Python code, runs it in a secure sandbox,
    self-corrects errors, and serializes results for Streamlit.
    """
    from .interpreter import execute_code
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())

    # Step 1: Initialize empty/skipped state for all output files to prevent frontend hang
    logger.info("Initializing default output JSON states...")
    _write_json("eda_result.json", {"status": "skipped", "observations": [], "summary_stats": {}, "outlier_flags": []})
    _write_json("query_result.json", {"status": "skipped", "result_rows": [], "row_count": 0})
    _write_json("chart.json", {"status": "skipped", "plotly_spec": None, "chart_generated": False})
    _write_json("hypothesis_test.json", {"status": "skipped"})
    _write_json("prediction.json", {"status": "skipped"})
    _write_json("insight.json", {
        "insight_text": "Preparing analysis...",
        "key_metric": "",
        "follow_up_suggestions": []
    })

    if task_callback:
        task_callback(MockTaskOutput("run_eda"))

    # Step 2: Build the dataframe context & schema
    schema_str = build_schema_string(dataframe)
    shape_str = f"{dataframe.shape[0]} rows, {dataframe.shape[1]} columns"
    sample_str = dataframe.head(5).to_string()

    system_prompt = """You are Omega V3, an autonomous senior data scientist agent.
Your goal is to solve the user's business and data analytics queries by generating executable Python code.
The user dataset is loaded in memory as a pandas DataFrame named `df`.

You have access to the following python packages:
- `pandas` as `pd`
- `numpy` as `np`
- `scipy`
- `plotly`
- `plotly.graph_objects` as `go`
- `plotly.express` as `px`

You also have access to these pre-injected helper functions:
- `get_output_path(filename: str) -> str`: gets the correct output path for files.
- `write_output_json(filename: str, data: dict) -> None`: writes data as JSON to the correct output path.
- `fit_regression_model(df, target_col: str, feature_cols: list) -> dict`: Fits a multiple linear regression model, writes results to `prediction.json`, and returns result dict.
- `fit_classification_model(df, target_col: str, feature_cols: list) -> dict`: Fits a logistic regression classification model, writes results to `prediction.json`, and returns result dict.
- `fit_kmeans_clustering(df, feature_cols: list, k: int) -> dict`: Fits a K-Means clustering model, writes results to `prediction.json`, and returns result dict.
- `forecast_time_series(df, time_col: str, metric_col: str) -> dict`: Generates time-series forecast projections, writes results to `prediction.json`, and returns result dict.

If the user query asks to build, train, fit, forecast, segment, cluster, or predict models, do NOT import `sklearn` or build custom fitting routines. Simply call the appropriate pre-injected helper function directly on the dataframe `df`! Do not call `write_output_json` for `prediction.json` manually if you use these functions, as they will save `prediction.json` automatically.

You MUST write a Python script that calculates the answers and writes outputs to the output directory using `write_output_json`.
Specifically, you should write the following files depending on what is relevant to the user's query:

1. `query_result.json`: REQUIRED for any data filtering, aggregation, comparison, or SQL-like questions.
   Format:
   {
     "status": "success",
     "sql_query": "An equivalent SQL query representing the operations performed",
     "result_rows": list of dicts (records),
     "row_count": number of rows,
     "truncated": false
   }

2. `eda_result.json`: REQUIRED for descriptive, distribution, or data overview questions.
   Format:
   {
     "status": "success",
     "summary_stats": dict of descriptive metrics,
     "observations": list of strings outlining key observations,
     "outlier_flags": list of dicts/strings flagging anomalies
   }

3. `chart.json`: REQUIRED if a chart/visualization was requested or makes sense for the analysis.
   Format:
   {
     "status": "success",
     "chart_generated": true,
     "chart_type": "bar/scatter/line/etc",
     "chart_title": "Descriptive title",
     "plotly_spec": the plotly figure exported as a JSON-serializable dict (e.g., call `json.loads(fig.to_json())` or `fig.to_plotly_json()`)
   }
   IMPORTANT: Create interactive Plotly figures using `go` or `px`, use beautiful colors, and call `fig.to_json()` to populate the spec.

4. `hypothesis_test.json`: REQUIRED if the user asks for correlation, significance, or comparison tests.
   Format:
   {
     "status": "success",
     "test_name": "T-test/Pearson/Chi-Square/etc",
     "statistic_name": "Name of the statistic (e.g. t-statistic, correlation coefficient)",
     "statistic_value": float,
     "p_value": float,
     "null_hypothesis": "...",
     "alternative_hypothesis": "...",
     "interpretation": "Detailed statistical interpretation and conclusion...",
     "is_significant": bool
   }

5. `prediction.json`: REQUIRED if the user asks for forecasting, regression, classification, or clustering models.
   Format must match the model type:
   - For forecasting:
     {"status": "success", "time_column": "col_name", "metric_column": "col_name", "historical_dates": [...], "historical_values": [...], "forecast_dates": [...], "forecast_values": [...], "lower_bound": [...], "upper_bound": [...], "model_metrics": {...}}
   - For regression:
     {"status": "regression", "target_column": "col_name", "features": [...], "intercept": float, "coefficients": {col: float}, "model_metrics": {...}, "dummy_mappings": {}}
   - For classification:
     {"status": "classification", "model_mode": "binary" or "multiclass", "target_column": "col_name", "features": [...], "model_metrics": {...}, "target_label": "col_name", "class_0_label": "0", "class_1_label": "1", "intercept": float, "coefficients": {col: float}, "classes": [...], "intercepts": {...}, "coefficients": {...}, "dummy_mappings": {}}
   - For clustering:
     {"status": "clustering", "features": [...], "clusters": [...], "labels": [...], "pc_coords": [...], "sample_size": int, "means": {...}}

6. `insight.json`: ALWAYS REQUIRED. Composes the final executive insight.
   Format:
   {
     "insight_text": "Comprehensive analysis of the findings...",
     "key_metric": "Single highlight metric (e.g. '87% Adoption' or '-0.65 correlation')",
     "follow_up_suggestions": list of follow-up questions,
     "intent_type": "descriptive/forecast/regression/classification/clustering/prescriptive",
     "strategies": list of strings for strategic recommendations,
     "priority_matrix": list of dicts with keys "action", "impact", "effort" (each 'High/Medium/Low'),
     "risks": list of strings for potential risks/limitations
   }

RULES FOR YOUR PYTHON CODE:
- Always handle missing/null values gracefully in code (e.g. dropna or fillna).
- If asked to calculate correlations against a categorical column (e.g., target or feature is non-numeric/string), you MUST encode it numerically (e.g. using `df[col].astype('category').cat.codes` or label mapping) before running the correlation.
- To maintain optimal speed and prevent memory issues on the cloud server, always sample the dataframe to a maximum of 15,000 rows (e.g. `df_sample = df.sample(n=min(15000, len(df)), random_state=42)`) before performing complex modeling, model training, or large statistics computations.
- Never mix string indicators (like "Other" or "N/A") with numeric values in the same list or pandas Series if you plan to sort, group, or pass them to Plotly. Always cast the series to `astype(str)` if it contains mixed types.
- When writing plotly charts, NEVER add a Figure object as a trace. Do not call `fig.add_trace(px.pie(...))`. Use Plotly Express figures directly or use `fig.add_trace(go.Pie(...))`.
- Write clean, robust, and commented code.
- Return ONLY the executable python code block enclosed inside ```python ... ``` fences. Do not include markdown text or explanations outside the code block.
"""

    user_message = f"""User query: {user_query}

Dataset metadata:
- Shape: {shape_str}
- Schema:
{schema_str}

Dataset Sample (first 5 rows):
{sample_str}

Please generate the Python code to perform this analysis and write the required json output files.
"""

    code = ""
    max_attempts = 3
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    for attempt in range(1, max_attempts + 1):
        logger.info(f"Attempting to generate python code (Attempt {attempt}/{max_attempts})...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=history,
                temperature=0.1,
            )
            reply = response.choices[0].message.content
            history.append({"role": "assistant", "content": reply})

            # Extract code block
            import re
            code_match = re.search(r"```python(.*?)```", reply, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                code = reply.strip()
                if code.startswith("```"):
                    code = code.strip("`").strip("python").strip()

            logger.info("Executing generated code in sandbox...")
            if task_callback:
                task_callback(MockTaskOutput("run_query"))

            result = execute_code(code, dataframe)

            if result["success"]:
                logger.info("Sandbox execution completed successfully!")
                break
            else:
                logger.warning(f"Execution error on attempt {attempt}: {result['error']}")
                
                error_msg = result['error'] or ""
                error_guidance = ""
                if "TypeError" in error_msg and "str" in error_msg and ("int" in error_msg or "float" in error_msg or "NoneType" in error_msg):
                    error_guidance = (
                        "\n\n💡 TIP: You hit a type comparison error. Ensure that any column you are "
                        "sorting, grouping, or passing to Plotly is explicitly cast to a uniform type "
                        "(e.g., call `.astype(str)` or `.astype(float)`) before performing comparisons, "
                        "grouping, unique value extraction, or sorting."
                    )
                
                history.append({
                    "role": "user",
                    "content": f"The code execution failed with the following error:\n{error_msg}{error_guidance}\n\nPlease fix the bug and return the corrected python code."
                })
        except Exception as e:
            logger.error(f"Error during agent completion loop: {e}")
            break

    if task_callback:
        task_callback(MockTaskOutput("render_chart"))
        task_callback(MockTaskOutput("generate_insight"))

    # Read the final written insight.json
    final_insight = _load_json_safely("insight.json")
    if not final_insight or final_insight.get("insight_text") == "Preparing analysis...":
        logger.info("Sandbox script did not write insight.json. Invoking fallback LLM insight generator...")
        eda_data = _load_json_safely("eda_result.json")
        query_data = _load_json_safely("query_result.json")
        chart_data = _load_json_safely("chart.json")
        hypothesis_data = _load_json_safely("hypothesis_test.json")
        prediction_data = _load_json_safely("prediction.json")
        
        # Infer intent type for system prompt selection
        intent_type = "descriptive"
        if prediction_data and prediction_data.get("status") in ["regression", "classification", "clustering", "success"]:
            intent_type = prediction_data.get("status")
            if intent_type == "success":
                intent_type = "forecast"
        elif "correlation" in user_query.lower() or "test" in user_query.lower():
            intent_type = "correlation"
        
        final_insight = _generate_insight_via_llm(
            user_query=user_query,
            intent_type=intent_type,
            eda_data=eda_data,
            query_data=query_data,
            chart_data=chart_data,
            hypothesis_data=hypothesis_data,
            prediction_data=prediction_data
        )

    # Write the final result back to disk so Streamlit UI updates
    _write_json("insight.json", final_insight)
    return final_insight