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
You are a senior data analyst and consultant. Your job is to translate descriptive statistics and data profiling results into a clear, jargon-free data health and completeness overview for a non-technical user.

You must return ONLY a valid JSON object with exactly these eight keys:
- "insight_text": A string summarizing the data health and key insights.
- "key_metric": A short, clean string representing the overall scale of the data.
- "follow_up_suggestions": A list of exactly 2 plain-English follow-up questions.
- "strategies": A list of 3 to 5 specific, highly detailed strategic actions based on the data.
- "priority_matrix": A list of 3 to 5 objects: [{"action": "Action description", "impact": "High"/"Medium"/"Low", "effort": "High"/"Medium"/"Low"}].
- "risks": A list of 2 to 4 potential risks or data quality concerns.
- "error": null (or error message).
- "components": A list of layout components to render. Follow this structure:
  [
    {"type": "markdown", "content": "detailed markdown content"},
    {"type": "metric_grid", "metrics": [{"label": "Metric Name", "value": "Metric Value"}]},
    {"type": "table", "headers": ["Col1", "Col2"], "rows": [["Val1", "Val2"]]},
    {"type": "chart", "plotly_spec": {}}
  ]
"""

_ANALYTICAL_INSIGHT_PROMPT = """
You are a senior business analyst and consultant. Your job is to translate statistical data, queries, and analytical findings into a clear, jargon-free business insight for a non-technical user.

You must return ONLY a valid JSON object with exactly these eight keys:
- "insight_text": A string summarizing the business insight (most important takeaway, context, and recommendation).
- "key_metric": A short string representing the single most important number or finding.
- "follow_up_suggestions": A list of exactly 2 plain-English, conversational follow-up questions.
- "strategies": A list of 3 to 5 specific, highly detailed strategic actions based on the analysis.
- "priority_matrix": A list of 3 to 5 objects: [{"action": "Action description", "impact": "High"/"Medium"/"Low", "effort": "High"/"Medium"/"Low"}].
- "risks": A list of 2 to 4 potential risks or operational concerns.
- "error": null (or error message).
- "components": A list of layout components to render. Follow this structure:
  [
    {"type": "markdown", "content": "detailed markdown content"},
    {"type": "metric_grid", "metrics": [{"label": "Metric Name", "value": "Metric Value"}]},
    {"type": "table", "headers": ["Col1", "Col2"], "rows": [["Val1", "Val2"]]},
    {"type": "chart", "plotly_spec": {}}
  ]
"""

_CONVERSATIONAL_INSIGHT_PROMPT = """
You are the user's senior business consultant, named Omega.
Your goal is to answer the user's question, provide strategic recommendations, and outline decisions.

You must return ONLY a valid JSON object with exactly these eight keys:
- "insight_text": A detailed, thorough string containing a comprehensive response (typically 2 to 3 detailed paragraphs).
- "key_metric": A short, clean string representing the advice topic.
- "follow_up_suggestions": A list of exactly 2 conversational follow-up questions.
- "strategies": A list of 3 to 5 specific, highly detailed strategic actions.
- "priority_matrix": A list of 3 to 5 objects: [{"action": "Action description", "impact": "High"/"Medium"/"Low", "effort": "High"/"Medium"/"Low"}].
- "risks": A list of 2 to 4 potential risks.
- "error": null (or error message).
- "components": A list of layout components to render. Follow this structure:
  [
    {"type": "markdown", "content": "detailed markdown content"},
    {"type": "metric_grid", "metrics": [{"label": "Metric Name", "value": "Metric Value"}]},
    {"type": "table", "headers": ["Col1", "Col2"], "rows": [["Val1", "Val2"]]}
  ]
"""

_PRESCRIPTIVE_INSIGHT_PROMPT = """
You are the user's senior business consultant and prescriptive analytics advisor, named Omega.
Your goal is to propose a concrete, actionable, and data-grounded strategy plan to address the user's query.

You must return ONLY a valid JSON object with exactly these eight keys:
- "insight_text": A detailed, thorough string containing a comprehensive strategic analysis.
- "key_metric": A short topic label.
- "strategies": A list of 3 to 5 specific, highly detailed strategies.
- "priority_matrix": A list of 3 to 5 objects: [{"action": "Action description", "impact": "High"/"Medium"/"Low", "effort": "High"/"Medium"/"Low"}].
- "risks": A list of 2 to 4 potential risks or data quality concerns.
- "follow_up_suggestions": A list of exactly 2 plain-English, conversational follow-up questions.
- "error": null (or error message).
- "components": A list of layout components to render. Follow this structure:
  [
    {"type": "markdown", "content": "detailed markdown content"},
    {"type": "metric_grid", "metrics": [{"label": "Metric Name", "value": "Metric Value"}]},
    {"type": "table", "headers": ["Proposed Action", "Impact", "Effort"], "rows": [["Action 1", "High", "Low"]]}
  ]
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

# Intents that need EDA (descripti# ── Public Entry Point ────────────────────────────────────────────────────────

_PLANNER_SYSTEM_PROMPT = """You are a senior data science and analytics consulting planner. Given a user query, a dataset schema, sample rows, chat history, and the Semantic Business Model, formulate a highly detailed, step-by-step statistical execution plan.
Your goal is to guide a python coder agent on exactly how to analyze the dataset and what output files to write.

Instead of answering the query with simple descriptive text or code, you must design a structured, multi-stage Decision Intelligence workflow:
1. DESCRIPTIVE: Calculate the current trend, magnitude, and direct stats of interest.
2. DIAGNOSTIC: Investigate root causes and driver metrics (e.g. drill-down by organizational/geographic hierarchies, cohort analysis, correlation tests).
3. PREDICTIVE: Forecast future trajectory if the current trend continues.
4. PRESCRIPTIVE: Identify actionable business recommendations, estimate the expected business impact/revenue recovery of those recommendations, and detail associated risks.

You MUST structure your plan using the following XML tags:

<data_profile>
Identify the columns, shapes, and types relevant to the query.
</data_profile>

<data_cleaning>
Detail data cleaning steps (e.g. drop nulls in target columns, type conversions, filtering conditions).
</data_cleaning>

<analysis_steps>
Describe the exact pandas/numpy calculations, groupings, correlations, or mathematical formulas to address the descriptive, diagnostic, predictive, and prescriptive steps. If a statistical or predictive model (e.g. regression, classification, clustering, forecasting) is required, specify the pre-injected helper function to call.
</analysis_steps>

<chart_spec>
Describe the Plotly visualization type, axes, labels, and titles to render, adhering to Plotly formatting rules.
</chart_spec>

<output_files>
Detail which JSON files to write (query_result.json, eda_result.json, chart.json, hypothesis_test.json, prediction.json, insight.json) and their expected structures. Ensure that `insight.json` contains strategic recommendations, impact matrices, and risks.
</output_files>

Do NOT write any Python code blocks. Focus 100% on logical and mathematical reasoning steps."""

_CODER_SYSTEM_PROMPT = """You are an expert Python programmer.
Your goal is to translate a detailed data science execution plan into clean, executable, robust Python code.
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

If the plan asks to build, train, fit, forecast, segment, cluster, or predict models, do NOT import `sklearn` or build custom fitting routines. Simply call the appropriate pre-injected helper function directly on the dataframe `df`! Do not call `write_output_json` for `prediction.json` manually if you use these functions, as they will save `prediction.json` automatically.

CRITICAL RULES FOR PANDAS AND PLOTLY:
- PANDAS MONTHLY FREQUENCY DEPRECATION: Never use 'M' as a frequency parameter in resample() or offsets. It raises a ValueError in current pandas versions. Always use 'ME' (Month End) instead!
- When using `fig.add_vline(x=...)` or `fig.add_hline(y=...)`, the value of `x` or `y` must be a clean numeric float or int. Never pass a string, a pandas Series, or a numpy object directly. Convert it using `float(value)` first.
- If labeling categories or bins on the x-axis, do not use `fig.add_vline` with category string coordinates. Only use numeric coordinates on numeric axes.

You MUST write a Python script that executes the plan and writes outputs using `write_output_json`.
Depending on the plan requirements, you must structure the JSON output files EXACTLY as follows:

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

3. `chart.json`: REQUIRED if a chart/visualization was requested or makes sense.
   Format:
   {
     "status": "success",
     "chart_generated": true,
     "chart_type": "bar/scatter/line/etc",
     "chart_title": "Descriptive title",
     "plotly_spec": the plotly figure exported as a dict (call `json.loads(fig.to_json())`)
   }

4. `hypothesis_test.json`: REQUIRED if the user asks for correlation, significance, or comparison tests.
   Format:
   {
     "status": "success",
     "test_name": "T-test/Pearson/Chi-Square/etc",
     "statistic_name": "Name of statistic",
     "statistic_value": float,
     "p_value": float,
     "null_hypothesis": "...",
     "alternative_hypothesis": "...",
     "interpretation": "...",
     "is_significant": bool
   }

5. `prediction.json`: REQUIRED if the user asks for forecasting, regression, classification, or clustering models.
   Format must match the model helper return format.

6. `insight.json`: ALWAYS REQUIRED. Composes the final executive business insights.
   Format:
   {
     "insight_text": "A string containing exactly 3 to 5 sentences of business insight summary. Explain not just WHAT happened, but WHY it happened, what will happen next, and what decision should be made. Do NOT output HTML tags like <div> or <p> inside this string; return clean, plain text.",
     "key_metric": "Single highlight metric (e.g. '87% Adoption' or '-0.65 correlation')",
     "follow_up_suggestions": list of exactly 2 plain-English, conversational follow-up questions,
     "intent_type": "descriptive/forecast/regression/classification/clustering/prescriptive",
     "strategies": list of strings for strategic recommendations (Provide at least 3 concrete, data-grounded strategic actions),
     "priority_matrix": list of dicts with keys "action", "impact" (High/Medium/Low), "effort" (High/Medium/Low) matching your recommendations,
     "risks": list of strings for potential risks/limitations of the decisions recommended
   }

Return ONLY the executable python code block enclosed inside ```python ... ``` fences. Do not include markdown text or explanations outside the code block."""


# ── Public Entry Point ────────────────────────────────────────────────────────

def run_omega(
    user_query:    str,
    dataframe,
    step_callback: Optional[Callable] = None,
    task_callback: Optional[Callable] = None,
    chat_history:  Optional[list] = None,
) -> Dict[str, Any]:
    """
    Agentic Reasoning Platform (Omega V3).
    Formulates an analytical plan, writes Python code, runs it in a secure sandbox,
    self-corrects errors, and serializes results for Streamlit.
    """
    from .interpreter import execute_code
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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

    # Step 2.3: Load the persistent business model if it exists
    business_model_str = ""
    try:
        bm_path = Path(get_output_path("business_model.json"))
        if bm_path.exists():
            with open(bm_path, "r", encoding="utf-8") as f:
                business_model_str = f.read()
    except Exception as e:
        logger.warning(f"Could not load business model: {e}")

    bm_context = f"\nSemantic Business Model:\n{business_model_str}\n" if business_model_str else ""

    # Step 2.5: Build chat history prompt context
    history_context_str = ""
    if chat_history:
        history_context_str = "\nPrevious Conversation turns:\n"
        for turn in chat_history:
            q = turn.get("query", "")
            ans = turn.get("insight_text", "")
            history_context_str += f"- User asked: \"{q}\"\n- Insight returned: \"{ans}\"\n"

    # Step 3: Run the Planner Agent to generate a markdown plan
    logger.info("Planner Agent generating analytical plan...")
    planner_messages = [
        {"role": "system", "content": _PLANNER_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": f"User Query: {user_query}\n\nDataset Shape: {shape_str}\nDataset Schema:\n{schema_str}\nDataset Sample (first 5 rows):\n{sample_str}\n{bm_context}\n{history_context_str}"}
    ]
    try:
        planner_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=planner_messages,
            temperature=0.2,
        )
        analytical_plan = planner_response.choices[0].message.content
        logger.info(f"Analytical plan formulated successfully:\n{analytical_plan[:300]}...")
    except Exception as e:
        logger.error(f"Planner Agent failed: {e}")
        analytical_plan = f"Analyze the dataset schema and user query '{user_query}' to extract key stats and render a plotly chart."

    # Step 4: Setup Coder Agent instructions & loop
    # Filter schema_str to only include columns referenced in the plan or query (Dynamic Schema Truncation)
    truncated_schema_lines = []
    columns_in_df = dataframe.columns.tolist()
    referenced_cols = []
    for col in columns_in_df:
        if col.lower() in user_query.lower() or (analytical_plan and col.lower() in analytical_plan.lower()):
            referenced_cols.append(col)
            
    if referenced_cols:
        schema_lines = schema_str.split("\n")
        for line in schema_lines:
            if "|" not in line:
                truncated_schema_lines.append(line)
            else:
                is_referenced = False
                for col in referenced_cols:
                    if f"  {col} |" in line or f"| {col} |" in line or line.strip().startswith(f"{col} |"):
                        is_referenced = True
                        break
                if is_referenced:
                    truncated_schema_lines.append(line)
                else:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 3:
                        truncated_schema_lines.append(f"  {parts[0]} | {parts[1]} | {parts[2]}")
                    else:
                        truncated_schema_lines.append(line)
        coder_schema_str = "\n".join(truncated_schema_lines)
    else:
        coder_schema_str = schema_str

    coder_instruction = f"""User query: {user_query}

Dataset metadata:
- Shape: {shape_str}
- Schema (Optimized/Truncated to relevant columns):
{coder_schema_str}

Dataset Sample (first 5 rows):
{sample_str}

Execution Plan generated by Planner Agent:
{analytical_plan}

Please generate the Python code to perform this analysis and write the required json output files following the plan."""

    code = ""
    max_attempts = 3
    history = [
        {"role": "system", "content": _CODER_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": coder_instruction}
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
                history.append({
                    "role": "user",
                    "content": f"The code execution failed with the following error:\n{result['error']}\n\nPlease fix the bug and return the corrected python code."
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


def bootstrap_omega(dataframe) -> dict:
    """Exposes the business model bootstrapping functionality."""
    from .semantic_model import bootstrap_business_model
    return bootstrap_business_model(dataframe)