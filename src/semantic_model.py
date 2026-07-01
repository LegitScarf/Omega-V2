import os
import json
import logging
import pandas as pd
import numpy as np
from openai import OpenAI
from typing import Dict, Any
from .schema import build_schema_string
from .utils import get_output_path

logger = logging.getLogger("Omega.SemanticModel")

_BOOTSTRAP_SYSTEM_PROMPT = """
You are a senior analytics consultant. Your job is to analyze a dataset's schema and sample data, and build a persistent "Semantic Business Model" representing the business domain, hierarchies, key KPIs, and relationships.

Your output will be used to guide future analytical queries and pre-populate an executive dashboard.

Return ONLY a valid JSON object with exactly these keys:
{
  "business_domain": "SaaS / Retail / Finance / Manufacturing / Healthcare / etc.",
  "executive_summary": "A 2-3 sentence overview of what business operation this dataset represents.",
  "kpis": [
    {
      "name": "KPI name (e.g. Churn Rate, Revenue, Average Ticket Size)",
      "column": "column_name in dataset",
      "metric_type": "sum / average / percentage / ratio",
      "business_importance": "Primary / Secondary",
      "description": "Why this KPI matters"
    }
  ],
  "hierarchies": [
    {
      "name": "Hierarchy Name (e.g. Geographic, Product, Customer Class)",
      "levels": ["highest_level_column", "middle_level_column", "lowest_level_column"]
    }
  ],
  "suggested_analyses": [
    {
      "type": "descriptive / diagnostic / predictive / prescriptive",
      "title": "Short title of analysis (e.g. Regional Profitability Analysis)",
      "description": "What business decision this analysis would support and why it matters."
    }
  ]
}

Rules:
- Only reference columns that actually exist in the schema.
- Infer natural organizational or geographic hierarchies.
- Be precise and business-focused.
- Do not return markdown, explanation, or code fences - JSON only.
"""

def bootstrap_business_model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes the dataframe to build a persistent business model.
    Saves the output to 'business_model.json'.
    """
    logger.info("Starting Semantic Business Modeling...")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    schema_str = build_schema_string(df)
    sample_str = df.head(10).to_string()
    shape_str = f"{df.shape[0]} rows, {df.shape[1]} columns"
    
    user_message = (
        f"Dataset shape: {shape_str}\n\n"
        f"Dataset schema:\n{schema_str}\n\n"
        f"Dataset sample (first 10 rows):\n{sample_str}"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _BOOTSTRAP_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1
        )
        model_data = json.loads(response.choices[0].message.content)
        
        # Enrich the model data with actual calculated metric statistics from the dataframe
        model_data = _enrich_with_stats(df, model_data)
        
        # Save to output path
        output_path = get_output_path("business_model.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2)
            
        logger.info("Semantic Business Model successfully bootstrapped and written.")
        return model_data
    except Exception as e:
        logger.exception(f"Semantic bootstrapping failed: {e}")
        fallback = {
            "business_domain": "Generic Business Operations",
            "executive_summary": "Auto-profiled dataset representing company records.",
            "kpis": [],
            "hierarchies": [],
            "suggested_analyses": [],
            "metrics": {"Total Rows": len(df), "Total Columns": len(df.columns)}
        }
        output_path = get_output_path("business_model.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(fallback, f, indent=2)
        return fallback

def _enrich_with_stats(df: pd.DataFrame, model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates summary stats for inferred KPIs to populate dashboard immediately."""
    enriched_kpis = []
    summary_metrics = {}
    
    for kpi in model_data.get("kpis", []):
        col = kpi.get("column")
        mtype = kpi.get("metric_type")
        if col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            val = None
            if pd.api.types.is_numeric_dtype(series):
                if mtype == "sum":
                    val = float(series.sum())
                elif mtype == "average":
                    val = float(series.mean())
                elif mtype in ("percentage", "ratio"):
                    # Check if standard ratio or needs mean calculation
                    val = float(series.mean())
                else:
                    val = float(series.mean())
            elif pd.api.types.is_bool_dtype(series):
                val = float(series.mean() * 100) # percentage
            
            if val is not None:
                kpi["current_value"] = val
                # Add formatted string
                if mtype == "percentage" or "percent" in kpi["name"].lower() or "%" in kpi["name"].lower():
                    kpi["formatted_value"] = f"{val:,.2f}%" if val <= 100 else f"{val:,.2f}"
                elif mtype == "ratio":
                    kpi["formatted_value"] = f"{val:.2f}"
                else:
                    kpi["formatted_value"] = f"{val:,.2f}"
                
                enriched_kpis.append(kpi)
                
    model_data["kpis"] = enriched_kpis
    
    # Calculate some general metric statistics
    summary_metrics["Total Records"] = f"{len(df):,}"
    summary_metrics["Total Features"] = len(df.columns)
    
    # Check for date ranges
    date_cols = df.select_dtypes(include=["datetime", "object"]).columns
    for col in date_cols:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            if not parsed.empty:
                summary_metrics["Date Range"] = f"{parsed.min().strftime('%Y-%m-%d')} to {parsed.max().strftime('%Y-%m-%d')}"
                break
        except Exception:
            pass
            
    model_data["metrics"] = summary_metrics
    return model_data
