import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.report import generate_pdf_report
from src.utils import load_json_output

def test_pdf_generation():
    print("Loading last run outputs...")
    insight = load_json_output("insight.json") or {}
    chart = load_json_output("chart.json") or {}
    query = load_json_output("query_result.json") or {}
    prediction = load_json_output("prediction.json") or {}
    hypothesis = load_json_output("hypothesis_test.json") or {}

    mock_result = {
        "query": "Show the distribution of EV adoption likelihood",
        "insight_text": insight.get("insight_text", "Sample summary insight text for EV Adoption distribution analysis."),
        "key_metric": insight.get("key_metric", "65.3% High Adoption Likelihood"),
        "follow_ups": insight.get("follow_up_suggestions", ["Are there differences by city type?"]),
        "intent_type": insight.get("intent_type", "descriptive"),
        "strategies": insight.get("strategies", ["Invest in charging infrastructure.", "Deploy targeted marketing campaigns."]),
        "priority_matrix": insight.get("priority_matrix", [{"action": "Install chargers", "impact": "High", "effort": "Medium"}]),
        "risks": insight.get("risks", ["Budget constraints"]),
        "chart_spec": chart.get("plotly_spec"),
        "chart_gen": chart.get("chart_generated", False),
        "rows": query.get("result_rows", [{"ev_adoption_likelihood": "High", "percentage": 0.5934}, {"ev_adoption_likelihood": "Medium", "percentage": 0.2416}]),
        "truncated": query.get("truncated", False),
        "row_count": query.get("row_count", 0),
        "prediction": prediction,
        "hypothesis": hypothesis
    }

    print("Compiling PDF...")
    try:
        pdf_buffer = generate_pdf_report(mock_result)
        output_path = project_root / "output" / "test_report.pdf"
        with open(output_path, "wb") as f:
            f.write(pdf_buffer.getvalue())
        print(f"Success! PDF report compiled and saved to: {output_path}")
    except Exception as e:
        print(f"Error during PDF generation: {e}")

if __name__ == "__main__":
    test_pdf_generation()
