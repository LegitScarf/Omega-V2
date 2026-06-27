import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import plotly.graph_objects as go
from src.report import generate_pdf_report

def run_test():
    # Simulate a chart generated in a dark mode template
    fig = go.Figure(
        data=[go.Bar(x=['A', 'B', 'C'], y=[4, 5, 6], marker_color='#ff7f0e')],
    )
    # Apply streamlit-like dark theme layout settings
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF")
    )
    
    mock_result = {
        "query": "Test Dark Chart Query",
        "insight_text": "This is a test of a dark chart export.",
        "key_metric": "100% Test",
        "intent_type": "descriptive",
        "chart_spec": fig.to_dict(),
        "chart_gen": True,
        "rows": [{"A": 1, "B": 2}]
    }
    
    print("Generating report with dark chart...")
    pdf_buffer = generate_pdf_report(mock_result)
    output_path = project_root / "output" / "test_dark_report.pdf"
    with open(output_path, "wb") as f:
        f.write(pdf_buffer.getvalue())
    print(f"Report generated successfully: {output_path}")

if __name__ == "__main__":
    run_test()
