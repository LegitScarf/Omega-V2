import io
import logging
from io import BytesIO
from typing import Dict, Any, Optional
from fpdf import FPDF
import plotly.graph_objects as go
import pandas as pd

logger = logging.getLogger("Omega.Report")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2014": "-",
        "\u2013": "-",
        "\u2022": "*",
        "\u2026": "...",
        "\u2b21": "",
        "\u2014": "-",
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    return text.encode("latin-1", errors="replace").decode("latin-1")

class OmegaPDFReport(FPDF):
    def header(self):
        # Draw header banner
        self.set_fill_color(37, 99, 235)  # Omega Accent Blue #2563EB
        self.rect(0, 0, 210, 15, "F")
        
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 11)
        self.cell(180, 5, "OMEGA - DATA ANALYTICS REPORT", align="L")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_text_color(150, 150, 150)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Omega AI Analytics Platform", align="C")

def _clean_plotly_spec(spec: Any) -> Any:
    if not isinstance(spec, dict):
        return spec
        
    cleaned_spec = spec.copy()
    if "data" in cleaned_spec:
        raw_data = cleaned_spec["data"]
        new_data = []
        
        if isinstance(raw_data, dict):
            raw_data = [raw_data]
        elif not isinstance(raw_data, list):
            raw_data = []
            
        for item in raw_data:
            if isinstance(item, dict):
                if "data" in item:
                    nested_data = item["data"]
                    if isinstance(nested_data, list):
                        for sub_item in nested_data:
                            new_data.append(_clean_plotly_spec(sub_item))
                    elif isinstance(nested_data, dict):
                        new_data.append(_clean_plotly_spec(nested_data))
                else:
                    new_data.append(_clean_plotly_spec(item))
            else:
                new_data.append(item)
                
        cleaned_spec["data"] = new_data
        
    return cleaned_spec


def generate_pdf_report(result: Dict[str, Any]) -> BytesIO:
    """
    Compiles insights, charts, tables, and prescriptive actions
    from the run result into a styled PDF document bytes buffer.
    """
    pdf = OmegaPDFReport(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.alias_nb_pages()
    pdf.add_page()

    # Available content width is 180mm
    width = 180

    # 1. Metadata / Header Info
    pdf.set_text_color(50, 50, 50)
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(width, 7, clean_text(f"Analysis: {result.get('query', 'Custom Query')}"))
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(90, 5, clean_text(f"Date: {now_str}"), ln=0)
    pdf.cell(90, 5, clean_text(f"Intent Mode: {result.get('intent_type', 'descriptive').upper()}"), ln=1)
    pdf.ln(4)

    # 2. Key Insight Callout Box
    pdf.set_fill_color(245, 245, 245) # Light grey background
    pdf.set_draw_color(220, 220, 220)
    pdf.rect(15, pdf.get_y(), width, 40, "FD")
    
    # Inner Content
    pdf.set_y(pdf.get_y() + 3)
    pdf.set_x(20)
    pdf.set_text_color(37, 99, 235)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(170, 5, "KEY INSIGHT / SUMMARY")
    pdf.ln(6)
    
    key_metric = result.get("key_metric", "")
    if key_metric:
        pdf.set_x(20)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(17, 17, 17)
        pdf.cell(170, 5, clean_text(f"Metric Highlight: {key_metric}"))
        pdf.ln(6)
        
    pdf.set_x(20)
    pdf.set_text_color(85, 85, 85)
    pdf.set_font("Helvetica", "", 9.5)
    insight_text = result.get("insight_text", "No summary insight generated.")
    # Standardize spaces / clean text
    insight_text = insight_text.replace("\n", " ")
    pdf.multi_cell(170, 5.5, clean_text(insight_text))
    
    # Restore position after relative rect positioning
    pdf.set_y(pdf.get_y() + 6)
    pdf.ln(4)

    # 3. Embedding Chart
    chart_spec = result.get("chart_spec")
    chart_gen = result.get("chart_gen", False)
    
    if chart_gen and chart_spec:
        pdf.set_text_color(17, 17, 17)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(width, 5, "VISUALIZATION", ln=1)
        pdf.ln(3)
        
        try:
            # Recreate figure using Plotly dict spec
            cleaned_spec = _clean_plotly_spec(chart_spec)
            fig = go.Figure(cleaned_spec)
            
            # Reset layout template to plotly_white and apply premium design rules
            fig.update_layout(
                template="plotly_white",
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(
                    family="Helvetica Neue, Helvetica, Arial, sans-serif",
                    size=9,
                    color='#374151'  # Tailwind Gray-700
                ),
                title=dict(
                    font=dict(
                        family="Helvetica Neue, Helvetica, Arial, sans-serif",
                        size=13,
                        color='#111827'  # Tailwind Gray-900
                    )
                ),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            
            # Refine axes lines and gridlines
            fig.update_xaxes(
                showgrid=True,
                gridcolor='#F3F4F6',  # Very light grey gridlines
                linecolor='#E5E7EB',  # Clean border line
                tickfont=dict(color='#6B7280'),
                zerolinecolor='#E5E7EB'
            )
            fig.update_yaxes(
                showgrid=True,
                gridcolor='#F3F4F6',
                linecolor='#E5E7EB',
                tickfont=dict(color='#6B7280'),
                zerolinecolor='#E5E7EB'
            )
            
            # Update trace aesthetics dynamically (e.g. lines, markers, colors)
            for trace in fig.data:
                # Map old template/generic colors to premium colors
                if hasattr(trace, 'marker') and trace.marker:
                    if hasattr(trace.marker, 'color') and trace.marker.color == "#4f86c6":
                        trace.marker.color = "#2563EB"  # Vibrant brand blue
                if hasattr(trace, 'line') and trace.line:
                    if hasattr(trace.line, 'color'):
                        if trace.line.color == "#4f86c6":
                            trace.line.color = "#2563EB"
                        elif trace.line.color == "#e07b39":
                            trace.line.color = "#F97316"  # Vibrant orange
                    # Smooth line charts
                    if hasattr(trace, 'mode') and trace.mode and "lines" in trace.mode:
                        trace.line.shape = "spline"
                        trace.line.width = 2.5
            
            # Convert to PNG image in memory
            # Kaleido engine runs inside plotly's to_image
            img_bytes = fig.to_image(format="png", width=700, height=380, scale=2)
            
            # Add to document
            pdf.image(BytesIO(img_bytes), x=15, w=width)
            pdf.ln(6)
        except Exception as e:
            logger.warning(f"Failed to compile and embed chart in PDF: {e}")
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(150, 150, 150)
            pdf.cell(width, 5, "[Chart visualization could not be rendered in PDF format]", ln=1)
            pdf.ln(4)

    # 4. Data Table
    rows = result.get("rows", [])
    if rows:
        pdf.set_text_color(17, 17, 17)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(width, 5, "DATA RECORDS", ln=1)
        pdf.ln(3)
        
        try:
            if isinstance(rows, dict):
                if all(not isinstance(v, (list, dict, tuple)) for v in rows.values()):
                    df_table = pd.DataFrame([rows])
                else:
                    df_table = pd.DataFrame(rows)
            elif isinstance(rows, list):
                if rows and not isinstance(rows[0], dict) and not isinstance(rows[0], (list, tuple)):
                    df_table = pd.DataFrame(rows, columns=["Value"])
                else:
                    df_table = pd.DataFrame(rows)
            else:
                df_table = pd.DataFrame(rows)
            # Select first 20 rows to avoid extremely long tables
            df_table = df_table.head(20)
            headers = df_table.columns.tolist()
            
            pdf.set_font("Helvetica", "", 8.5)
            pdf.set_text_color(50, 50, 50)
            
            with pdf.table(col_widths=None, text_align="LEFT", padding=2) as table:
                # Headers Row
                header_row = table.row()
                for header in headers:
                    header_row.cell(clean_text(str(header).upper()))
                
                # Data Rows
                for _, row in df_table.iterrows():
                    data_row = table.row()
                    for col in headers:
                        val = row[col]
                        if isinstance(val, float):
                            val_str = f"{val:.4f}"
                        else:
                            val_str = str(val)
                        data_row.cell(clean_text(val_str))
            pdf.ln(6)
        except Exception as e:
            logger.warning(f"Failed to generate PDF table: {e}")

    # 5. Prescriptive details
    strategies = result.get("strategies", [])
    priority_matrix = result.get("priority_matrix", [])
    
    if strategies or priority_matrix:
        pdf.set_text_color(17, 17, 17)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(width, 5, "RECOMMENDED STRATEGIES & MATRIX", ln=1)
        pdf.ln(3)
        
        if strategies:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(85, 85, 85)
            for strat in strategies:
                pdf.multi_cell(width, 5.5, clean_text(f"• {strat}"))
            pdf.ln(4)
            
        if priority_matrix:
            try:
                pdf.set_font("Helvetica", "", 8.5)
                with pdf.table(col_widths=[100, 40, 40], text_align="LEFT", padding=2) as table:
                    # Headers
                    h_row = table.row()
                    h_row.cell("Proposed Action")
                    h_row.cell("Impact")
                    h_row.cell("Effort")
                    
                    # Rows
                    for row in priority_matrix:
                        r_row = table.row()
                        r_row.cell(clean_text(str(row.get("action", ""))))
                        r_row.cell(clean_text(str(row.get("impact", ""))))
                        r_row.cell(clean_text(str(row.get("effort", ""))))
                pdf.ln(6)
            except Exception as e:
                logger.warning(f"Failed to draw priority matrix table: {e}")

    # Compile the final document to a buffer and return bytes
    buffer = BytesIO()
    pdf_bytes = pdf.output()
    buffer.write(pdf_bytes)
    buffer.seek(0)
    return buffer
