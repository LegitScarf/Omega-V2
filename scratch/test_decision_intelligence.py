import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from src.crew import bootstrap_omega, run_omega
from src.utils import get_output_path, clear_output_dir

def generate_mock_retail_data():
    np.random.seed(42)
    n_rows = 200
    
    dates = pd.date_range(start="2026-01-01", periods=n_rows, freq="D")
    regions = np.random.choice(["North India", "South India", "East India", "West India"], n_rows)
    categories = np.random.choice(["Electronics", "Apparel", "Home & Kitchen"], n_rows)
    products = np.random.choice(["Prod-A", "Prod-B", "Prod-C"], n_rows)
    sales = np.random.uniform(100.0, 1500.0, n_rows)
    cost = sales * np.random.uniform(0.5, 0.8, n_rows)
    discount = np.random.uniform(0.0, 20.0, n_rows)
    customer_ids = [f"Cust-{np.random.randint(1000, 1050)}" for _ in range(n_rows)]
    units = np.random.randint(1, 10, n_rows)
    
    df = pd.DataFrame({
        "Transaction_Date": dates,
        "Region_Name": regions,
        "Category": categories,
        "Product_SKU": products,
        "Sales_Revenue": sales,
        "Product_Cost": cost,
        "Discount_Pct": discount,
        "Customer_ID": customer_ids,
        "Units_Sold": units
    })
    return df

def test_decision_intelligence():
    print("Generating mock retail dataset...")
    df = generate_mock_retail_data()
    
    print("Clearing output directory...")
    clear_output_dir()
    
    print("Running bootstrap_omega...")
    bm = bootstrap_omega(df)
    
    print("\n--- Bootstrapped Business Model Metadata ---")
    print(f"Domain: {bm.get('business_domain')}")
    print(f"Summary: {bm.get('executive_summary')}")
    print(f"Metrics: {bm.get('metrics')}")
    print("KPIs Inferred:")
    for kpi in bm.get("kpis", []):
        print(f" - {kpi.get('name')}: {kpi.get('formatted_value')} (Column: {kpi.get('column')})")
        
    # Basic assertions
    assert bm.get("business_domain") is not None
    assert len(bm.get("kpis")) > 0
    assert len(bm.get("hierarchies")) > 0
    
    # Check physical file creation
    bm_file = Path(get_output_path("business_model.json"))
    assert bm_file.exists()
    
    print("\nRunning a multi-stage Decision Intelligence query...")
    query = "Why did Sales_Revenue decrease in North India?"
    insight = run_omega(query, df)
    
    print("\n--- Query Insight Result ---")
    print(f"Insight Text: {insight.get('insight_text')}")
    print(f"Key Metric: {insight.get('key_metric')}")
    print(f"Strategies Recommended:")
    for s in insight.get("strategies", []):
        print(f" - {s}")
    print(f"Priority Matrix:")
    for item in insight.get("priority_matrix", []):
        print(f" - {item.get('action')}: Impact={item.get('impact')}, Effort={item.get('effort')}")
        
    assert insight.get("insight_text") is not None
    assert len(insight.get("strategies")) >= 3
    assert len(insight.get("priority_matrix")) > 0
    assert len(insight.get("risks")) > 0
    print("\nAll Decision Intelligence tests passed successfully!")

if __name__ == "__main__":
    test_decision_intelligence()
