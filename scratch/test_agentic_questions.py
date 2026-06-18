import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.crew import run_omega
from src.utils import clear_output_dir, load_json_output

def generate_mock_ev_dataset(num_rows=200):
    np.random.seed(42)
    data = {
        "age": np.random.randint(18, 80, num_rows),
        "annual_income": np.random.randint(20000, 150000, num_rows),
        "education_level": np.random.choice(["High School", "Bachelor", "Master", "PhD"], num_rows),
        "city_type": np.random.choice(["Urban", "Suburban", "Rural"], num_rows),
        "daily_commute_km": np.random.uniform(5, 120, num_rows),
        "weekly_travel_distance_km": np.random.uniform(30, 700, num_rows),
        "current_vehicle_type": np.random.choice(["Gasoline", "Hybrid", "None", "EV"], num_rows),
        "vehicle_age_years": np.random.randint(0, 15, num_rows),
        "fuel_expense_per_month": np.random.uniform(50, 400, num_rows),
        "charging_station_accessibility": np.random.randint(1, 11, num_rows),
        "nearest_charging_station_km": np.random.uniform(0.1, 25.0, num_rows),
        "home_charging_available": np.random.choice([True, False], num_rows),
        "electricity_cost_per_kwh": np.random.uniform(0.08, 0.35, num_rows),
        "environmental_awareness_score": np.random.randint(1, 11, num_rows),
        "government_incentive_awareness": np.random.randint(1, 11, num_rows),
        "technology_affinity_score": np.random.randint(1, 11, num_rows),
        "range_anxiety_score": np.random.randint(1, 11, num_rows),
        "battery_replacement_concern": np.random.randint(1, 11, num_rows),
        "ev_knowledge_score": np.random.randint(1, 11, num_rows),
        "previous_ev_experience": np.random.choice([True, False], num_rows),
        "ev_adoption_likelihood": np.random.randint(1, 11, num_rows),
        "monthly_energy_consumption_kwh": np.random.uniform(100, 800, num_rows),
        "monthly_charging_cost": np.random.uniform(10, 120, num_rows)
    }
    
    # Introduce correlation for test validity
    # High charging accessibility + high environmental awareness = high adoption likelihood
    df = pd.DataFrame(data)
    df["ev_adoption_likelihood"] = (
        (df["environmental_awareness_score"] * 0.4) + 
        (df["charging_station_accessibility"] * 0.4) + 
        (df["home_charging_available"].astype(int) * 2.0) + 
        np.random.normal(0, 1, num_rows)
    ).clip(1, 10).round().astype(int)
    
    return df

def test_query(query_text, df):
    print("\n" + "="*80)
    print(f"Testing Query: {query_text}")
    print("="*80)
    
    clear_output_dir()
    
    def mock_task_callback(task_output):
        print(f"Task Complete Callback: {task_output.name}")

    # Set OpenAI API key if not in env for debugging (fallback to standard system setup)
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        return False

    result = run_omega(
        user_query=query_text,
        dataframe=df,
        task_callback=mock_task_callback
    )
    
    print("\nExecution Results Summary:")
    print(f"Key Metric: {result.get('key_metric', 'N/A')}")
    print(f"Insight Text Summary: {result.get('insight_text', 'N/A')[:200]}...")
    
    # Verify outputs written to files
    chart_output = load_json_output("chart.json")
    query_output = load_json_output("query_result.json")
    prediction_output = load_json_output("prediction.json")
    
    print(f"Chart Generated: {chart_output is not None and chart_output.get('chart_generated', False)}")
    print(f"Query Results Extracted: {query_output is not None and query_output.get('status') == 'success'}")
    
    if chart_output and chart_output.get("chart_generated"):
        spec = chart_output.get("plotly_spec")
        print(f"Plotly spec data type: {type(spec)}")
        
    return True

if __name__ == "__main__":
    df = generate_mock_ev_dataset()
    
    # Test Q1
    test_query(
        query_text="What are the strongest factors influencing EV adoption likelihood? Find the correlation of ev_adoption_likelihood against age, annual_income, charging_station_accessibility, environmental_awareness_score, and range_anxiety_score, and show a ranked bar chart.",
        df=df
    )
    
    # Test Q4
    test_query(
        query_text="What is the optimal charging station coverage required to maximize EV adoption? Group nearest_charging_station_km into 5km bins, calculate the average ev_adoption_likelihood for each bin, and identify the threshold where adoption likelihood declines steepest.",
        df=df
    )
    
    # Test Q10
    test_query(
        query_text="What customer profiles exhibit high environmental awareness but low EV adoption likelihood? Filter records where environmental_awareness_score > 7 and ev_adoption_likelihood < 4, and list their top characteristics compared to the general dataset.",
        df=df
    )
