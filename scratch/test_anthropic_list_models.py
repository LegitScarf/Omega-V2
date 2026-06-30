import os
import sys
import httpx
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def list_models():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv(project_root / ".env")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found.")
        return
        
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    try:
        url = "https://api.anthropic.com/v1/models"
        print(f"Querying {url}...")
        response = httpx.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(response.text)
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    list_models()
