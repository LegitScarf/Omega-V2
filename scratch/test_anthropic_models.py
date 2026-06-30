import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from anthropic import Anthropic

def test_models():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Try loading from .env
        from dotenv import load_dotenv
        load_dotenv(project_root / ".env")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment or .env file.")
        return
        
    client = Anthropic(api_key=api_key)
    
    # List of candidate models to try
    models_to_test = [
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-20241022",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229"
    ]
    
    print("Testing Anthropic models access...")
    for model in models_to_test:
        try:
            print(f"Testing '{model}'...", end=" ", flush=True)
            response = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Say hi"}],
            )
            print(f"-> SUCCESS! Response: {response.content[0].text.strip()}")
        except Exception as e:
            print(f"-> FAILED: {e}")

if __name__ == "__main__":
    test_models()
