import sys
import os

# Add the root project dir to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steps.specs import fetch_and_calculate_specs
import json

def test_tool():
    print("Testing the Equipment Fetcher Tool (DuckDuckGo + Local DB Fallback)...")
    specs = fetch_and_calculate_specs(num_aps=3)
    print("\nResulting Specs:")
    print(json.dumps(specs, indent=2))

if __name__ == "__main__":
    test_tool()
