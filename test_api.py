import requests
import json

# Base URL for the API
base_url = "http://127.0.0.1:5000"

def test_endpoint(url, description):
    """Test an API endpoint and print results."""
    print(f"\n{description}")
    print("=" * 50)
    try:
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError:
        print(f"Response: {response.text}")

def main():
    print("Testing Hanami Bloom Prediction API")
    print("=" * 50)
    
    # Test health endpoint
    test_endpoint(f"{base_url}/health", "Testing Health Endpoint")
    
    # Test models list endpoint
    test_endpoint(f"{base_url}/models", "Testing Models List Endpoint")
    
    # Test home endpoint
    test_endpoint(f"{base_url}/", "Testing Home Endpoint")
    
    # Test prediction endpoint with sample data
    sample_inputs = "1.5/2.0/3.5/4.0/5.5/6.0"
    test_endpoint(f"{base_url}/prediction/tokyo/{sample_inputs}", "Testing Prediction Endpoint")

if __name__ == "__main__":
    main()