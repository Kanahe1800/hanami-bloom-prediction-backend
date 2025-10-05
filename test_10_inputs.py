import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
import json

def test_10_inputs():
    """Test the API endpoint with 10 inputs."""
    print("Testing API endpoint with 10 inputs...")
    
    with app.test_client() as client:
        # Test health endpoint
        response = client.get('/health')
        print(f"Health check: {response.status_code}")
        
        # Test prediction endpoint with 10 inputs
        inputs = "1.0/2.0/3.0/4.0/5.0/6.0/7.0/8.0/9.0/10.0"
        response = client.get(f'/prediction/tokyo/{inputs}')
        print(f"\nPrediction status: {response.status_code}")
        print(f"Prediction response: {json.dumps(response.get_json(), indent=2)}")
        
        # Test with wrong number of inputs (should fail)
        wrong_inputs = "1.0/2.0/3.0/4.0/5.0/6.0"  # Only 6 inputs
        response = client.get(f'/prediction/tokyo/{wrong_inputs}')
        print(f"\nWrong inputs test status: {response.status_code}")
        print(f"Wrong inputs response: {json.dumps(response.get_json(), indent=2)}")
        
        # Test models endpoint
        response = client.get('/models')
        print(f"\nModels endpoint status: {response.status_code}")
        print(f"Models response: {json.dumps(response.get_json(), indent=2)}")

if __name__ == "__main__":
    test_10_inputs()