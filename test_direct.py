import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, DummyModel
import json

def test_dummy_model():
    """Test the dummy model directly."""
    print("Testing DummyModel...")
    import numpy as np
    
    dummy = DummyModel()
    test_input = np.array([[1.5, 2.0, 3.5, 4.0, 5.5, 6.0]])
    
    prediction = dummy.predict(test_input)
    print(f"Dummy model prediction: {prediction}")
    
    probabilities = dummy.predict_proba(test_input)
    print(f"Dummy model probabilities: {probabilities}")

def test_api_endpoint():
    """Test the API endpoint using Flask test client."""
    print("\nTesting API endpoint...")
    
    with app.test_client() as client:
        # Test health endpoint
        response = client.get('/health')
        print(f"Health check: {response.status_code}")
        print(f"Health response: {json.dumps(response.get_json(), indent=2)}")
        
        # Test prediction endpoint
        response = client.get('/prediction/tokyo/1.5/2.0/3.5/4.0/5.5/6.0')
        print(f"\nPrediction status: {response.status_code}")
        print(f"Prediction response: {json.dumps(response.get_json(), indent=2)}")

if __name__ == "__main__":
    test_dummy_model()
    test_api_endpoint()