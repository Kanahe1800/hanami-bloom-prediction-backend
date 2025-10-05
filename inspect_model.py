import pickle
import os

# Load the model to understand its structure
model_path = "models/tokyo.pkl"
print(f"Loading model from: {model_path}")

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model type: {type(model)}")
    print(f"Model: {model}")
    
    # Try to get model attributes if it's a scikit-learn model
    if hasattr(model, 'feature_names_in_'):
        print(f"Feature names: {model.feature_names_in_}")
    if hasattr(model, 'n_features_in_'):
        print(f"Number of features: {model.n_features_in_}")
    if hasattr(model, 'classes_'):
        print(f"Classes: {model.classes_}")
    
    # Check if it has predict method
    if hasattr(model, 'predict'):
        print("Model has predict method")
    if hasattr(model, 'predict_proba'):
        print("Model has predict_proba method")
        
    # Try a test prediction with 6 dummy inputs
    import numpy as np
    test_input = np.array([[1, 2, 3, 4, 5, 6]])
    try:
        prediction = model.predict(test_input)
        print(f"Test prediction with [1,2,3,4,5,6]: {prediction}")
    except Exception as pred_error:
        print(f"Error making test prediction: {pred_error}")
        
except Exception as e:
    print(f"Error loading model: {e}")