# Hanami Bloom Prediction API

A Flask-based REST API for predicting hanami (cherry blossom) bloom dates using machine learning models.

## Features

- **Dynamic Input Handling**: Automatically detects the number of features required by each city's model
- **Multiple City Support**: Load different models for different cities
- **Robust Error Handling**: Graceful fallbacks when models fail to load
- **Health Monitoring**: Built-in health check and model status endpoints
- **Cross-Version Compatibility**: Handles different scikit-learn versions automatically

## Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the API:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

### API Endpoints

#### 1. Make Predictions
```
GET /prediction/<city_name>/<input1>/<input2>/.../<input10>
```

**Example:**
```
GET /prediction/tokyo/1.0/2.0/3.0/4.0/5.0/6.0/7.0/8.0/9.0/10.0
```

**Response:**
```json
{
  "status": "success",
  "city": "tokyo",
  "inputs": {
    "input1": 1.0,
    "input2": 2.0,
    ...
    "input10": 10.0
  },
  "prediction": [12.076272727272727],
  "expected_features": 10,
  "message": "Prediction successful for tokyo"
}
```

#### 2. Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "hanami-bloom-prediction-api",
  "loaded_models": ["tokyo"]
}
```

#### 3. List Available Models
```
GET /models
```

**Response:**
```json
{
  "status": "success",
  "available_models": ["tokyo"],
  "loaded_models": ["tokyo"]
}
```

#### 4. API Documentation
```
GET /
```

Returns API documentation and usage examples.

## Model Requirements

- Models should be saved as `.pkl` files in the `models/` directory
- Model filename determines the city name (e.g., `tokyo.pkl` → city name "tokyo")
- Models must implement `predict()` method
- The API automatically detects the number of input features required

## Supported Model Formats

The API supports multiple model loading strategies:
1. **Standard pickle** - Default Python pickle format
2. **Joblib** - Scikit-learn's preferred format
3. **Pickle5** - For compatibility with older versions
4. **Dummy Model** - Fallback for testing when real models fail

## Error Handling

The API provides detailed error messages for common issues:

- **404**: Model not found for specified city
- **400**: Invalid input parameters (wrong number of inputs, non-numeric values)
- **500**: Internal server errors during prediction

## Testing

Run the test scripts to verify functionality:

```bash
# Test with 10 inputs
python test_10_inputs.py

# Direct API testing
python test_direct.py
```

## Production Deployment

For production use, replace the development server with a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## File Structure

```
├── app.py                 # Main Flask application
├── models/
│   └── tokyo.pkl         # Model files
├── requirements.txt      # Python dependencies
├── test_10_inputs.py     # Test script for 10-input functionality
├── test_direct.py        # Direct API testing
└── README.md            # This file
```

## API Usage Examples

### Using curl
```bash
# Make a prediction
curl "http://localhost:5000/prediction/tokyo/1.0/2.0/3.0/4.0/5.0/6.0/7.0/8.0/9.0/10.0"

# Check health
curl "http://localhost:5000/health"

# List models
curl "http://localhost:5000/models"
```

### Using Python requests
```python
import requests

# Make a prediction
response = requests.get("http://localhost:5000/prediction/tokyo/1.0/2.0/3.0/4.0/5.0/6.0/7.0/8.0/9.0/10.0")
print(response.json())
```

## Contributing

To add support for a new city:
1. Save your trained model as `models/{city_name}.pkl`
2. The API will automatically detect and load the new model
3. Use the city name in the prediction endpoint

## Notes

- The API expects exactly 10 input features for the tokyo model
- Input validation ensures all parameters are numeric
- Models are cached in memory after first load for better performance
- Debug mode is enabled by default - disable for production use