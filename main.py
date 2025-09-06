from fastapi import FastAPI
from pydantic import BaseModel
import pickle  # Use pickle for consistency with notebook saving
import pandas as pd
from predict import extract_features  # Assumes predict.py is in the same directory

# Load the bundle (dict containing model, feature_names, class_mapping)
with open("rf_url_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle['model']  # Extract the RandomForestClassifier
feature_names = bundle['feature_names']  # For aligning input features
class_mapping = bundle['class_mapping']  # For mapping indices to class names (e.g., 0: 'phishing')

# FastAPI app
app = FastAPI(
    title="URL Classification API",
    description="Detect benign/defacement/malware/phishing URLs",
    version="1.0"
)

# Input schema
class URLRequest(BaseModel):
    url: str

@app.post("/predict")
def predict_url(request: URLRequest):
    try:
        url = request.url
        features = extract_features(url)
        if not features:
            return {"error": "Could not extract features from URL"}
        
        # Create DataFrame and align columns to match training order
        df = pd.DataFrame([features])[feature_names].fillna(0.0)
        
        # Predict
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        
        # Map to class name and probabilities
        predicted_class = class_mapping[int(pred)]
        probabilities = {class_mapping[i]: round(float(p), 4) for i, p in enumerate(proba)}
        
        return {
            "url": url,
            "prediction": predicted_class,
            "probabilities": probabilities,
            "confidence": round(float(max(proba)), 4)  # Max probability as confidence
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "URL Classification API is running"}