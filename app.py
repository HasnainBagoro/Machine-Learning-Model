from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import logging
import requests
import os
from predict import extract_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download model from Google Drive
MODEL_URL = "https://drive.google.com/file/d/1OtSbRvuSn1XvcUtaKYEIoBXMnybBZmcG/view?usp=sharing"
MODEL_PATH = "rf_url_model.pkl"

if not os.path.exists(MODEL_PATH):
    logger.info("Downloading model from Google Drive...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    logger.info("Model downloaded successfully")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
model = bundle['model']
feature_names = bundle['feature_names']
class_mapping = bundle['class_mapping']

app = FastAPI(title="URL Classification API", description="Detect benign/defacement/malware/phishing URLs", version="1.0")

class URLRequest(BaseModel):
    url: str

@app.post("/predict")
def predict_url(request: URLRequest):
    try:
        url = request.url
        logger.info(f"Processing URL: {url}")
        features = extract_features(url)
        if not features:
            logger.error("Feature extraction failed")
            return {"error": "Could not extract features from URL"}
        df = pd.DataFrame([features])[feature_names].fillna(0.0)
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        predicted_class = class_mapping[int(pred)]
        probabilities = {class_mapping[i]: round(float(p), 4) for i, p in enumerate(proba)}
        return {
            "url": url,
            "prediction": predicted_class,
            "probabilities": probabilities,
            "confidence": round(float(max(proba)), 4)
        }
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "URL Classification API is running"}
