from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import logging
import gdown
import os
import time
from predict import extract_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download model from Google Drive with retries
MODEL_URL = "https://drive.google.com/file/d/1OtSbRvuSn1XvcUtaKYEIoBXMnybBZmcG/view?usp=sharing"
MODEL_PATH = "/tmp/rf_url_model.pkl"  # Use /tmp for Render's ephemeral storage

def download_model(url, path, retries=3, delay=5):
    for attempt in range(retries):
        try:
            logger.info(f"Attempt {attempt + 1} to download model from Google Drive...")
            gdown.download(url, path, quiet=False)
            logger.info("Model downloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Download attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
    logger.error("All download attempts failed")
    return False

if not os.path.exists(MODEL_PATH):
    if not download_model(MODEL_URL, MODEL_PATH):
        raise Exception("Failed to download model after retries")

try:
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle['model']
    feature_names = bundle['feature_names']
    class_mapping = bundle['class_mapping']
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

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
