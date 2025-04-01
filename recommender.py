from fastapi import FastAPI
import joblib
import requests
import pandas as pd

app = FastAPI()

# Load pre-trained models
collab_model = joblib.load("collaborative_model.sav")
content_model = joblib.load("content_model.sav")

AZURE_ML_ENDPOINT = "https://your-azureml-endpoint.com/score"
AZURE_ML_API_KEY = "your_api_key"

@app.get("/")
def root():
    return {"message": "Recommendation API is running"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    # Get recommendations from each model
    collab_recs = get_collab_recommendations(user_id)
    content_recs = get_content_recommendations(user_id)
    azure_recs = get_azure_ml_recommendations(user_id)

    return {
        "collaborative": collab_recs,
        "content": content_recs,
        "azure_ml": azure_recs
    }

def get_collab_recommendations(user_id):
    """Retrieve recommendations from collaborative filtering model."""
    try:
        user_vector = collab_model.get(user_id, [])
        return user_vector[:5]  # Return top 5 recommendations
    except:
        return []

def get_content_recommendations(user_id):
    """Retrieve recommendations from content filtering model."""
    try:
        item_vector = content_model.get(user_id, [])
        return item_vector[:5]  # Return top 5 recommendations
    except:
        return []

def get_azure_ml_recommendations(user_id):
    """Call Azure ML endpoint for recommendations."""
    headers = {"Authorization": f"Bearer {AZURE_ML_API_KEY}"}
    payload = {"user_id": user_id}

    response = requests.post(AZURE_ML_ENDPOINT, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("recommendations", [])[:5]
    return []
