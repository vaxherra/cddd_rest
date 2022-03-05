import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from utils import get_model, smiles_to_embedding


app = FastAPI(title="Predicting Continuous and Data-Driven" \
                    "Descriptors (CDDD) from SMILE IDs")


class SMILE(BaseModel):
    """Represent input as a list of batches containing smiles"""
    batches: list


@app.on_event("startup")
def load_model():
    """ Load model on server startup"""
    global model
    model = get_model()


@app.get("/")
def home():
    """Just provide a simple link to API docs"""
    return "Continuous and Data-Driven Descriptors (CDDD) API.  "\
        "For documenation head over to http://localhost:80/docs"


@app.post("/predict")
def predict(smile: SMILE):
    """Given a list containing """
    batches = smile.batches
    batches = [item for sublist in batches for item in sublist]

    preds = smiles_to_embedding(batches, model)
    preds = preds.to_json()  # serialize pandas DF to send over REST

    return {"Prediction": preds}
