# 1. Library imports
import pandas as pd
import uvicorn
from fastapi import FastAPI
import joblib
from pydantic import BaseModel

# Récupération des modèle et base d'échantillons clients
perfect_model = joblib.load("pipeline_lgbm.joblib")
echantillon_clients = pd.read_csv("data/echantillon_clients.csv", index_col="SK_ID_CURR")
echantillon_clients = echantillon_clients.drop(columns=["threshold"])

# Class which describes ID client
class ID_client(BaseModel):
    client_choice: int

app = FastAPI()
@app.post("/invocations")
def operate(input:ID_client):
    client_choice = input.client_choice
    data_client = echantillon_clients.loc[echantillon_clients.index == client_choice]
    proba = perfect_model.predict_proba(data_client)[0][1]
    return proba

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
