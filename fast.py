from fastapi import FastAPI, Request
from flask import render_template
import numpy as np
from fraud_detection.core_detection import detect_fraud
from fraud_detection.helpers import load_data_subset
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd


app = FastAPI(title="Fraud Detection API",
              description="API for detecting fraud using an autoencoder model",
              version="1.0.0")

templates = Jinja2Templates(directory="templates")

# Serve static files (JS)
app.mount("/static", StaticFiles(directory="static"), name="static")


class FraudFeatures(BaseModel):
    TransactionType: float
    Channel: float
    CustomerAge: float
    CustomerOccupation: float
    TransactionDuration: float
    LoginAttempts: float
    AccountBalance: float
    PreviousTransactionDate: float
    TransactionDate_hour: float
    TransactionWeekNumber: float
    DaysSinceLastPurchase: float

# data_path = "./dataset/synthetic_dataset.h5"
data_path = "./dataset/credit_card_fraud.h5"
# data_points = load_data_subset(data_path=data_path, n_samples=-1)
df_hdf5 = pd.read_hdf(data_path, key='fraud_dataset')



@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("interface.html", {"request": request})

@app.get("/data")
def get_data():
    return JSONResponse(df_hdf5.to_dict(orient="records"))




@app.post('/predict/')
async def predict(data: dict):


    model_path = './models/fraud_autoencoder_model.h5'



    row = np.array([list(data.values())]).astype(float)                 
    
    prediction = detect_fraud(model_path, row, threshold=2.0)

    # return fraud_dict
    return {"fraud": prediction['fraud'],
            "zscore": prediction['zscore']}
#     # Assuming the model is already loaded and ready to use
    # pred = model.predict(data_points)

# templates.TemplateResponse("result.html", 
#                                       {"request": Request, 
#                                        "fraud": False, 
#                                        "zscore": 0.0})

# if __name__ == '__main__':
    # app.run(debug=False, host='127.0.0.1', port=5000)