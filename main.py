import io
import joblib


import numpy as np
import pandas as pd
from pydantic import BaseModel


from conv import Conv_AE
import inference

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

class Model(BaseModel):
    X: list[str]

app = FastAPI()

app.mount("/static", StaticFiles(directory="public"))

# load the model from disk
filename = './models/model.pkl'
loaded_model = joblib.load(filename)

@app.get("/")
def read_root():

    return RedirectResponse("/static/index.html")

@app.post("/predict")
def predict_model(model:Model):
    string_data="""datetime;Accelerometer1RMS;Accelerometer2RMS;Current;Pressure;Temperature;Thermocouple;Voltage;Volume Flow RateRMS
""" + '\n'.join(model.X)
    df = pd.read_csv(io.StringIO(string_data), sep=";", index_col="datetime", parse_dates=True)
    result = inference.model_inference(df, loaded_model, 60)
    return {"result": ';'.join(map(str, result))}

def main():
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
