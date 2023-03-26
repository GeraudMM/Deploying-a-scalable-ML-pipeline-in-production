# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import yaml

from ml.data import process_data
from ml.model import inference
from train_model import cat_features

with open('params.yaml', encoding='utf-8') as f:
    params = yaml.safe_load(f)

# Open file in binary read mode with utf-8 encoding
with open(params['model_path'], 'rb') as f:
    # Load data from file with utf-8 encoding
    model_config = pickle.load(f, encoding='utf-8')

app = FastAPI()

class InputData(BaseModel):
    age: int = Field(..., example="42")
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example="77516")
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example="13")
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Sales")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="Amer-Indian-Eskimo")
    sex: str = Field(..., example="Female")
    capital_gain: int = Field(..., example="2174")
    capital_loss: int = Field(..., example="0")
    hours_per_week: int = Field(..., example="35")
    native_country: str = Field(..., example="Germany")


@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI model inference API!"}

@app.post("/predict")
async def predict(input_data: InputData):    
    print(InputData)
    #breakpoint()
    input_df = pd.DataFrame([input_data.dict()])
    input_df.rename(lambda x: x.replace("_", "-"), axis="columns", inplace=True)
    # Prepare the input data for prediction
    X, _, _, _ = process_data(input_df, categorical_features=cat_features, label=None, encoder=model_config['encoder'], lb=model_config['lb'], training=False)
    
    # Make the prediction
    y_pred = inference(model_config['model'], X)

    # Return the predicted species
    return {"predicted_output": str(y_pred[0])}