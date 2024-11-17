from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Cargar el modelo y el scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Cargar los LabelEncoders
label_encoders = {
    'Primary Type': joblib.load('label_encoder_primary_type.pkl'),
    'Location Description': joblib.load('label_encoder_location_description.pkl')
}

# Definir el esquema de los datos de entrada
class PredictionRequest(BaseModel):
    Primary_Type: str
    Location_Description: str
    Arrest: bool
    Day_Date: int
    District: int
    Beat: int

# Definir el endpoint para las predicciones
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Convertir los datos de entrada en un DataFrame
        input_data = pd.DataFrame([request.dict()])

        # Ajustar los nombres de las columnas para que coincidan con los nombres originales
        input_data.rename(columns={
            'Primary_Type': 'Primary Type',
            'Location_Description': 'Location Description',
            'Day_Date': 'Day Date'
        }, inplace=True)

        # Transformar las variables categóricas usando los LabelEncoders cargados previamente
        input_data['Primary Type'] = label_encoders['Primary Type'].transform(input_data['Primary Type'].astype(str))
        input_data['Location Description'] = label_encoders['Location Description'].transform(input_data['Location Description'].astype(str))

        # Escalar los datos
        input_data_scaled = scaler.transform(input_data)

        # Hacer la predicción
        prediction = model.predict(input_data_scaled)
        
        # Devolver la predicción como respuesta
        return {"Domestic": bool(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
