import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

load_dotenv('../.env')
GEMINI_API_KEY = os.getenv("API_KEY")

app = FastAPI()

# Cargar modelos en memoria RAM una sola vez al arrancar
modelo = joblib.load('../models/modelo_naive_bayes.joblib')
vectorizador = joblib.load('../models/vectorizador_tfidf.joblib')
client = genai.Client(api_key=GEMINI_API_KEY)

class ReseñaInput(BaseModel):
    texto: str

@app.post("/analizar")
def analizar(data: ReseñaInput):
    prediccion = modelo.predict(vectorizador.transform([data.texto]))[0]
    
    if prediccion == 1:
        return {"sentimiento": "POSITIVO", "respuesta": "¡Muchas gracias por tus comentarios!"}
    
    prompt = f"Redacta una disculpa breve y empática para esta reseña negativa: '{data.texto}'"
    res = client.models.generate_content(model='gemini-3.6-flash', contents=prompt)
    
    return {"sentimiento": "NEGATIVO", "respuesta": res.text}