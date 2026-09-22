import os
import sys
import joblib
from google import genai
from dotenv import load_dotenv

from entrenar_guardar import entrenar_MNB

load_dotenv('../.env')
GEMINI_API_KEY = os.getenv("API_KEY")

# ==========================================
# PARTE 2 - CARGA DEL MODELO YA ENTRENADO
# ==========================================
ruta_modelo = '../models/modelo_naive_bayes.joblib'
ruta_vectorizador = '../models/vectorizador_tfidf.joblib'

#Comprueba si existe el archivo del modelo y vectorizador
#Si no existen entrena al modelo y guarda los archivos
if not os.path.exists(ruta_modelo) or not os.path.exists(ruta_vectorizador):
    print("No se encontro archivos de modelo y vectorizador")
    print("Entrenando modelo..")
    entrenar_MNB()
    print("Modelo entrenado con exito")
    

modelo_ml = joblib.load(ruta_modelo)
vectorizador = joblib.load(ruta_vectorizador)

print("Modelo y Vectorizador cargados.\n")

# ==========================================
# PARTE 3 - PIPELINE CON IA GENERATIVA
# ==========================================

def procesar_nueva_reseña(nueva_reseña):
    # Predicción con el modelo cargado
    reseña_vectorizada = vectorizador.transform([nueva_reseña])
    prediccion = modelo_ml.predict(reseña_vectorizada)[0]

    # Muestra la probabilidad de [Negativo (0), Positivo (1)]
    probabilidades = modelo_ml.predict_proba(reseña_vectorizada)[0]
    
    
    sentimiento_texto = "POSITIVO" if prediccion == 1 else "NEGATIVO"

   
    print(f"\n=== ANÁLISIS DE SISTEMA ===")
    print(f"Probabilidades -> Negativo: {probabilidades[0]:.2%}, Positivo: {probabilidades[1]:.2%}")
    print(f"Reseña recibida: '{nueva_reseña}'")
    print(f"Predicción ML: {sentimiento_texto}")

    # Para una buena reseña envía una respuesta genérica
    if prediccion == 1:
        print("\n=== RESPUESTA GENÉRICA ===")
        return (
            "¡Hola! ¡Muchísimas gracias por tu comentario! Nos alegra enormemente saber que todo estuvo excelente. "
            "\nTe invitamos a seguir muy pendiente de nuestras redes sociales para que no te pierdas ninguna de nuestras próximas novedades y sorpresas. ¡Nos encanta tenerte con nosotros!"
            "\n\nCon entusiasmo,"
            "**\nEl equipo de Atención de Empresas Patito**"
        )

    # Para una mala reseña genera una respuesta personalizada con ayuda de un LLM
    instruccion_ia = (
            "El cliente dejó una reseña NEGATIVA. Redacta una respuesta pidiendo disculpas formales, "
            "muestra empatía y ofrécele un cupón de 15% de descuento para su siguiente compra. "
            "El mensaje debe estar a nombre del equipo de atención de empresas patito."
        )

    prompt_final = f"{instruccion_ia}\n\nReseña del cliente: \"{nueva_reseña}\""
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt_final,
            )
        print("\n=== RESPUESTA GENERADA POR IA ===")

        return response.text
        
    except Exception as e:
        print(f"\nError al conectar con Gemini: {e}")

if __name__ == "__main__":

    #===========================
    # PRUEBA RÁPIDA
    #===========================
    
    print("--- CASO 1: Reseña Positiva ---")
    reseña_positiva = "El producto llegó a tiempo y la calidad es excelente, muy contento con la compra."
    print(procesar_nueva_reseña(reseña_positiva))
    

    print("\n" + "="*50 + "\n")

    print("--- CASO 2: Reseña Negativa ---")
    reseña_negativa ="No me gusto el producto, es de muy mala calidad, no vale la pena"
    print(procesar_nueva_reseña(reseña_negativa))