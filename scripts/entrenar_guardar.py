#!/usr/bin/env python3
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import sys

# ==========================================
# PARTE 1: MACHINE LEARNING CLÁSICO
# ==========================================
def entrenar_MNB():

    # Datos de entrenamiento reseñas IMDB 
    try:
        df = pd.read_csv("../dataset/IMDB Dataset SPANISH.csv",encoding='utf-8')

    except Exception as e:
        print(f" Error al cargar el dataset: {e}")
        sys.exit()

    #Convertimos positivo a 1 y negativo a 0
    df['sentimiento_binario'] = (df['sentiment'] == 'positive').astype(int)
    #print(df['review_es'])
    #sys.exit()

    #
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['review_es'], 
        df['sentimiento_binario'],
        test_size=0.20, 
        random_state=42, 
        stratify=df['sentimiento_binario']
        )

    # Convertimos el texto a números usando TF-IDF
    vectorizador = TfidfVectorizer(max_features=25000,
                                    ngram_range=(1,2), #N-gramas para que el vectorizador tome parejas de palabras
                                    min_df=2,
                                    stop_words=['el', 'la', 'los', 'las', 'de', 'del', 'un', 'una', 'y', 'en', 'con', 'por', 'para', 'mis', 'sus']
                                   )
    X_train = vectorizador.fit_transform(X_train_raw)
    X_test = vectorizador.transform(X_test_raw)

    # Entrenamos el modelo matemático (Naive Bayes)
    modelo_ml = MultinomialNB()
    modelo_ml.fit(X_train, y_train)

    print("¡Modelo clásico entrenado con éxito!")

    predicciones = modelo_ml.predict(X_test)
    print(f"Exactitud en prueba: {accuracy_score(y_test, predicciones) * 100:.2f}%")

    joblib.dump(modelo_ml, '../models/modelo_naive_bayes.joblib')
    joblib.dump(vectorizador, '../models/vectorizador_tfidf.joblib')

if __name__ == "__main__":
    entrenar_MNB()
    
