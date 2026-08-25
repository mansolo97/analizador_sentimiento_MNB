
# Clasificador Híbrido de Sentimiento en Reseñas

Este proyecto es un sistema de análisis de sentimiento para reseñas en español que utiliza una **arquitectura híbrida**:

1. **Modelado Local (Machine Learning Clásico):** Un clasificador **Multinomial Naive Bayes** entrenado con TF-IDF, determina si un texto tiene un sentimiento POSITIVO o NEGATIVO.
2. **LLM en la Nube (Google Gemini):** Cuando se detecta una reseña **negativa**, el sistema se enruta a la API de Gemini (`gemini-3.6-flash`) para redactar una respuesta de atención al cliente personalizada, empática y profesional.

---

## Requisitos Previos

* **Python 3.10+** instalado.
* Una **API Key de Google Gemini** ([Obtener aquí](https://aistudio.google.com/)).
* El dataset de entrenamiento descargado de Kaggle.

---

## Estructura del Proyecto

```text
.
├── dataset/
│   └── IMDB Dataset SPANISH.csv   # Dataset descargado
├── models/
│   ├── modelo_naive_bayes.joblib  # Modelo entrenado
│   └── vectorizador_tfidf.joblib  # Vectorizador TF-IDF
├── scripts/
│   ├── entrenar_guardar.py        # Script para entrenar el modelo
│   └── app.py                     # Script principal / Inferencia
│   └── app_fastapi.py             # Script para generar un endpoint
├── .env.example                   # Plantilla de variables de entorno
├── .env                           # Configuración local (no subir a git)
└── requirements.txt               # Dependencias de Python

```

---

## 1. Descarga del Dataset

El modelo se entrena utilizando el dataset en español de reseñas de IMDB:

* **Link de descarga:** [IMDB Dataset of 50K Movie Reviews (Spanish) en Kaggle](https://www.kaggle.com/datasets/luisdiegofv97/imdb-dataset-of-50k-movie-reviews-spanish/data)

Crea una carpeta llamada `dataset/` en la raíz del proyecto y coloca ahí el archivo `IMDB Dataset SPANISH.csv`.

---

## 2. Instalación y Configuración

### Pasos de instalación:

1. **Clonar el repositorio:**
```bash
git clone https://github.com/mansolo97/analizador_sentimiento_MNB.git
cd analizador_sentimiento_MNB

```


2. **Crear y activar un entorno virtual:**
```bash
python -m venv venv

# En Windows (PowerShell):
.\venv\Scripts\activate

# En Linux/macOS:
source venv/bin/activate

```


3. **Instalar dependencias:**
```bash
pip install -r requirements.txt

```


4. **Configurar variables de entorno:**
Copia el archivo `.env.example` para crear tu `.env`:
```bash
cp .env.example .env

```


Abre `.env` y coloca tu API Key de Gemini:
```env
GEMINI_API_KEY=tu_api_key_aqui

```

---

## 3. Uso del Proyecto

### Paso A: Entrenar el Modelo Local

Para generar los archivos `.joblib` en la carpeta `models/`, ejecuta el script de entrenamiento:

```bash
cd scripts
python entrenar_guardar.py

```

> **Resultado esperado:**
> * `¡Modelo clásico entrenado con éxito!`
> * Métricas de exactitud alcanzadas en el set de prueba (~86-88%).
> * Creación de `modelo_naive_bayes.joblib` y `vectorizador_tfidf.joblib`.
> 
> 

### Paso B: Ejecutar la Inferencia

Para probar el flujo del analizador híbrido:

```bash
python app.py

```

---

## Flujo de Trabajo (Workflow)

```text
[ Reseña de Usuario ]
         │
         ▼
[ Vectorizador TF-IDF ]
         │
         ▼
[ Multinomial Naive Bayes ] ──(¿Es Positiva?)──► [ RESPUESTA ESTÁNDAR ]
         │                                     
    (Es Negativa)
         │
         ▼
[ Google Gemini API ] ────────────────────────► [ DISCULPA PERSONALIZADA ]
                                                (Usa cuota de API)

```

---

## Dependencias Principales

* `scikit-learn` - Vectorización TF-IDF y modelo Naive Bayes.
* `pandas` - Carga y tratamiento de datasets.
* `joblib` - Persistencia de modelos en disco.
* `google-genai` - SDK oficial de Google GenAI para Gemini.
* `python-dotenv` - Carga de variables de entorno desde `.env`.