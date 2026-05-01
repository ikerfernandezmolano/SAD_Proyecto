# SAD_Proyecto

## 🧾 Preparación del entorno

La versión de Python a utilizar es **Python 3.12.3**

La primera vez que ejecutes el proyecto, prepara el entorno con:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Ejecución del entorno
Para realizar el proceso de clustering y análisis de tópicos (LDA):

```bash
source venv/bin/activate
python3 clustering.py -j clustering.json
```

## ⚙ Config

```json
{
    "data_file": "../Datos/datos.csv",
    "company_name": "Nombre_Empresa",
    "text_column": "columna_de_texto", 
    "sentiment_column": "columna_de_puntuacion",
    "target_sentiment": ["negative", "positive", "neutral"],
    "language": "english/spanish",
    "lda": {
        "k_min": valueMin,
        "k_max": valueMax,
        "step": valueStep
    }
}
```
- data_file: Ruta relativa al archivo CSV que contiene los datos a analizar.

- company_name: Nombre de la empresa. Se utiliza para crear la subcarpeta de salida en /output.

- text_column: Nombre de la columna que contiene las reseñas o comentarios de texto.

- sentiment_column: Nombre de la columna con la valoración numérica (el script mapea 1-2 como negativo, 3 neutro y 4-5 positivo).

- target_sentiment: Lista de sentimientos que se procesarán (ej. ["negative", "positive", "neutral"]).

- language: Idioma para el preprocesamiento de NLTK (eliminación de stopwords).

- n_gram: Nivel de agrupación de palabras. Puede ser "unigram" (palabras sueltas), "bigram" (pares de palabras, ej. apple_music) o "trigram" (grupos de tres).

- lda: Configuración para la búsqueda del número óptimo de tópicos:

    - k_min: Número mínimo de tópicos a evaluar.

    - k_max: Número máximo de tópicos a evaluar.

    - step: Incremento en el número de tópicos por cada iteración.