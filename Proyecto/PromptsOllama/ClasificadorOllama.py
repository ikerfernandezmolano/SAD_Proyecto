import os
import pandas as pd
import requests
from collections import Counter
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma2:2b"   # puedes cambiar a llama3.1:8b

PROMPT_TEMPLATE = """Clasifica el sentimiento del siguiente comentario sobre una aplicación de música.
Las únicas etiquetas válidas son: Positivo, Negativo, Neutro.
Responde únicamente con una de esas tres palabras.

Comentario: "{texto}"
Respuesta:"""

VALID_LABELS = {
    "positivo": "Positivo",
    "negativo": "Negativo",
    "neutro": "Neutro"
}


# ------------------------------
# Llamada a Ollama
# ------------------------------
def call_ollama(prompt: str, model: str = MODEL) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 3
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"].strip()


# ------------------------------
# Normalizar salida del modelo
# ------------------------------
def normalize_label(text: str) -> str:
    cleaned = text.strip().lower()

    for ch in [".", ",", ":", ";", '"', "'", "(", ")", "\n"]:
        cleaned = cleaned.replace(ch, "")

    if cleaned in VALID_LABELS:
        return VALID_LABELS[cleaned]

    for key, value in VALID_LABELS.items():
        if key in cleaned:
            return value

    return "DESCONOCIDO"


# ------------------------------
# Clasificar un comentario
# ------------------------------
def classify_review(review_text: str):
    prompt = PROMPT_TEMPLATE.format(texto=review_text)
    raw_output = call_ollama(prompt)
    label = normalize_label(raw_output)
    return label, raw_output


# ------------------------------
# Lectura robusta del CSV
# ------------------------------
def cargar_csv(ruta_csv):
    try:
        # intento automático (mejor opción)
        df = pd.read_csv(
            ruta_csv,
            sep=None,
            engine="python",
            encoding="utf-8",
            on_bad_lines="skip"
        )
    except:
        # fallback por si hay problemas de encoding
        df = pd.read_csv(
            ruta_csv,
            sep=None,
            engine="python",
            encoding="latin1",
            on_bad_lines="skip"
        )

    print("\nColumnas detectadas:", df.columns.tolist())
    print(df.head(3))
    return df


# ------------------------------
# MAIN
# ------------------------------
def main():
    print("=== Clasificación generativa de comentarios ===\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    datos_dir = os.path.join(base_dir, "..", "Datos")

    if not os.path.exists(datos_dir):
        print(f"No se encontró la carpeta Datos en: {datos_dir}")
        return

    archivos_csv = [f for f in os.listdir(datos_dir) if f.endswith(".csv")]

    if not archivos_csv:
        print("No hay archivos CSV en la carpeta Datos.")
        return

    print("Archivos disponibles en Datos:")
    for i, archivo in enumerate(archivos_csv, start=1):
        print(f"{i}. {archivo}")

    # ------------------------------
    # Selección de archivo
    # ------------------------------
    try:
        opcion = int(input("\nSelecciona el número del archivo CSV: "))
        archivo_elegido = archivos_csv[opcion - 1]
    except:
        print("Selección no válida.")
        return

    ruta_csv = os.path.join(datos_dir, archivo_elegido)

    # ------------------------------
    # Leer CSV (robusto)
    # ------------------------------
    df = cargar_csv(ruta_csv)

    if "review" not in df.columns:
        print("\n❌ ERROR: No se encontró la columna 'review'")
        return

    total = len(df)
    print(f"\nEl archivo tiene {total} comentarios.")

    # ------------------------------
    # Número de muestras
    # ------------------------------
    try:
        n = int(input("¿Cuántos comentarios quieres analizar aleatoriamente? "))
    except:
        print("Entrada inválida.")
        return

    if n <= 0 or n > total:
        print("Número fuera de rango.")
        return

    # ------------------------------
    # Muestreo aleatorio
    # ------------------------------
    muestra = df.sample(n=n, random_state=42).copy()

    predicciones = []
    salidas = []

    print("\nClasificando comentarios...\n")

    for i, review_text in enumerate(muestra["review"].fillna("").astype(str)):

        try:
            pred, raw = classify_review(review_text)
        except Exception as e:
            pred = "DESCONOCIDO"
            raw = str(e)

        predicciones.append(pred)
        salidas.append(raw)

        print(f"{i+1}/{n} -> {pred}")

        time.sleep(0.1)

    muestra["prediccion"] = predicciones
    muestra["salida_modelo"] = salidas

    # ------------------------------
    # Resumen
    # ------------------------------
    conteo = Counter(predicciones)

    print("\n=== RESUMEN ===")
    print(f"Positivos: {conteo.get('Positivo', 0)}")
    print(f"Neutros:   {conteo.get('Neutro', 0)}")
    print(f"Negativos: {conteo.get('Negativo', 0)}")
    print(f"Desconocidos: {conteo.get('DESCONOCIDO', 0)}")

    # ------------------------------
    # Guardar resultados
    # ------------------------------
    salida_path = os.path.join(base_dir, f"resultado_{n}_{archivo_elegido}")
    muestra.to_csv(salida_path, index=False, encoding="utf-8")

    print(f"\nArchivo guardado en:\n{salida_path}")


if __name__ == "__main__":
    main()