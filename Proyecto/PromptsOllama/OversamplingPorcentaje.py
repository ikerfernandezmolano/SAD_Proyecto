import os
import time
import uuid
import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:latest"

OVERSAMPLING_PERCENTAGE = 0.50
NUM_CLASSES_TO_OVERSAMPLE = 3


def score_to_label(score: int) -> str:
    if score in [1, 2]:
        return "Negativo"
    elif score == 3:
        return "Neutro"
    elif score in [4, 5]:
        return "Positivo"
    return "Desconocido"


def cargar_csv(ruta_csv):
    try:
        return pd.read_csv(ruta_csv, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
    except:
        return pd.read_csv(ruta_csv, sep=None, engine="python", encoding="latin1", on_bad_lines="skip")


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "num_predict": 700
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["response"].strip()


def limpiar_parafrases(output: str):
    lineas = []
    for linea in output.splitlines():
        linea = linea.strip()
        linea = linea.strip("-•0123456789. )(").strip()
        if linea:
            lineas.append(linea)
    return lineas


def construir_prompt(comentarios, etiqueta, num_parafrases):
    texto = "\n".join([f"- {c}" for c in comentarios])

    return f"""Generate {num_parafrases} paraphrases of the following comments while keeping exactly the same sentiment.

IMPORTANT:
- Output MUST be in the SAME LANGUAGE as the input comments.
- DO NOT translate.
- If comments are in English, output must be in English.

Sentiment: {etiqueta}

Requirements:
- Sound like real user reviews.
- Do not copy text literally.
- Keep the same meaning.
- Return ONLY {num_parafrases} lines.
- No explanations.
- No numbering.
- No offensive content.

Original comments:
{texto}
"""

#     return f"""Generate {num_parafrases} paraphrases of the following comments while keeping exactly the same sentiment.
#
# IMPORTANT:
# - Output MUST be in the SAME LANGUAGE as the input comments.
# - DO NOT translate.
#
# Sentiment: {etiqueta}
#
# Here are examples of good paraphrasing:
#
# Original: "This app is amazing, I love it"
# Paraphrase: "I really enjoy this app, it's fantastic"
#
# Original: "The app crashes all the time"
# Paraphrase: "It keeps crashing constantly, very frustrating"
#
# Original: "It's okay, nothing special"
# Paraphrase: "It's fine but doesn't stand out much"
#
# Now generate paraphrases:
#
# Requirements:
# - Sound natural
# - Keep meaning
# - Do not copy
# - No explanations
# - One sentence per line
#
# Original comments:
# {texto}
# """


def generar_review_id():
    return "gen_" + str(uuid.uuid4())[:8]


def generar_filas_para_score(df, score_objetivo, cantidad_generar):
    subset = df[df["score"] == score_objetivo].copy()

    if subset.empty or cantidad_generar <= 0:
        return []

    etiqueta = score_to_label(score_objetivo)

    resultados = []
    generadas = 0
    lote = 1

    ejemplos_por_prompt = min(5, len(subset))
    parafrases_por_lote = 10

    print(f"\nGenerando {cantidad_generar} para score {score_objetivo}")

    while generadas < cantidad_generar:
        comentarios = subset["review"].sample(n=ejemplos_por_prompt).tolist()

        restantes = cantidad_generar - generadas
        n_paraf = min(parafrases_por_lote, restantes)

        prompt = construir_prompt(comentarios, etiqueta, n_paraf)

        try:
            salida = call_ollama(prompt)
            parafrases = limpiar_parafrases(salida)
        except Exception as e:
            print(f"Error: {e}")
            break

        if not parafrases:
            print("No output generado, deteniendo.")
            break

        for p in parafrases[:n_paraf]:
            base = subset.sample(n=1).iloc[0]

            nueva = {
                "reviewId": generar_review_id(),
                "review": p,
                "score": score_objetivo,
                "gender": base["gender"],
                "location": base["location"],
                "date": base["date"]
            }

            resultados.append(nueva)
            generadas += 1

            if generadas >= cantidad_generar:
                break

        print(f"Lote {lote}: {generadas}/{cantidad_generar}")
        lote += 1
        time.sleep(0.3)

    return resultados


def main():
    print("=== Oversampling automático ===\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    datos_dir = os.path.join(base_dir, "..", "Datos")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    archivos = [f for f in os.listdir(datos_dir) if f.endswith(".csv")]

    for i, f in enumerate(archivos, 1):
        print(f"{i}. {f}")

    opcion = int(input("\nSelecciona CSV: "))
    archivo = archivos[opcion - 1]

    df = cargar_csv(os.path.join(datos_dir, archivo))

    columnas = ["reviewId", "review", "score", "gender", "location", "date"]
    df = df[columnas].copy()

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["review", "score"])
    df["score"] = df["score"].astype(int)

    print("\nDistribución original:")
    dist = df["score"].value_counts().sort_index()
    print(dist)

    # 3 clases más pequeñas
    scores_min = dist.sort_values().head(NUM_CLASSES_TO_OVERSAMPLE).index.tolist()

    print(f"\nClases minoritarias: {scores_min}")

    nuevas = []

    for s in scores_min:
        cantidad = int(round(dist[s] * OVERSAMPLING_PERCENTAGE))

        print(f"\nScore {s} -> generar {cantidad}")

        nuevas.extend(
            generar_filas_para_score(df, s, cantidad)
        )

    df_new = pd.DataFrame(nuevas, columns=columnas)
    df_final = pd.concat([df, df_new], ignore_index=True)

    print("\nDistribución final:")
    print(df_final["score"].value_counts().sort_index())

    base = archivo.split(".")[0]

    df_final.to_csv(
        os.path.join(output_dir, f"{base}_oversampling.csv"),
        index=False
    )

    df_new.to_csv(
        os.path.join(output_dir, f"{base}_generadas.csv"),
        index=False
    )

    print("\nGuardado en carpeta output")


if __name__ == "__main__":
    main()