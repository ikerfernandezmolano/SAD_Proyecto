import os
import time
import uuid
import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma2:2b"

TARGET_LABEL = "Neutro"
TARGET_SCORE = 3
TOTAL_TO_GENERATE = 1000

PARAPHRASES_PER_BATCH = 30
EXAMPLES_PER_PROMPT = 3


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
        return pd.read_csv(
            ruta_csv,
            sep=None,
            engine="python",
            encoding="utf-8",
            on_bad_lines="skip"
        )
    except Exception:
        return pd.read_csv(
            ruta_csv,
            sep=None,
            engine="python",
            encoding="latin1",
            on_bad_lines="skip"
        )


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "num_predict": 900
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


def construir_prompt(comentarios, num_parafrases):
    texto = "\n".join([f"- {c}" for c in comentarios])

    return f"""Generate {num_parafrases} paraphrases of the following comments.

IMPORTANT:
- The sentiment MUST remain strictly NEUTRAL.
- Do NOT introduce positive or negative tone.
- Keep the tone balanced, objective, and descriptive.
- Avoid strong opinions or emotional language.

Language:
- Output MUST be in the SAME LANGUAGE as the input comments.
- DO NOT translate.
- If the original comments are in English, output must be in English.

Requirements:
- Sound like real user reviews of a music application.
- Keep meaning similar but not identical.
- Use simple, neutral wording.
- Return ONLY {num_parafrases} lines.
- One paraphrase per line.
- No explanations.
- No numbering.
- No offensive content.

Original comments:
{texto}
"""


def generar_review_id():
    return "gen_" + str(uuid.uuid4())[:8]


def generar_neutros(df):
    subset = df[df["label"] == TARGET_LABEL].copy()

    if subset.empty:
        print("No hay comentarios neutros en el CSV para usar como base.")
        return []

    resultados = []
    generadas = 0
    lote = 1

    ejemplos_por_prompt = min(EXAMPLES_PER_PROMPT, len(subset))

    print(f"\nGenerando {TOTAL_TO_GENERATE} comentarios neutros con score {TARGET_SCORE}")

    while generadas < TOTAL_TO_GENERATE:
        comentarios = (
            subset["review"]
            .dropna()
            .astype(str)
            .sample(n=ejemplos_por_prompt, replace=False)
            .tolist()
        )

        restantes = TOTAL_TO_GENERATE - generadas
        n_paraf = min(PARAPHRASES_PER_BATCH, restantes)

        prompt = construir_prompt(
            comentarios=comentarios,
            num_parafrases=n_paraf
        )

        try:
            salida = call_ollama(prompt)
            parafrases = limpiar_parafrases(salida)
        except Exception as e:
            print(f"Error en lote {lote}: {e}")
            print("Deteniendo para evitar bucle infinito.")
            break

        if not parafrases:
            print("No se generaron paráfrasis. Deteniendo.")
            break

        for p in parafrases[:n_paraf]:
            base = subset.sample(n=1).iloc[0]

            nueva = {
                "reviewId": generar_review_id(),
                "review": p,
                "score": TARGET_SCORE,
                "gender": base["gender"],
                "location": base["location"],
                "date": base["date"]
            }

            resultados.append(nueva)
            generadas += 1

            if generadas >= TOTAL_TO_GENERATE:
                break

        print(f"Lote {lote}: {generadas}/{TOTAL_TO_GENERATE}")
        lote += 1
        time.sleep(0.3)

    return resultados


def main():
    print("=== Oversampling generativo SOLO de comentarios neutros ===\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    datos_dir = os.path.join(base_dir, "..", "Datos")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    archivos = [f for f in os.listdir(datos_dir) if f.endswith(".csv")]

    if not archivos:
        print("No se encontraron CSV en la carpeta Datos.")
        return

    print("Archivos disponibles:")
    for i, f in enumerate(archivos, 1):
        print(f"{i}. {f}")

    try:
        opcion = int(input("\nSelecciona CSV: "))
        archivo = archivos[opcion - 1]
    except Exception:
        print("Selección no válida.")
        return

    ruta_csv = os.path.join(datos_dir, archivo)
    df = cargar_csv(ruta_csv)

    columnas = ["reviewId", "review", "score", "gender", "location", "date"]

    for col in columnas:
        if col not in df.columns:
            print(f"Falta la columna obligatoria: {col}")
            return

    df = df[columnas].copy()

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["review", "score"])
    df["score"] = df["score"].astype(int)
    df = df[df["score"].isin([1, 2, 3, 4, 5])].copy()

    df["label"] = df["score"].apply(score_to_label)

    print("\nDistribución original por score:")
    print(df["score"].value_counts().sort_index())

    print("\nDistribución original por clase:")
    print(df["label"].value_counts())

    nuevas = generar_neutros(df)

    df_new = pd.DataFrame(nuevas, columns=columnas)
    df_final = pd.concat([df[columnas], df_new], ignore_index=True)

    df_final["label_temp"] = df_final["score"].apply(score_to_label)

    print("\nDistribución final por score:")
    print(df_final["score"].value_counts().sort_index())

    print("\nDistribución final por clase:")
    print(df_final["label_temp"].value_counts())

    df_final = df_final.drop(columns=["label_temp"])

    base = os.path.splitext(archivo)[0]

    ruta_final = os.path.join(output_dir, f"{base}_oversampling_1000_neutros.csv")
    ruta_generadas = os.path.join(output_dir, f"{base}_generadas_1000_neutros.csv")

    df_final.to_csv(
        ruta_final,
        index=False,
        sep=";",
        encoding="utf-8"
    )

    df_new.to_csv(
        ruta_generadas,
        index=False,
        sep=";",
        encoding="utf-8"
    )

    print("\n=== FINALIZADO ===")
    print(f"Instancias neutras generadas: {len(df_new)}")
    print(f"CSV final guardado en:\n{ruta_final}")
    print(f"CSV solo generadas guardado en:\n{ruta_generadas}")


if __name__ == "__main__":
    main()