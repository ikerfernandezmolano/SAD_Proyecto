import sys
import signal
import argparse
import json
import os
import pandas as pd
from colorama import Fore
import pickle

global data

# ===========================
# Funciones de Sistema
# ===========================

def signal_handler(sig, frame):
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """
    print("\nSaliendo del programa...")
    sys.exit(0)

def parse_args():
    """
    Función para parsear los argumentos de entrada y combinar con JSON
    """
    parser = argparse.ArgumentParser(description="Sistema de Apoyo a la Decisión - Tester")
    parser.add_argument("-j", "--json", help="Archivo de configuración JSON", required=True)
    parser.add_argument("-v", "--verbose", help="Mostrar métricas extendidas", action="store_true") # si se pone -v o --verbose se mostrarán métricas extendidas, si no, solo las básicas

    args = parser.parse_args() # obtiene el json que le pasamos

    try:
        # Se guarda el Json como diccionario en config y se guarda su contenido en args para que sea accesible desde cualquier función
        with open(args.json, 'r') as f:
            config = json.load(f)

        for key, value in config.items():
            setattr(args, key, value)

        if not hasattr(args, 'preprocessing') or args.preprocessing is None:
            args.preprocessing = {}

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {args.json}")
        sys.exit(1)

    return args

# =====================================
# Funciones para predecir con un modelo
# =====================================

def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """

    try:
        global data
        data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN + "Datos cargados con éxito" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al cargar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)

def load_model():
    """
    Carga el modelo desde el archivo 'output/modelo.pkl' y lo devuelve.

    Returns:
        model: El modelo cargado desde el archivo 'output/modelo.pkl'.

    Raises:
        Exception: Si ocurre un error al cargar el modelo.
    """
    try:
        with open(f'output/{args.model}.pkl', 'rb') as file:
            model = pickle.load(file)
            print(Fore.GREEN + "Modelo cargado con éxito" + Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED + "Error al cargar el modelo" + Fore.RESET)
        print(e)
        sys.exit(1)


def predict():
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    global data
    # Predecimos
    prediction = model.predict(data)

    # Añadimos la prediccion al dataframe data
    data = pd.concat([data, pd.DataFrame(prediction, columns=[args.prediction])], axis=1)

# ===========================
# Entrada principal
# ===========================

if __name__ == "__main__":
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Cargamos los datos con los que vamos a trabajar
    print("\n- Cargando datos...")
    load_data(args.data_file)
    # Cargamos el modelo
    print("\n- Cargando modelo...")
    model = load_model()
    try:
        os.makedirs('output')
        print(Fore.GREEN + "Carpeta output creada con éxito" + Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN + "La carpeta output ya existe" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al crear la carpeta output" + Fore.RESET)
        print(e)
        sys.exit(1)
    # Predecimos
    print("\n- Prediciendo...")
    try:
        predict()
        print(Fore.GREEN+"Predicción realizada con éxito"+Fore.RESET)
        # Guardamos el dataframe con la prediccion
        data.to_csv('output/Prediccion.csv', index=False)
        print(Fore.GREEN+"Predicción guardada con éxito"+Fore.RESET)
        sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)