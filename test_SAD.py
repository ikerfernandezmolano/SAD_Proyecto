# -*- coding: utf-8 -*-
import sys
import signal
import argparse
import pandas as pd
import string
import pickle
import json
import os
import warnings
from colorama import Fore
from pandas.errors import Pandas4Warning
# Preprocesado
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

global data
global dataAux

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

def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """

    try:
        global data, dataAux
        data = pd.read_csv(file, encoding='utf-8')
        dataAux = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN + "Datos cargados con éxito" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al cargar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)

# ===========================
# Preprocesado
# ===========================

def select_features():
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

    Returns:
        numerical_feature (DataFrame): DataFrame que contiene las características numéricas.
        text_feature (DataFrame): DataFrame que contiene las características de texto.
        categorical_feature (DataFrame): DataFrame que contiene las características categóricas.
    """
    try:
        global data
        # Eliminar la columna que quieres predecir para que no entre al modelo
        if args.prediction in data.columns:
            data = data.drop(columns=[args.prediction])
        # Numerical features
        numerical_feature = data.select_dtypes(include=['int64', 'float64'])  # Columnas numéricas
        if args.prediction in numerical_feature.columns:
            numerical_feature = numerical_feature.drop(columns=[args.prediction])

        unique_threshold = args.preprocessing.get("unique_category_threshold", 20)
        # Categorical features
        categorical_feature = data.select_dtypes(include='object')
        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= unique_threshold]

        # Text features
        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)

        print(Fore.GREEN + "Datos separados con éxito" + Fore.RESET)

        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(Fore.RED + "Error al separar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)


def process_missing_values(numerical_feature, categorical_feature):
    """
    Procesa los valores faltantes en los datos según la estrategia especificada en los argumentos.

    Args:
        numerical_feature (DataFrame): El DataFrame que contiene las características numéricas.
        categorical_feature (DataFrame): El DataFrame que contiene las características categóricas.

    Returns:
        None

    Raises:
        None
    """
    global data
    try:
        # Numéricas: mediana
        for col in numerical_feature.columns:
            if col in data.columns and data[col].isnull().any():
                data[col] = pd.to_numeric(data[col], errors="coerce")
                data[col] = data[col].fillna(data[col].median())

        # Categóricas: moda
        for col in categorical_feature.columns:
            if col in data.columns and data[col].isnull().any():
                moda = data[col].mode(dropna=True)
                fill_value = moda.iloc[0] if not moda.empty else "missing"
                data[col] = data[col].fillna(fill_value)

        # Resto de columnas object/texto
        object_cols = data.select_dtypes(include=["object"]).columns
        for col in object_cols:
            if data[col].isnull().any():
                data[col] = data[col].fillna("missing")

        print(Fore.GREEN + "Valores faltantes procesados con éxito" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al procesar valores faltantes" + Fore.RESET)
        print(e)
        sys.exit(1)

def reescaler(numerical_feature):
    """
    Rescala las características numéricas en el conjunto de datos utilizando diferentes métodos de escala.

    Args:
        numerical_feature (DataFrame): El dataframe que contiene las características numéricas.

    Returns:
        None

    Raises:
        Exception: Si hay un error al reescalar los datos.

    """
    global data
    try:
        if numerical_feature.columns.size == 0:
            print(Fore.YELLOW + "No se han encontrado columnas numéricas para reescalar" + Fore.RESET)
            return

        scaler = package['scaler']

        cols = [col for col in numerical_feature.columns if col in data.columns]
        if cols:
            data[cols] = scaler.transform(data[cols])
            print(Fore.GREEN + "Datos reescalados con éxito" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al reescalar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)

def cat2num(categorical_feature):
    """
    Convierte las características categóricas en características numéricas utilizando la codificación de etiquetas.

    Parámetros:
    categorical_feature (DataFrame): El DataFrame que contiene las características categóricas a convertir.

    """
    global data
    try:
        cols = [col for col in categorical_feature.columns if col in data.columns and col != args.prediction]
        if not cols:
            print(Fore.YELLOW + "No se han encontrado columnas categóricas para convertir" + Fore.RESET)
            return

        data = pd.get_dummies(data, columns=cols, dummy_na=False, drop_first=False)
        print(Fore.GREEN + "Variables categóricas convertidas a numéricas con éxito" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al convertir variables categóricas a numéricas" + Fore.RESET)
        print(e)
        sys.exit(1)

def simplify_text(text_feature):
    """
    Función que simplifica el texto de una columna dada en un DataFrame. lower,stemmer, tokenizer, stopwords del NLTK....

    Parámetros:
    - text_feature: DataFrame - El DataFrame que contiene la columna de texto a simplificar.

    Retorna:
    None
    """
    global data
    try:
        if text_feature.columns.size == 0:
            print(Fore.YELLOW + "No se han encontrado columnas de texto para simplificar" + Fore.RESET)
            return

        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        def clean_text(text):
            text = str(text).lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            tokens = word_tokenize(text)
            tokens = [tok for tok in tokens if tok.isalpha()]
            tokens = [tok for tok in tokens if tok not in stop_words]
            tokens = [stemmer.stem(tok) for tok in tokens]
            tokens.sort()
            return " ".join(tokens)

        for col in text_feature.columns:
            if col in data.columns:
                data[col] = data[col].fillna("").astype(str).apply(clean_text)

        print(Fore.GREEN + "Texto simplificado con éxito" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al simplificar el texto" + Fore.RESET)
        print(e)
        sys.exit(1)

def process_text(text_feature):
    """
    Procesa las características de texto utilizando técnicas de vectorización como TF-IDF o BOW.

    Parámetros:
    text_feature (pandas.DataFrame): Un DataFrame que contiene las características de texto a procesar.

    """
    global data
    try:
        if text_feature.columns.size > 0:
            vectorizer = package['vectorizer']
            text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
            matrix = vectorizer.transform(text_data)
            text_features_df = pd.DataFrame(matrix.toarray(),
                                            columns=vectorizer.get_feature_names_out())
            data = pd.concat([data, text_features_df], axis=1)
            data.drop(text_feature.columns, axis=1, inplace=True)
            print(Fore.GREEN + "Texto tratado con éxito" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas de texto a procesar" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al tratar el texto" + Fore.RESET)
        print(e)
        sys.exit(1)

def preprocesar_datos():
    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

    # Silencia los avisos de depuración de Pandas
    warnings.filterwarnings("ignore", category=Pandas4Warning)

    numerical_feature, text_feature, categorical_feature = select_features() # dividir los datos con los que trabajamos en numéricos, categóricos y de texto
    simplify_text(text_feature)
    cat2num(categorical_feature)
    process_missing_values(numerical_feature, categorical_feature)
    reescaler(numerical_feature)
    process_text(text_feature)
    return data

# =====================================
# Funciones para predecir con un modelo
# =====================================

def load_model():
    try:
        with open(f'output/{args.model}', 'rb') as file:
            package = pickle.load(file)
            print(Fore.GREEN + "Modelo y metadatos cargados" + Fore.RESET)
            # Retornamos el diccionario completo
            return package
    except Exception as e:
        print(Fore.RED + f"Error al cargar: {e}" + Fore.RESET)
        sys.exit(1)

def predict():
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    global data, dataAux
    # Predecimos
    prediction = package['model'].predict(data)

    # Añadimos la prediccion al dataframe data
    dataAux = pd.concat([dataAux,pd.DataFrame(prediction, columns=[args.prediction])], axis=1)

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
    package = load_model()
    # Preprocesamos los datos
    preprocesar_datos()
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
        dataAux.to_csv('output/Prediccion.csv', index=False)
        print(Fore.GREEN+"Predicción guardada con éxito"+Fore.RESET)
        sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)