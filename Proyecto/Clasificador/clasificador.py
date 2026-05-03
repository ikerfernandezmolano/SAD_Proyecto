# -*- coding: utf-8 -*-
import random
import sys
import signal
import argparse
import unicodedata
import pandas as pd
import numpy as np
import pickle
import json
import csv
import os
import time
import warnings
from colorama import Fore
from pandas.errors import Pandas4Warning
# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.model_selection import GridSearchCV
# Preprocesado
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, LabelEncoder, Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import emoji, contractions
# kNN
from sklearn.neighbors import KNeighborsClassifier
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
# Logistic Regression
from sklearn.linear_model import LogisticRegression
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
# tqdm
from tqdm import tqdm

global package
global model
package = {}

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
    parser = argparse.ArgumentParser(description="Sistema de Apoyo a la Decisión - Clasificador")
    parser.add_argument("-j", "--json", help="Archivo de configuración JSON", required=True)
    parser.add_argument("-m", "--mode", help="Train o test", required=True)  # train o test
    parser.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1,type=int)
    parser.add_argument("-v", "--verbose", help="Mostrar métricas extendidas", action="store_true") # si se pone -v o --verbose se mostrarán métricas extendidas, si no, solo las básicas

    args = parser.parse_args() # obtiene el json que le pasamos

    try:
        # Se guarda el Json como diccionario en config y se guarda su contenido en args para que sea accesible desde cualquier función
        with open(args.json, 'r') as f:
            config = json.load(f)

        for key, value in config.items():
            setattr(args, key, value)

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
        data = pd.read_csv(file, encoding='utf-8', sep=args.sep)
        try:
            if "ColumnaAPredecir" not in str(args.prediction):
                data.rename(columns={args.prediction: "ColumnaAPredecir:" + args.prediction}, inplace=True)
                args.prediction = "ColumnaAPredecir:" + args.prediction
            else:
                data.rename(columns={str(args.prediction).split(":")[1]: args.prediction}, inplace=True)
        except Exception as e:
            print(Fore.RED + "Surgió un problema al renombrar la columna a predecir" + Fore.RESET)
        print(Fore.GREEN + "Datos cargados con éxito" + Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED + "Error al cargar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)

# ===========================
# Preprocesado
# ===========================

def select_features(data):
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

    Returns:
        numerical_feature (DataFrame): DataFrame que contiene las características numéricas.
        text_feature (DataFrame): DataFrame que contiene las características de texto.
        categorical_feature (DataFrame): DataFrame que contiene las características categóricas.
    """
    try:
        if args.mode == "train":
            # 1. Identificar numéricas
            numerical_feature = data.select_dtypes(include=['int64', 'float64'])

            # 2. Identificar categóricas
            unique_threshold = args.preprocessing.get("unique_category_threshold", 20)
            categorical_feature = data.select_dtypes(include='object')
            categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= unique_threshold]

            # 3. Identificar texto
            all_objects = data.select_dtypes(include='object')
            text_feature = all_objects.drop(columns=categorical_feature.columns)

            if args.prediction in numerical_feature.columns:
                numerical_feature = numerical_feature.drop(columns=[args.prediction])

            if args.prediction in categorical_feature.columns:
                categorical_feature = categorical_feature.drop(columns=[args.prediction])

            if args.prediction in text_feature.columns:
                text_feature = text_feature.drop(columns=[args.prediction])

            print(Fore.GREEN + "Datos separados con éxito (Predicción excluida de X)" + Fore.RESET)
            return data, numerical_feature, text_feature, categorical_feature
        elif args.mode == "test":
            # Eliminar la columna que quieres predecir para que no entre al modelo
            if args.prediction in data.columns:
                data = data.drop(columns=[args.prediction])
            # Numerical features
            numerical_feature = data.select_dtypes(include=['int64', 'float64'])  # Columnas numéricas
            if args.prediction in numerical_feature.columns:
                numerical_feature = numerical_feature.drop(columns=[args.prediction])

            unique_threshold = model['unique_category_threshold']

            # Categorical features
            categorical_feature = data.select_dtypes(include='object')
            categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= unique_threshold]

            # Text features
            text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)

            print(Fore.GREEN + "Datos separados con éxito" + Fore.RESET)

            return data, numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(Fore.RED + "Error al separar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)


def process_missing_values(data, numerical_feature, categorical_feature):
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
    try:
        if args.mode == "train":
            # Leemos la estrategia del JSON (si no la pones, usa mediana y moda por defecto)
            impute_num = args.preprocessing.get("impute_num", "median").lower()
            impute_cat = args.preprocessing.get("impute_cat", "mode").lower()
            package['impute_num'] = impute_num
            package['impute_cat'] = impute_cat

            # Numéricas
            for col in numerical_feature.columns:
                if col in data.columns and data[col].isnull().any():
                    data[col] = pd.to_numeric(data[col], errors="coerce")

            if impute_num == "constant":
                while True:
                    try:
                        package['impute_num_const'] = int(input("¿Qué constante quieres utilizar para la imputación de valores numéricos?: "))
                        break
                    except ValueError:
                        print("Por favor, introduce un número entero válido.")

            if impute_num == "delete":
                data = data.dropna(subset=numerical_feature.columns)
            else:
                for col in numerical_feature.columns:
                    if col in data.columns and data[col].isnull().any():
                        if impute_num == "mean":
                            data[col] = data[col].fillna(data[col].mean())
                        elif impute_num == "mode":
                            data[col] = data[col].fillna(data[col].mode()[0])
                        elif impute_num == "constant":
                            data[col] = data[col].fillna(package['impute_num_const'])
                        else:
                            data[col] = data[col].fillna(data[col].median())
            # Categóricas
            if impute_cat == "constant":
                package['impute_cat_const'] = str(input("¿Qué constante quieres utilizar para la imputación de valores categoriales?: "))
            if impute_cat == "delete":
                data = data.dropna(subset=categorical_feature.columns)
            else:
                for col in categorical_feature.columns:
                    if col in data.columns and data[col].isnull().any():
                        if impute_cat == "constant":
                            data[col] = data[col].fillna(package['impute_cat_const'])
                        else: # Por defecto: moda
                            moda = data[col].mode(dropna=True)
                            fill_value = moda.iloc[0] if not moda.empty else "Desconocido"
                            data[col] = data[col].fillna(fill_value)

            print(Fore.GREEN + "Valores faltantes procesados con éxito" + Fore.RESET)
        elif args.mode == "test":
            # Leemos la estrategia del JSON (si no la pones, usa mediana y moda por defecto)
            impute_num = model['impute_num']
            impute_cat = model['impute_cat']

            # Numéricas
            for col in numerical_feature.columns:
                if col in data.columns and data[col].isnull().any():
                    data[col] = pd.to_numeric(data[col], errors="coerce")

            if impute_num == "delete":
                data = data.dropna(subset=numerical_feature.columns)
            else:
                for col in numerical_feature.columns:
                    if col in data.columns and data[col].isnull().any():
                        if impute_num == "mean":
                            data[col] = data[col].fillna(data[col].mean())
                        elif impute_num == "mode":
                            data[col] = data[col].fillna(data[col].mode()[0])
                        elif impute_num == "constant":
                            data[col] = data[col].fillna(model['impute_num_const'])
                        else:  # Por defecto: mediana
                            data[col] = data[col].fillna(data[col].median())

            if impute_cat == "delete":
                data = data.dropna(subset=categorical_feature.columns)
            else:
                for col in categorical_feature.columns:
                    if col in data.columns and data[col].isnull().any():
                        if impute_cat == "constant":
                            data[col] = data[col].fillna(package['impute_cat_const'])
                        else:  # Por defecto: moda
                            moda = data[col].mode(dropna=True)
                            fill_value = moda.iloc[0] if not moda.empty else "Desconocido"
                            data[col] = data[col].fillna(fill_value)

            print(Fore.GREEN + "Valores faltantes procesados con éxito" + Fore.RESET)
        else:
            print(Fore.RED + "Modo no soportado" + Fore.RESET)
            sys.exit(1)
        return data

    except Exception as e:
        print(Fore.RED + "Error al procesar valores faltantes" + Fore.RESET)
        print(e)
        sys.exit(1)


def reescaler(data, numerical_feature, dev):
    """
    Rescala las características numéricas en el conjunto de datos utilizando diferentes métodos de escala.

    Args:
        numerical_feature (DataFrame): El dataframe que contiene las características numéricas.

    Returns:
        None

    Raises:
        Exception: Si hay un error al reescalar los datos.

    """
    if args.mode == "train":
        try:
            if numerical_feature.columns.size > 0:
                global package
                if not dev:
                    scaler_name = args.preprocessing.get("scaler", "standard").lower()

                    if scaler_name == "minmax":
                        scaler = MinMaxScaler()
                        print(Fore.GREEN + f"Datos reescalados con éxito utilizando {scaler_name}" + Fore.RESET)
                    elif scaler_name == "maxabs":
                        scaler = MaxAbsScaler()
                        print(Fore.GREEN + f"Datos reescalados con éxito utilizando {scaler_name}" + Fore.RESET)
                    elif scaler_name == "zscore":
                        scaler = StandardScaler()
                        print(Fore.GREEN + f"Datos reescalados con éxito utilizando {scaler_name}" + Fore.RESET)
                    elif scaler_name == "normalizer":
                        scaler = Normalizer()
                        print(Fore.GREEN + f"Datos reescalados con éxito utilizando {scaler_name}" + Fore.RESET)
                    else:
                        print(Fore.YELLOW+"No se están escalando los datos"+Fore.RESET)
                        return data
                    data[numerical_feature.columns] = scaler.fit_transform(data[numerical_feature.columns])
                    package['scaler'] = scaler
                else:
                    scaler = package['scaler']
                    data[numerical_feature.columns] = scaler.transform(data[numerical_feature.columns])
            else:
                print(Fore.YELLOW + "No se han encontrado columnas numéricas para reescalar" + Fore.RESET)
                return data

        except Exception as e:
                print(Fore.RED + "Error al reescalar los datos" + Fore.RESET)
                print(e)
                sys.exit(1)
    elif args.mode == "test":
        try:
            if numerical_feature.columns.size == 0:
                print(Fore.YELLOW + "No se han encontrado columnas numéricas para reescalar" + Fore.RESET)
                return data

            scaler = model['scaler']

            cols = [col for col in numerical_feature.columns if col in data.columns]
            if cols and scaler:
                data[cols] = scaler.transform(data[cols])
                print(Fore.GREEN + "Datos reescalados con éxito" + Fore.RESET)
            elif not scaler:
                print(Fore.YELLOW + "No se están escalando los datos" + Fore.RESET)
                return data
        except Exception as e:
            print(Fore.RED + "Error al reescalar los datos" + Fore.RESET)
            print(e)
            sys.exit(1)
    else:
        print(Fore.RED + f"Modo no soportado." + Fore.RESET)
        sys.exit(1)

    return data


def cat2num(data, categorical_feature, dev):
    """
    Convierte las características categóricas en características numéricas utilizando la codificación de etiquetas.

    Parámetros:
    categorical_feature (DataFrame): El DataFrame que contiene las características categóricas a convertir.

    """
    global package
    try:
        if categorical_feature.columns.size > 0:

            # Filtrar columnas que realmente existen en data
            cols = [col for col in categorical_feature.columns if col in data.columns]
            if not cols:
                return data

            if args.mode == "train":
                encoders = {}
                for col in categorical_feature.columns:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col])
                    encoders[col] = le
                package['categorical_encoder'] = encoders

            elif args.mode == "test" or dev:
                if args.mode == "test":
                    encoders = model['categorical_encoder']
                elif dev:
                    encoders = package['categorical_encoder']
                for col in categorical_feature.columns:
                    if col in encoders:
                        le = encoders[col]
                        data[col] = le.transform(data[col])
                    else:
                        print(Fore.YELLOW + f"No hay encoder para la columna {col}" + Fore.RESET)

            else:
                print(Fore.RED + f"Modo no soportado." + Fore.RESET)
                sys.exit(1)

            print(Fore.GREEN+"Datos categóricos pasados a numéricos con éxito"+Fore.RESET)
            return data
        else:
            print(Fore.YELLOW + "No se han encontrado columnas categóricas para convertir" + Fore.RESET)
            return data
        
    except Exception as e:
        print(Fore.RED + f"Error en cat2num: {e}" + Fore.RESET)
        sys.exit(1)

def simplify_text(data, text_feature):
    """
    Función que simplifica el texto de una columna dada en un DataFrame. lower,stemmer, tokenizer, stopwords del NLTK....

    Parámetros:
    - text_feature: DataFrame - El DataFrame que contiene la columna de texto a simplificar.

    Retorna:
    None
    """

    try:
        if text_feature.columns.size > 0:
            for col in text_feature.columns:
                # Emojis
                data[col] = data[col].apply(lambda x: emoji.demojize(x))

                # Contracciones
                data[col] = data[col].apply(lambda x: contractions.fix(x))

                # Minúsculas
                data[col] = data[col].apply(lambda x: x.lower())

                # Tokenizamos
                data[col] = data[col].apply(lambda x: RegexpTokenizer(r'\w+|:\w+:').tokenize(x))

                # Borrar numeros
                data[col] = data[col].apply(lambda x: [word for word in x if not word.isnumeric()])

                # Cargar stopwords
                stop_words = set(stopwords.words(args.preprocessing.get("language", "english")))

                # Palabras que NO quieres eliminar
                keep_words = {"no", "not"}

                # Eliminar esas palabras de las stopwords
                stop_words = stop_words - keep_words

                # Borrar stopwords
                data[col] = data[col].apply(lambda x: [word for word in x if word not in stop_words])

                # Lemmatizar
                data[col] = data[col].apply(lambda x: [WordNetLemmatizer().lemmatize(word) for word in x])

                # Borrar caracteres especiales
                data[col] = data[col].apply(lambda x: ''.join(c for c in unicodedata.normalize('NFD', ' '.join(x)) if unicodedata.category(c) != 'Mn'))

            print(Fore.GREEN + "Texto simplificado con éxito" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas de texto para simplificar" + Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED + "Error al simplificar el texto" + Fore.RESET)
        print(e)
        sys.exit(1)

def process_text(data, text_feature, dev):
    """
    Procesa las características de texto utilizando técnicas de vectorización como TF-IDF o BOW.

    Parámetros:
    text_feature (pandas.DataFrame): Un DataFrame que contiene las características de texto a procesar.

    """
    try:
        if text_feature.columns.size > 0:
            text_data = data[text_feature.columns].astype(str).agg(' '.join, axis=1)

            if args.mode == "train":
                global package
                if not dev:
                    if args.preprocessing["text_process"] == "tf-idf":
                        vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=args.preprocessing["ngram_range"])
                    elif args.preprocessing["text_process"] == "bow":
                        vectorizer = CountVectorizer(min_df=5, max_df=0.8, ngram_range=args.preprocessing["ngram_range"])
                    else:
                        print(Fore.YELLOW + "No se están tratando los textos" + Fore.RESET)
                        return data
                    matrix = vectorizer.fit_transform(text_data)
                    package['vectorizer'] = vectorizer
                else:
                    vectorizer = package['vectorizer']
                    matrix = vectorizer.transform(text_data)

            elif args.mode == "test":
                vectorizer = model['vectorizer']
                matrix = vectorizer.transform(text_data)

            else:
                print(Fore.RED + "No existe soporte para este modo" + Fore.RESET)
                sys.exit(1)

            data.drop(text_feature.columns, axis=1, inplace=True)
            text_features_df = pd.DataFrame(matrix.toarray(),columns=vectorizer.get_feature_names_out())
            data = pd.concat([data, text_features_df], axis=1)
            print(Fore.GREEN + "Texto tratado con éxito" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error al tratar el texto" + Fore.RESET)
        print(e)
        sys.exit(1)

    return data

def sampling(data):
    """
    Realiza oversampling o undersampling en los datos según la estrategia especificada en args.preprocessing["sampling"].

    Args:
        None

    Returns:
        None

    Raises:
        Exception: Si ocurre algún error al realizar el oversampling o undersampling.
    """

    try:
        if args.mode != "test":
            if args.preprocessing["sampling"].lower() == "oversampling":
                sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
            elif args.preprocessing["sampling"].lower() == "undersampling":
                sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            elif args.preprocessing["sampling"].lower() == "smote":
                sampler = SMOTE(sampling_strategy='auto', random_state=42)
            elif args.preprocessing["sampling"].lower() == "adasyn":
                sampler = ADASYN(sampling_strategy='auto', random_state=42)
            else:
                print(Fore.YELLOW + "No se están realizando oversampling, undersampling, SMOTE o ADASYN" + Fore.RESET)

            if args.preprocessing["sampling"].lower() == "oversampling" or args.preprocessing["sampling"].lower() == "smote" or args.preprocessing["sampling"].lower() == "undersampling" or args.preprocessing["sampling"].lower() == "adasyn":
                x = data.drop(columns=[args.prediction])
                y = data[args.prediction]
                x, y = sampler.fit_resample(x, y)
                x = pd.DataFrame(x, columns=data.drop(columns=[args.prediction]).columns)
                y = pd.Series(y, name=args.prediction)
                data = pd.concat([x, y], axis=1)
                print(Fore.GREEN + f"{args.preprocessing["sampling"]} realizado con éxito" + Fore.RESET)
        else:
            print(Fore.GREEN + "No se realiza oversampling, undersampling o SMOTE en modo test" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error al realizar oversampling, undersampling, SMOTE o ADASYN" + Fore.RESET)
        print(e)
        sys.exit(1)

    return data

def drop_features(data):
    """
    Elimina las columnas especificadas en el argumento 'drop_features' del dataset global 'data'.

    Si una columna no existe en el dataset, se muestra un mensaje de advertencia en color amarillo.
    Si el argumento 'debug' está activado, se muestra un mensaje en color magenta indicando la columna eliminada.
    Al finalizar, se muestra un mensaje en color verde indicando que las columnas han sido eliminadas.

    En caso de producirse un error al eliminar las columnas, se muestra un mensaje de error en color rojo y se finaliza el programa.

    Parámetros:
    - Ninguno

    Retorna:
    - Ninguno
    """
    try:
        if args.preprocessing["drop_features"] != []:
            for column in args.preprocessing["drop_features"]:
                if column not in data.columns:
                    print(Fore.YELLOW+f"La columna {column} no existe en el dataset"+Fore.RESET)
                else:
                    data.drop(column, axis=1, inplace=True)
            print(Fore.GREEN+f"Columnas eliminadas"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se han especificado columnas a eliminar"+Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED+"Error al eliminar columnas"+Fore.RESET)
        print(e)
        sys.exit(1)

def convertirRating(data):
    if args.mode == "train":
        global package
        package['pnn'] = args.preprocessing["pnn"]
        if args.preprocessing["pnn"] is True:
            data[args.prediction] = data[args.prediction].apply(lambda x: "positive" if x >= 4 else "negative" if x <= 2 else "neutral")
            print(Fore.GREEN+"Columna a predecir convertida con éxito"+Fore.RESET)
    return data

def preprocesar_datos(data, dev):
    """
    Función para preprocesar los datos
        1. Separamos los datos por tipos (Categoriales, numéricos y textos)
        2. Pasar los datos de categoriales a numéricos
        3. Tratamos missing values (Eliminar y imputar)
        4. Reescalamos los datos datos (MinMax, Normalizer, MaxAbsScaler)
        5. Simplificamos el texto (Normalizar, eliminar stopwords, stemming y ordenar alfabéticamente)
        6. Tratamos el texto (TF-IDF, BOW)
        7. Realizamos Oversampling o Undersampling
        8. Borrar columnas no necesarias
    :param data: Datos a preprocesar
    :return: Datos preprocesados y divididos en train y test
    """

    # Silencia los avisos de depuración de Pandas
    warnings.filterwarnings("ignore", category=Pandas4Warning)

    # Eliminamos columnas no necesarias
    data = drop_features(data)

    # Separamos los datos por tipos
    data, numerical_feature, text_feature, categorical_feature = select_features(data)

    # Simplificamos el texto
    data = simplify_text(data, text_feature)

    # Tratamos missing values
    data = process_missing_values(data, numerical_feature, categorical_feature)

    # Pasar los datos a categoriales a numéricos
    data = cat2num(data, categorical_feature, dev)

    # Convertimos los ratings a categorias
    data = convertirRating(data)

    # Reescalamos los datos numéricos
    data = reescaler(data, numerical_feature, dev)

    # Tratamos el texto
    data = process_text(data, text_feature, dev)

    if not dev:
        # Balanceo
        data = sampling(data)

    return data

# ===========================
# Funciones calculos
# ===========================

# =========================================================================
# EXPLICACIÓN DE METRICAS
# =========================================================================
#
# PRECISION (Precisión):
#    De todas las predicciones positivas que hizo el modelo, ¿cuántas
#    eran correctas en la realidad?
#    -> Una precisión alta indica que el modelo minimiza los falsos positivos.
#
# RECALL (Sensibilidad / Exhaustividad):
#    De todos los casos positivos reales que existen en el conjunto de datos,
#    ¿cuántos logró identificar correctamente el modelo?
#    -> Un recall alto indica que el modelo minimiza los falsos negativos.
#
# F1-SCORE:
#    La media armónica entre Precision y Recall. Es la métrica de referencia
#    cuando se trabaja con conjuntos de datos desbalanceados.
#       - F-SCORE MACRO: Calcula la métrica para cada clase de forma
#         independiente y calcula su media no ponderada. Trata a todas las
#         clases por igual, penalizando si el modelo falla en la clase minoritaria.
#       - F-SCORE MICRO: Calcula las métricas sumando globalmente los verdaderos
#         positivos, falsos negativos y falsos positivos. En datasets desbalanceados,
#         la clase mayoritaria dominará el resultado.
#
# MATRIZ DE CONFUSION:
#    Tabla que detalla los aciertos y errores exactos de clasificación.
#    En clasificación binaria muestra: Verdaderos Positivos, Verdaderos Negativos,
#    Falsos Positivos y Falsos Negativos.
#
# CLASSIFICATION REPORT:
#    Genera un informe en texto plano que desglosa las métricas de Precision,
#    Recall, F1-score y Soporte (número total de apariciones reales) para
#    cada una de las clases evaluadas.
# =========================================================================

def calcular_metricas(y_test, y_pred):
    requested_fscore = str(getattr(args, 'f_score', 'macro')).lower() # obtenemos el tipo de fscore que queremos calcular, por defecto macro, y lo convertimos a minúsculas para evitar problemas
    if requested_fscore not in ["micro", "macro", "weighted", "none"]:
        requested_fscore = "macro"

    # Cálculo de precisión, recall y fscore según el tipo solicitado (micro, macro o weighted)
    precision = precision_score(y_test, y_pred, average=requested_fscore if requested_fscore != "none" else "macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average=requested_fscore if requested_fscore != "none" else "macro", zero_division=0)
    fscore = f1_score(y_test, y_pred, average=requested_fscore if requested_fscore != "none" else "macro", zero_division=0)

    return precision, recall, fscore

def calculate_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return cm

def calculate_fscore(y_true, y_pred):
    return (
        f1_score(y_true, y_pred, average='micro', zero_division=0),
        f1_score(y_true, y_pred, average='macro', zero_division=0)
    )

def calculate_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, zero_division=0, digits=4)

# =========================================
# Funciones comunes para los modelos
# =========================================

def save_model(gs):
    """
    Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.

    Parámetros:
    - gs: objeto GridSearchCV, el cual contiene el modelo y los resultados de la búsqueda de hiperparámetros.

    Excepciones:
    - Exception: Si ocurre algún error al guardar el modelo.

    Archivos generados:
    - output/modelo.sav: Archivo que contiene el modelo guardado en formato pickle.
    - output/modelo.csv: Archivo CSV que contiene los parámetros probados y sus respectivas puntuaciones obtenidas durante la búsqueda de hiperparámetros.
    """
    try:
        package['model'] = gs
        package['unique_category_threshold'] = args.preprocessing.get("unique_category_threshold", 20)
        algorithm = args.algorithm
        with open(f'output/Modelo{algorithm}.sav', 'wb') as file:
            pickle.dump(package, file)
            print(Fore.CYAN + "Modelo guardado con éxito" + Fore.RESET)

        # Extraemos los datos necesarios de cv_results_
        results = gs.cv_results_
        params_list = results['params']
        mean_precision = results['mean_test_precision']
        mean_recall = results['mean_test_recall']
        mean_f1 = results['mean_test_f1']

        with open(f'output/Results_{algorithm}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            fscore = getattr(args, 'f_score', 'macro')
            writer.writerow(['NombreMod', 'Precisión', 'Recall', f'F_score({fscore})'])

            # Iteramos usando el índice para acceder a todas las listas a la vez
            for i in range(len(params_list)):
                params = params_list[i]

                if algorithm.lower() == 'knn':
                    nombreMod = f"{algorithm}_k{params['n_neighbors']}_p{params['p']}_w{params['weights']}"
                elif algorithm.lower() == 'decision_tree':
                    nombreMod = f"{algorithm}_depth{params.get('max_depth')}_split{params.get('min_samples_split')}_leaf{params.get('min_samples_leaf')}_criterion{params.get('criterion')}"
                elif algorithm.lower() == 'random_forest':
                    nombreMod = f"{algorithm}_nest{params.get('n_estimators')}_depth{params.get('max_depth')}_split{params.get('min_samples_split')}_leaf{params.get('min_samples_leaf')}_criterion{params.get('criterion')}"
                elif algorithm.lower() == 'naive_bayes':
                    nombreMod = f"{algorithm}_alpha{params.get('alpha')}_fitprior{params.get('fit_prior')}"
                elif algorithm.lower() == 'logistic_regression':
                    nombreMod = f"{algorithm}_C{params.get('C')}_l1ratio{params.get('l1_ratio')}_solver{params.get('solver')}_maxiter{params.get('max_iter')}"

                # Escribimos los valores correspondientes a esa combinación de parámetros
                writer.writerow([
                    nombreMod,
                    f"{mean_precision[i]:.4f}",
                    f"{mean_recall[i]:.4f}",
                    f"{mean_f1[i]:.4f}"
                ])
    except Exception as e:
        print(Fore.RED + "Error al guardar el modelo" + Fore.RESET)
        print(e)

def mostrar_resultados(gs, x_dev, y_dev):
    """
    Muestra los resultados del clasificador.

    Parámetros:
    - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
    - x_dev: array-like, las características del conjunto de desarrollo.
    - y_dev: array-like, las etiquetas del conjunto de desarrollo.

    Imprime en la consola los siguientes resultados:
    - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
    - Mejor puntuación obtenida por el clasificador.
    - F1-score micro del clasificador en el conjunto de desarrollo.
    - F1-score macro del clasificador en el conjunto de desarrollo.
    - Informe de clasificación del clasificador en el conjunto de desarrollo.
    - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """
    if args.verbose:
        print(Fore.MAGENTA + "> Mejores parametros:\n" + Fore.RESET, gs.best_params_)
        print(Fore.MAGENTA + "> Mejor puntuacion:\n" + Fore.RESET, gs.best_score_)
        print(Fore.MAGENTA + "> F1-score micro:\n" + Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[0])
        print(Fore.MAGENTA + "> F1-score macro:\n" + Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[1])
        print(Fore.MAGENTA + "> Informe de clasificación:\n" + Fore.RESET, calculate_classification_report(y_dev, gs.predict(x_dev)))
        print(Fore.MAGENTA + "> Matriz de confusión:\n" + Fore.RESET,calculate_confusion_matrix(y_dev, gs.predict(x_dev)))

# =========================================================================
# Algoritmo kNN
# =========================================================================
# ¿CÓMO PIENSA? Basado en distancia. Clasifica un dato nuevo mirando la 
# etiqueta de sus 'k' vecinos más cercanos en el espacio.
#
# ¿CUÁNDO USARLO? 
# - Datos numéricos continuos (Edad, Sueldo, Coordenadas).
# - Datasets pequeños o medianos (el cálculo de distancias es costoso).
# - Distribuciones de datos desconocidas (es un modelo no paramétrico).
#
# ¿CUÁNDO EVITARLO? 
# - Texto libre (TF-IDF): La alta dimensionalidad estropea el cálculo de distancias.
# - Datos muy desbalanceados (la clase mayoritaria domina, salvo que se use weights).
# =========================================================================
def kNN(train, dev):
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparametros para encontrar los parametros optimos

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """

    k = args.kNN["k"].get("kValues")

    if not k:
        k_min = args.kNN["k"].get("valueMin")
        k_max = args.kNN["k"].get("valueMax")
        k_step = args.kNN["k"].get("step")
        k = list(range(k_min, k_max + 1, k_step))

    weights = args.kNN.get("weights", ["distance","uniform"])
    p_values = args.kNN.get("p", [1,2])

    k_params = {
        'n_neighbors': k,
        'p': p_values,
        'weights': weights
    }

    # Dividimos los datos
    y_train = train[args.prediction]
    x_train = train.drop(columns=[args.prediction])
    y_dev = dev[args.prediction]
    x_dev = dev.drop(columns=[args.prediction])

    # Configuramos las métricas de evaluación
    fscore_param = getattr(args, 'f_score', 'macro').lower()
    scoring_metrics = {
        'precision': f'precision_{fscore_param}',
        'recall': f'recall_{fscore_param}',
        'f1': f'f1_{fscore_param}'
    }
    
    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando kNN', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(KNeighborsClassifier(), k_params, cv=5, n_jobs=args.cpu, scoring=scoring_metrics, refit='f1')
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random()*2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time
    print(f"Tiempo de ejecución: {Fore.MAGENTA}{execution_time:.2f}{Fore.RESET} segundos")

    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)

    # Guardamos el modelo utilizando pickle
    save_model(gs)

# =========================================================================
# Algoritmo Arból de Decisión
# =========================================================================
# ¿CÓMO PIENSA? Crea un diagrama de flujo con reglas lógicas encadenadas 
# para dividir los datos paso a paso.
#
# ¿CUÁNDO USARLO? 
# - Cuando se necesita EXPLICABILIDAD ("Caja Blanca", 100% interpretable).
# - Inmune a diferencias de escala (no requiere normalizar datos numéricos).
# - Útil para detectar patrones en valores nulos explícitos (ej: "Desconocido").
#
# ¿CUÁNDO EVITARLO? 
# - Alto riesgo de OVERFITTING (sobreajuste) si no se limita la profundidad 
#   (max_depth). Tiende a memorizar el dataset en lugar de generalizar.
# =========================================================================
def decision_tree(train, dev):
    """
    Función para implementar el algoritmo de árbol de decisión.

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """

    from sklearn.exceptions import UndefinedMetricWarning

    # Esto ignorará específicamente los avisos de precisión/f-score indefinidos
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # Dividimos los datos
    y_train = train[args.prediction]
    x_train = train.drop(columns=[args.prediction])
    y_dev = dev[args.prediction]
    x_dev = dev.drop(columns=[args.prediction])

    # Configuramos las métricas de evaluación
    fscore_param = getattr(args, 'f_score', 'macro').lower()
    scoring_metrics = {
        'precision': f'precision_{fscore_param}',
        'recall': f'recall_{fscore_param}',
        'f1': f'f1_{fscore_param}'
    }
    
    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando decision tree', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(DecisionTreeClassifier(), args.decision_tree, cv=5, n_jobs=args.cpu, scoring=scoring_metrics, refit='f1')
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random()*2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA + f" {execution_time:.2f} " + Fore.RESET + "segundos")

    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)

    # Guardamos el modelo utilizando pickle
    save_model(gs)

# =========================================================================
# Algoritmo Random Forest
# =========================================================================
# ¿CÓMO PIENSA? Entrena múltiples Árboles de Decisión independientes 
# (con subconjuntos de datos) y promedia/vota la predicción final.
#
# ¿CUÁNDO USARLO? 
# - Búsqueda de la máxima precisión y métricas altas (F1-Score).
# - Datasets complejos y desbalanceados.
# - Soluciona el problema de sobreajuste (overfitting) de los árboles simples.
#
# ¿CUÁNDO EVITARLO? 
# - Cuando es obligatorio explicar el motivo de la predicción ("Caja Negra").
# - Recursos limitados (consume mucha memoria RAM y CPU al entrenar).
# =========================================================================

def random_forest(train, dev):
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV para encontrar los mejores hiperparámetros.
    Divide los datos en entrenamiento y desarrollo, realiza la búsqueda de hiperparámetros, guarda el modelo entrenado
    utilizando pickle y muestra los resultados utilizando los datos de desarrollo.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """

    # Dividimos los datos
    y_train = train[args.prediction]
    x_train = train.drop(columns=[args.prediction])
    y_dev = dev[args.prediction]
    x_dev = dev.drop(columns=[args.prediction])

    # Configuramos las métricas de evaluación
    fscore_param = getattr(args, 'f_score', 'macro').lower()
    scoring_metrics = {
        'precision': f'precision_{fscore_param}',
        'recall': f'recall_{fscore_param}',
        'f1': f'f1_{fscore_param}'
    }

    with tqdm(total=100, desc='Procesando random forest', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(RandomForestClassifier(), args.random_forest, cv=5, n_jobs=args.cpu, scoring=scoring_metrics, refit='f1')
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random()*2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA + f" {execution_time} " + Fore.RESET + "segundos")

    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)

    # Guardamos el modelo utilizando pickle
    save_model(gs)

# =========================================================================
# Algoritmo Naive Bayes
# =========================================================================
# ¿CÓMO PIENSA? Basado en probabilidad (Teorema de Bayes). Es "ingenuo" 
# porque asume que todas las variables predictoras son independientes.
#
# ¿CUÁNDO USARLO? 
# - El estándar para Procesamiento de Lenguaje Natural (NLP).
# - Clasificación de texto libre (TF-IDF, Bag of Words, análisis de sentimientos).
# - Entrenamiento extremadamente rápido y muy ligero para la memoria.
#
# ¿CUÁNDO EVITARLO? 
# - Falla si las variables numéricas están fuertemente correlacionadas 
#   entre sí (porque rompe la regla matemática de independencia).
# =========================================================================

def naive_bayes(train, dev):
    """
    Función que entrena un modelo de Naive Bayes utilizando GridSearchCV.
    """

    # Dividimos los datos
    y_train = train[args.prediction]
    x_train = train.drop(columns=[args.prediction])
    y_dev = dev[args.prediction]
    x_dev = dev.drop(columns=[args.prediction])

    # Configuramos las métricas de evaluación
    fscore_param = getattr(args, 'f_score', 'macro').lower()
    scoring_metrics = {
        'precision': f'precision_{fscore_param}',
        'recall': f'recall_{fscore_param}',
        'f1': f'f1_{fscore_param}'
    }

    with tqdm(total=100, desc='Procesando naive bayes', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(MultinomialNB(), args.naive_bayes, cv=5, n_jobs=args.cpu, scoring=scoring_metrics, refit='f1')
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random() * 2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA + f" {execution_time} " + Fore.RESET + "segundos")

    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)

    # Guardamos el modelo utilizando pickle
    save_model(gs)

# =========================================================================
# Algoritmo Logistic Regression
# =========================================================================

def logistic_regression(train, dev):
    """
    Función que entrena un modelo de Logistic Regression utilizando GridSearchCV.
    """

    # Dividimos los datos
    y_train = train[args.prediction]
    x_train = train.drop(columns=[args.prediction])
    y_dev = dev[args.prediction]
    x_dev = dev.drop(columns=[args.prediction])

    # Configuramos las métricas de evaluación
    fscore_param = getattr(args, 'f_score', 'macro').lower()
    scoring_metrics = {
        'precision': f'precision_{fscore_param}',
        'recall': f'recall_{fscore_param}',
        'f1': f'f1_{fscore_param}'
    }

    with tqdm(total=100, desc='Procesando logistic regression', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(LogisticRegression(), args.logistic_regression, cv=5, n_jobs=args.cpu, scoring=scoring_metrics, refit='f1')
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random() * 2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA + f" {execution_time} " + Fore.RESET + "segundos")

    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)

    # Guardamos el modelo utilizando pickle
    save_model(gs)

# =====================================
# Funciones para predecir con un modelo
# =====================================

def load_model():
    """
    Carga el modelo desde el archivo 'output/modelo.sav' y lo devuelve.

    Returns:
        model: El modelo cargado desde el archivo 'output/modelo.sav'.

    Raises:
        Exception: Si ocurre un error al cargar el modelo.
    """
    try:
        with open(args.model, 'rb') as file:
            global model
            model = pickle.load(file)
            print(Fore.GREEN + "Modelo cargado con éxito" + Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED + "Error al cargar el modelo" + Fore.RESET)
        print(e)
        sys.exit(1)

def predict(data):
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    """
        Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

        Parámetros:
            Ninguno

        Retorna:
            Ninguno
        """

    # Predecimos
    prediction = model['model'].predict(data)

    file = pd.read_csv(args.data_file1, encoding='utf-8', sep=args.sep)
    predict = str(args.prediction).split(":")[1]
    pred_row = file[predict]
    if model['pnn']:
        pred_row = pred_row.apply(lambda x: "positive" if x >= 4 else "negative" if x <= 2 else "neutral")
    print(Fore.MAGENTA + "> Informe de clasificación:\n" + Fore.RESET, calculate_classification_report(pred_row, prediction))
    data = pd.concat([file.drop(columns=predict), pred_row, pd.DataFrame(prediction, columns=[args.prediction], index=data.index)], axis=1)
    return data

# ===========================
# Configuración inicial
# ===========================

def config():
    # Cargamos los datos con los que vamos a trabajar
    print("\n- Cargando datos...")
    data = load_data(args.data_file1)

    if args.mode == "train":
        data_aux = load_data(args.data_file2)

        print("\n- Descargando diccionarios...")
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

        # Preprocesamos los datos
        print("\n- Preprocesando train...")
        data = preprocesar_datos(data, False)
        print("\n- Preprocesando dev...")
        data_aux = preprocesar_datos(data_aux, True)

        if args.algorithm == "kNN":
            try:
                kNN(data, data_aux)
                print(Fore.GREEN + "Algoritmo kNN ejecutado con éxito" + Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "decision_tree":
            try:
                decision_tree(data, data_aux)
                print(Fore.GREEN + "Algoritmo árbol de decisión ejecutado con éxito" + Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "random_forest":
            try:
                random_forest(data, data_aux)
                print(Fore.GREEN + "Algoritmo random forest ejecutado con éxito" + Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "naive_bayes":
            try:
                naive_bayes(data, data_aux)
                print(Fore.GREEN + "Algoritmo naive bayes ejecutado con éxito" + Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "logistic_regression":
            try:
                logistic_regression(data, data_aux)
                print(Fore.GREEN + "Algoritmo logistic regression ejecutado con éxito" + Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        else:
            print(Fore.RED + "Algoritmo no soportado" + Fore.RESET)
            sys.exit(1)
    elif args.mode == "test":
        # Cargamos el modelo
        print("\n- Cargando modelo...")
        global model
        model = load_model()

        print("\n- Descargando diccionarios...")
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

        # Preprocesamos los datos
        print("\n- Preprocesando test...")
        data = preprocesar_datos(data, False)

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
            data = predict(data)
            print(Fore.GREEN + "Predicción realizada con éxito" + Fore.RESET)
            # Guardamos el dataframe con la prediccion
            data.to_csv('output/Prediccion.csv', index=False)
            print(Fore.GREEN + "Predicción guardada con éxito" + Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        print(Fore.RED + "Modo no soportado" + Fore.RESET)
        sys.exit(1)


# ===========================
# Entrada principal
# ===========================

if __name__ == "__main__":
    # Fijamos la semilla
    np.random.seed(42)
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Si la carpeta output no existe la creamos
    print("\n- Creando carpeta output...")
    try:
        os.makedirs('output')
        print(Fore.GREEN + "Carpeta output creada con éxito" + Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN + "La carpeta output ya existe" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al crear la carpeta output" + Fore.RESET)
        print(e)
        sys.exit(1)
    config()
