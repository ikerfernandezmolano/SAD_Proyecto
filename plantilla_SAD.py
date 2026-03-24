# -*- coding: utf-8 -*-
import sys
import signal
import argparse
import pandas as pd
import numpy as np
import string
import pickle
import json
import csv
import os
import time
from itertools import product
from colorama import Fore
# Sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.model_selection import GridSearchCV
# Preprocesado
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pandas.api.types import is_numeric_dtype
# kNN
from sklearn.neighbors import KNeighborsClassifier
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Sistema de Apoyo a la Decisión - Clasificador")
    parser.add_argument("-j", "--json", help="Archivo de configuración JSON", required=True)
    parser.add_argument("-v", "--verbose", help="Mostrar métricas extendidas", action="store_true")

    args = parser.parse_args()

    try:
        with open(args.json, 'r') as f:
            config = json.load(f)

        for key, value in config.items():
            setattr(args, key, value)

        if not hasattr(args, 'preprocessing') or args.preprocessing is None:
            args.preprocessing = {}
        if not hasattr(args, 'parameters') or args.parameters is None:
            args.parameters = {}

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
        data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN + "Datos cargados con éxito" + Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED + "Error al cargar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)

# ===========================
# Funciones CSV
# ===========================

def createCSV():
    fscore_name = args.parameters.get('f_score', 'macro')
    with open("Results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["NombreMod", "Precisión", "Recall", f"F_score({fscore_name})"])

def add_data(data):
    with open("Results.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)

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

        scaler_name = args.preprocessing.get("scaler", "standard").lower()

        if scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "normalizer":
            scaler = Normalizer()
        elif scaler_name == "maxabs":
            scaler = MaxAbsScaler()
        else:
            scaler = StandardScaler()

        cols = [col for col in numerical_feature.columns if col in data.columns]
        if cols:
            data[cols] = scaler.fit_transform(data[cols])
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
            if args.preprocessing["text_process"] == "tf-idf":
                tfidf_vectorizer = TfidfVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(tfidf_matrix.toarray(),
                                                columns=tfidf_vectorizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                data.drop(text_feature.columns, axis=1, inplace=True)
                print(Fore.GREEN + "Texto tratado con éxito usando TF-IDF" + Fore.RESET)
            elif args.preprocessing["text_process"] == "bow":
                bow_vectorizer = CountVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vectorizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                print(Fore.GREEN + "Texto tratado con éxito usando BOW" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No se están tratando los textos" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas de texto a procesar" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al tratar el texto" + Fore.RESET)
        print(e)
        sys.exit(1)


def over_under_sampling():
    global data
    sampling_type = args.preprocessing.get("sampling", "none")
    if sampling_type == "none": return

    try:
        X = data.drop(columns=[args.prediction])
        y = data[args.prediction]

        if sampling_type == "oversampling":
            sampler = RandomOverSampler(random_state=42)
        elif sampling_type == "undersampling":
            sampler = RandomUnderSampler(random_state=42)
        else: return

        X_res, y_res = sampler.fit_resample(X, y)
        data = pd.concat([X_res, y_res], axis=1)
        print(Fore.GREEN + f"Sampling ({sampling_type}) aplicado" + Fore.RESET)
    except Exception as e:
        print(Fore.YELLOW + "No se pudo aplicar sampling: " + str(e) + Fore.RESET)

def preprocesar_datos():
    numerical_feature, text_feature, categorical_feature = select_features()
    simplify_text(text_feature)
    cat2num(categorical_feature)
    process_missing_values(numerical_feature, categorical_feature)
    reescaler(numerical_feature)
    process_text(text_feature)
    over_under_sampling()
    return data

# ===========================
# Funciones compartidas
# ===========================

def exampleMessage(algorithm):
    if algorithm.lower() == 'knn':
        print("""JSON ejemplo para kNN:
        {
            "data_file": "file.csv",
            "algorithm": "kNN",
            "prediction": "nombreColAPredecir",
            "parameters": {
                "k": {
                    "valueMin": value,
                    "valueMax": value,
                    "step": value
                },
                "weights": "uniform/distance",
                "p": (1,2),
                "f_score": "macro/micro/weighted/none"
            }
        }""")
    elif algorithm.lower() == 'decision_tree':
        print("""JSON ejemplo para el árbol de decisión:
        {
            "data_file": "archivo csv",
            "algorithm": "decision_tree",
            "parameters": {
                "max_depth": [valor1, valor2, ..., valorn],
                "min_samples_leaf": [1, 2],
                "criterion": ["gini", "entropy"]
                "f_score": ["macro", "micro", "avg", "none"]
            }
        }""")
    else:
        print("""JSON ejemplo para algoritmo:""")

# ===========================
# Funciones calculos
# ===========================

def calcular_metricas(y_test, y_pred):
    requested_fscore = str(args.parameters.get("f_score", "macro")).lower()
    if requested_fscore not in ["micro", "macro", "weighted", "none"]:
        requested_fscore = "macro"

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
    return classification_report(y_true, y_pred, zero_division=0)

# ===========================
# Funciones comunes para los modelos
# ===========================

def divide_data():
    global data
    try:
        if args.prediction not in data.columns:
            raise ValueError(f"La columna objetivo '{args.prediction}' no existe en el dataset.")

        y = data[args.prediction].copy()
        X = data.drop(columns=[args.prediction]).copy()

        if not is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception as e:
        print(Fore.RED + "Error al dividir datos: " + str(e) + Fore.RESET)
        sys.exit(1)

def save_model(gs):
    """
    Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.

    Parámetros:
    - gs: objeto GridSearchCV, el cual contiene el modelo y los resultados de la búsqueda de hiperparámetros.

    Excepciones:
    - Exception: Si ocurre algún error al guardar el modelo.

    """
    try:
        with open('output/modelo.pkl', 'wb') as file:
            pickle.dump(gs, file)
            print(Fore.CYAN + "Modelo guardado con éxito" + Fore.RESET)
        with open('output/modelo.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Params', 'Score'])
            for params, score in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']):
                writer.writerow([params, score])
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

# ===========================
# Algoritmo kNN
# ===========================

def kNN():

    k_cfg = args.parameters.get("k", {})
    k_min = k_cfg.get("valueMin")
    k_max = k_cfg.get("valueMax")
    k_step = k_cfg.get("step")

    if not all(isinstance(v, int) and v > 0 for v in [k_min, k_max, k_step]) or k_min > k_max:
        print("Error en la configuración de k.")
        exampleMessage("kNN")
        sys.exit(1)

    weights = args.parameters.get("weights", ["uniform"])
    if isinstance(weights, str):
        weights = [weights]
    if not isinstance(weights, list) or not all(w in ["uniform", "distance"] for w in weights):
        print("Error en la configuración de weights.")
        exampleMessage("kNN")
        sys.exit(1)

    p_values = args.parameters.get("p", [2])
    if isinstance(p_values, int):
        p_values = [p_values]
    if not isinstance(p_values, list) or not all(p in [1, 2] for p in p_values):
        print("Error en la configuración de p.")
        exampleMessage("kNN")
        sys.exit(1)

    X_train, X_dev, y_train, y_dev = divide_data()

    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    min_class_count = class_counts.min()

    if min_class_count >= 5:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    elif min_class_count >= 2:
        cv = StratifiedKFold(n_splits=int(min_class_count), shuffle=True, random_state=42)
    else:
        raise ValueError("No es posible aplicar validación cruzada estratificada: alguna clase tiene menos de 2 muestras.")

    createCSV()
    resultados = []
    mejor_fscore = -1
    mejor_params = None

    start_time = time.time()
    for k, weight, p in product(range(k_min, k_max + 1, k_step), weights, p_values):
        modelo = KNeighborsClassifier(n_neighbors=k, weights=weight, p=p)

        y_pred = cross_val_predict(modelo, X_train, y_train, cv=cv, n_jobs=-1)

        precision, recall, fscore = calcular_metricas(y_train, y_pred)
        nombre_modelo = f"KNN_k{k}_p{p}_w{weight}"

        fila = [nombre_modelo, precision, recall, fscore]
        resultados.append(fila)
        add_data(fila)

        if fscore > mejor_fscore:
            mejor_fscore = fscore
            mejor_params = {'n_neighbors': k, 'weights': weight, 'p': p}

    if mejor_params:
        mejor_modelo = KNeighborsClassifier(**mejor_params)

        # Simulamos un objeto 'gs' mínimo para que tus funciones mostrar_resultados y save_model no fallen
        class DummyGS:
            def __init__(self, estimator, params, X, y):
                self.best_estimator_ = estimator.fit(X, y)
                self.best_params_ = params
                self.best_score_ = mejor_fscore
                self.cv_results_ = {'params': [params], 'mean_test_score': [mejor_fscore]}

            def predict(self, X): return self.best_estimator_.predict(X)

        gs_simulado = DummyGS(mejor_modelo, mejor_params, X_train, y_train)

        # Usamos tus funciones predefinidas
        mostrar_resultados(gs_simulado, X_dev, y_dev)
        save_model(gs_simulado)

        print(Fore.GREEN + f"\nResultados guardados en Results.csv" + Fore.RESET)
        print(Fore.CYAN + f"Mejor configuración: {mejor_params}" + Fore.RESET)

    end_time = time.time()
    print(f"Tiempo de ejecución: {Fore.MAGENTA}{end_time - start_time:.2f}{Fore.RESET} segundos")

    return resultados

# ===========================
# Algoritmo Arból de Decisión
# ===========================

def decision_tree():
    X_train, X_dev, y_train, y_dev = divide_data()

    max_depth = args.parameters.get('max_depth')
    if not isinstance(max_depth, list) or not all(isinstance(i, int) and i > 0 for i in max_depth):
        print("Error en el valor max_depth del algoritmo Árbol de Decisión.")
        exampleMessage("decision_tree")
        sys.exit(1)

    min_samples_leaf = args.parameters.get('min_samples_leaf')
    if not isinstance(min_samples_leaf, list) or not all(isinstance(i, int) and i > 0 for i in min_samples_leaf):
        print("Error en el valor min_samples_leaf del algoritmo Árbol de Decisión.")
        exampleMessage("decision_tree")
        sys.exit(1)

    criterion = args.parameters.get('criterion')
    if not isinstance(criterion, list) or not all(isinstance(i, str) and i in ['gini', 'entropy'] for i in criterion):
        print("Error en el valor criterion del algoritmo Árbol de Decisión.")
        exampleMessage("decision_tree")
        sys.exit(1)

    parametros_dt = {
        'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf,
        'criterion': criterion
    }

    dt = DecisionTreeClassifier(random_state=42)
    clf = GridSearchCV(dt, parametros_dt, cv=5, n_jobs=-1, scoring='f1_macro')
    print("Entrenando Árbol de Decisión y buscando los mejores parámetros...")
    clf.fit(X_train, y_train)

    resultados_todos = pd.DataFrame(clf.cv_results_)
    columnas_utiles = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    resultados_limpios = resultados_todos[columnas_utiles].sort_values(by='rank_test_score')
    resultados_limpios.to_csv('resultadosDeTodosModelos.csv', index=False)
    print("-> Archivo 'resultadosDeTodosModelos.csv' generado con éxito.")

    y_pred = clf.best_estimator_.predict(X_dev)
    return y_dev, y_pred

# ===========================
# Algoritmo Random Forest
# ===========================

def random_forest():
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV.
    """
    x_train, x_dev, y_train, y_dev = divide_data()

    params = args.parameters if hasattr(args, 'parameters') else {}
    param_grid = {
        'n_estimators': params.get('n_estimators', [100, 200]),
        'max_depth': params.get('max_depth', [None, 10, 20]),
        'min_samples_split': params.get('min_samples_split', [2, 5]),
        'min_samples_leaf': params.get('min_samples_leaf', [1, 2]),
        'criterion': params.get('criterion', ['gini', 'entropy'])
    }

    scoring_metric = f"f1_{params.get('f_score', 'macro')}" if params.get('f_score', 'macro') in ['macro', 'micro', 'weighted'] else 'f1_macro'

    start_time = time.time()
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    gs = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring=scoring_metric)

    with tqdm(total=1, desc='Procesando random forest', unit='modelo', leave=True) as pbar:
        gs.fit(x_train, y_train)
        pbar.update(1)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA + f" {execution_time} " + Fore.RESET + "segundos")

    mostrar_resultados(gs, x_dev, y_dev)
    save_model(gs)

# ===========================
# Algoritmo Naive Bayes
# ===========================

def naive_bayes():
    """
    Función que entrena un modelo de Naive Bayes utilizando GridSearchCV.
    """
    x_train, x_dev, y_train, y_dev = divide_data()

    params = args.parameters if hasattr(args, 'parameters') else {}
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0]
    }

    scoring_metric = f"f1_{params.get('f_score', 'macro')}" if params.get('f_score', 'macro') in ['macro', 'micro', 'weighted'] else 'f1_macro'

    start_time = time.time()

    nb = MultinomialNB()
    gs = GridSearchCV(nb, param_grid, cv=5, n_jobs=-1, scoring=scoring_metric)
    gs.fit(x_train, y_train)

    end_time = time.time()

    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA + f" {execution_time} " + Fore.RESET + "segundos")

    mostrar_resultados(gs, x_dev, y_dev)
    save_model(gs)

# ===========================
# Configuración inicial
# ===========================

def config():
    # Cargamos los datos
    print("\n- Cargando datos...")
    global data
    data = load_data(args.data_file)
    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

    # Preprocesamos los datos
    print("\n- Preprocesando datos...")
    preprocesar_datos()

    if args.algorithm == "kNN":
        try:
            kNN()
            print(Fore.GREEN + "Algoritmo kNN ejecutado con éxito" + Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
    elif args.algorithm == "decision_tree":
        try:
            decision_tree()
            print(Fore.GREEN + "Algoritmo árbol de decisión ejecutado con éxito" + Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
    elif args.algorithm == "random_forest":
        try:
            random_forest()
            print(Fore.GREEN + "Algoritmo random forest ejecutado con éxito" + Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
    elif args.algorithm == "naive_bayes":
        try:
            naive_bayes()
            print(Fore.GREEN + "Algoritmo naive bayes ejecutado con éxito" + Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
    else:
        print(Fore.RED + "Algoritmo no soportado" + Fore.RESET)
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