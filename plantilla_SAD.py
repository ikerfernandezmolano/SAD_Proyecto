# -*- coding: utf-8 -*-
import json
import sys
import numpy as np
import pandas as pd
import csv
import os

# ===========================
# Funciones compartidas
# ===========================

def exampleMessage(algorithm):
    if algorithm.lower() == 'knn':
        print("""JSON ejemplo para kNN:
        {
            "data_file": "file.csv",
            "algorithm": "kNN",
            "parameters": {
                "k": {
                    "valueMin": value,
                    "valueMax": value,
                    "step": value
                },
                "weights": "uniform/distance",
                "p": (1,2),
                "f_score": "macro/micro"
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

def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    data = pd.read_csv(file)

    # Limpiar strings
    data = data.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Separar X e y
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Convertir TRUE/FALSE
    X = X.replace({True: 1, False: 0, "TRUE": 1, "FALSE": 0})

    # One-hot encoding SOLO en X
    X = pd.get_dummies(X)

    return X, y

def createCSV(Fscore):
    if not os.path.exists("Results.csv"):
        with open("Results.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["NombreMod","Precisión","Recall",f"F_score({Fscore})"])

def add_data(csv_file,data):
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)

def calculate_fscore(y_test, y_pred, f_score):
    """
    Función para calcular el F-score
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: F-score (micro), F-score (macro)
    """
    from sklearn.metrics import f1_score
    if f_score == "micro":
        return f1_score(y_test, y_pred, average='micro')
    elif f_score == "macro":
        return f1_score(y_test, y_pred, average='macro')
    else:
        fscore_micro = f1_score(y_test, y_pred, average='micro')
        fscore_macro = f1_score(y_test, y_pred, average='macro')
        return fscore_micro, fscore_macro

def calculate_prec_recall(y_test, y_pred, avg):
    from sklearn.metrics import precision_score, recall_score

    precision = precision_score(y_test, y_pred, average=avg)
    recall = recall_score(y_test, y_pred, average=avg)

    return precision, recall

def calculate_confusion_matrix(y_test, y_pred):
    """
    Función para calcular la matriz de confusión
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Matriz de confusión
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm

# ===========================
# Algoritmo kNN
# ===========================

def kNN(X, y, params):
    """
    Función para implementar el algoritmo kNN
    
    :param data: Datos a clasificar
    :type data: pandas.DataFrame
    :param k: Número de vecinos más cercanos
    :type k: int
    :param weights: Pesos utilizados en la predicción ('uniform' o 'distance')
    :type weights: str
    :param p: Parámetro para la distancia métrica (1 para Manhattan, 2 para Euclídea)
    :type p: int
    :return: Clasificación de los datos
    :rtype: tuple
    """
    k = params.get('k')
    kMin = k.get('valueMin')
    kMax = k.get('valueMax')
    kStep = k.get('step')
    if not isinstance(kMin, int) or not isinstance(kMax, int) or not isinstance(kStep, int) or kMin<=0 or kMax<=0 or kStep<=0:
        print("Error en el valor k del algoritmo kNN.")
        exampleMessage("kNN")
        sys.exit(1)
    weights = params.get('weights')
    p = params.get('p')
    if not isinstance(p, int) or p not in [1, 2]:
        print("Error en el valor p del algoritmo kNN.")
        exampleMessage("kNN")
        sys.exit(1)
    f_score = params.get('f_score').lower()
    if not isinstance(f_score, str) or f_score not in ["macro","micro","avg","none"]:
        f_score = "macro"
    
    # Dividimos los datos en entrenamiento y test
    from sklearn.model_selection import train_test_split
    np.random.seed(42)  # Set a random seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    
    # Escalamos los datos
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    createCSV(f_score)
    
    # Entrenamos el modelo
    from sklearn.neighbors import KNeighborsClassifier
    for paramK in range(kMin,kMax+1,kStep):
        classifier = KNeighborsClassifier(n_neighbors = paramK, weights = weights, p = p)
        classifier.fit(X_train, y_train)

        # Predecimos los resultados
        y_pred = classifier.predict(X_test)

        data = [f"KNN_k{paramK}_p{p}_w{weights}"]
        data.extend(calculate_prec_recall(y_test, y_pred, f_score))
        fscore = calculate_fscore(y_test, y_pred, f_score)

        if isinstance(fscore, tuple):
            data.extend(fscore)
        else:
            data.append(fscore)

        add_data("Results.csv",data)
    
    return y_test, y_pred

# ===========================
# Algoritmo Arból de Decisión
# ===========================

def decision_tree(X, y, params):
    """
    Función para implementar el algoritmo Árboles de Decisión con barrido.
    
    :param data: Datos a clasificar
    :type data: pandas.DataFrame
    :param params: Parámetros para el árbol de decisión (max_depth, min_samples_leaf, criterion)
    :type params: dict
    :return: Clasificación de los datos
    :rtype: tuple
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split
    
    # Dividimos los datos en entrenamiento y test
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    # Obtenemos los parámetros del JSON (si no están, usamos los del guion por defecto)
    max_depth = params.get('max_depth', [3, 6, 9])
    min_samples_leaf = params.get('min_samples_leaf', [1, 2])
    criterion = params.get('criterion', ['gini', 'entropy'])
    
    parametros_dt = {
        'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf,
        'criterion': criterion
    }
    
    # Configuramos el árbol y el buscador de la mejor combinación (GridSearchCV)
    dt = DecisionTreeClassifier(random_state=42)
    # cv=5 significa que hace validación cruzada. n_jobs=-1 usa todos los procesadores para ir rápido
    clf = GridSearchCV(dt, parametros_dt, cv=5, n_jobs=-1, scoring='f1_macro')
    
    # Entrenamos el modelo (aquí prueba todas las combinaciones automáticamente)
    print("Entrenando Árbol de Decisión y buscando los mejores parámetros...")
    clf.fit(X_train, y_train)
    
    print("\n¡Barrido completado!")
    print(f"Mejores parámetros encontrados: {clf.best_params_}")
    
    # Predecimos los resultados usando la mejor combinación que ha encontrado
    y_pred = clf.predict(X_test)
    
    return y_test, y_pred

# ===========================
# Configuración inicial
# ===========================

def config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Cargamos datos
    X, y = load_data(config["data_file"])

    # Seleccionamos el algoritmo
    algorithm = config["algorithm"]
    params = config.get("parameters", {})

    if algorithm == "kNN":
        y_test, y_pred = kNN(X, y, params)
    elif algorithm == "decision_tree":
        y_test, y_pred = decision_tree(X, y, params)
    else:
        raise ValueError(f"Algoritmo '{algorithm}' no soportado")

    # Resultados
    print("\nMatriz de confusión:")
    print(calculate_confusion_matrix(y_test, y_pred))

# ===========================
# Entrada principal
# ===========================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python3 plantilla_SAD.py <config.json>")
        sys.exit(1)

    config(sys.argv[1])
    
