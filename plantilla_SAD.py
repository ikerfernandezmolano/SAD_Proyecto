# -*- coding: utf-8 -*-
import json
import sys
import sklearn as sk
import numpy as np
import pandas as pd
import joblib

# ===========================
# Funciones compartidas
# ===========================

def exampleMessage(algorithm):
    if algorithm.lower() == 'knn':
        print("""JSON ejemplo para kNN:
        {
            "data_file": "archivo csv",
            "algorithm": "kNN",
            "parameters": {
                "k": valor,
                "weights": "uniform/distance",
                "p": [1,2]
            }
        }""")
    elif algorithm.lower() == 'decision_tree':
        print("""JSON ejemplo para el árbol de decisión:
        {
            "data_file": "archivo csv",
            "algorithm": "decision_tree",
            "parameters": {
                "max_depth": [3, 6, 9],
                "min_samples_leaf": [1, 2],
                "criterion": ["gini", "entropy"]
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
    return data

def calculate_fscore(y_test, y_pred):
    """
    Función para calcular el F-score
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: F-score (micro), F-score (macro)
    """
    from sklearn.metrics import f1_score
    fscore_micro = f1_score(y_test, y_pred, average='micro')
    fscore_macro = f1_score(y_test, y_pred, average='macro')
    return fscore_micro, fscore_macro

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

def kNN(data, params):
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
    if not isinstance(k, int) or k<=0:
        print("Error en el valor k del algoritmo kNN.")
        exampleMessage("kNN")
        sys.exit(1)
    weights = params.get('weights')
    p = params.get('p')
    if not isinstance(p, int) or p not in [1, 2]:
        print("Error en el valor p del algoritmo kNN.")
        exampleMessage("kNN")
        sys.exit(1)

    # Seleccionamos las características y la clase
    X = data.iloc[:, :-1].values # Todas las columnas menos la última
    y = data.iloc[:, -1].values # Última columna
    
    # Dividimos los datos en entrenamiento y test
    from sklearn.model_selection import train_test_split
    np.random.seed(42)  # Set a random seed for reproducibility
    X_train, X_dev, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    # Escalamos los datos
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Entrenamos el modelo
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = k, weights = weights, p = p)
    classifier.fit(X_train, y_train)
    
    # Predecimos los resultados
    y_pred = classifier.predict(X_test)
    
    return y_test, y_pred

# ===========================
# Algoritmo Arból de Decisión
# ===========================

def decision_tree(data, params):
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

    # Seleccionamos las características y la clase
    X = data.iloc[:, :-1].values # Todas las columnas menos la última
    y = data.iloc[:, -1].values # Última columna
    
    # Dividimos los datos en entrenamiento y test
    np.random.seed(42)
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.20, stratify=y)
    
    # Obtenemos los parámetros del JSON (si no están, usamos los del guion por defecto)
    max_depth = params.get('max_depth')
    if not isinstance(max_depth, list) or not all(isinstance(i, int) and i > 0 for i in max_depth):
        print("Error en el valor max_depth del algoritmo Árbol de Decisión.")
        exampleMessage("decision_tree")
        sys.exit(1)
        
    min_samples_leaf = params.get('min_samples_leaf')
    if not isinstance(min_samples_leaf, list) or not all(isinstance(i, int) and i > 0 for i in min_samples_leaf):
        print("Error en el valor min_samples_leaf del algoritmo Árbol de Decisión.")
        exampleMessage("decision_tree")
        sys.exit(1)

    criterion = params.get('criterion')
    if not isinstance(criterion, list) or not all(isinstance(i, str) and i in ['gini', 'entropy'] for i in criterion):
        print("Error en el valor criterion del algoritmo Árbol de Decisión.")
        exampleMessage("decision_tree")
        sys.exit(1)


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
    
    # Guardar TODOS los resultados de los modelos en un CSV
    # clf.cv_results_ guarda las notas de todas las combinaciones que ha probado
    resultados_todos = pd.DataFrame(clf.cv_results_)
    # Filtramos las columnas para ver lo importante (parámetros y nota F-score)
    columnas_utiles = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    resultados_limpios = resultados_todos[columnas_utiles].sort_values(by='rank_test_score')
    resultados_limpios.to_csv('resultadosDeTodosModelos.csv', index=False)
    print("-> Archivo 'resultadosDeTodosModelos.csv' generado con éxito.")

    # Guardar el modelo ganador en disco
    # Extraemos el "cerebro" ganador y lo guardamos con joblib
    mejor_modelo = clf.best_estimator_
    joblib.dump(mejor_modelo, 'bestmodel.pkl')
    print("-> Modelo ganador guardado como 'bestmodel.pkl'.")
    
    # Predecimos los resultados sobre el conjunto Dev (X_dev) para ver qué tal lo hace
    y_pred = mejor_modelo.predict(X_dev)
    
    return y_dev, y_pred

# ===========================
# Configuración inicial
# ===========================

def config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Cargamos datos
    data = load_data(config["data_file"])

    # Seleccionamos el algoritmo
    algorithm = config["algorithm"]
    params = config.get("parameters", {})

    if algorithm == "kNN":
        y_test, y_pred = kNN(data, params)
    elif algorithm == "decision_tree":
        y_test, y_pred = decision_tree(data, params)
    else:
        raise ValueError(f"Algoritmo '{algorithm}' no soportado")

    # Resultados
    print("\nMatriz de confusión:")
    print(calculate_confusion_matrix(y_test, y_pred))

    print("\nF-score:")
    print(calculate_fscore(y_test, y_pred))

# ===========================
# Entrada principal
# ===========================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python3 main.py <config.json>")
        sys.exit(1)

    config(sys.argv[1])
    
