# COACHELLA CLASIFICADOR

## 🧾 Preparación del entorno

La versión de Python a utilizar es **Python 3.12.3**

La primera vez que ejecutes el proyecto, prepara el entorno con:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

De esta manera, se instalarán las librerías necesarias para la ejecución del programa.

Para cerrar el entorno virtual:

```bash
deactivate
```

## ▶️ Ejecución del entorno

Para utilizar la plantilla:

```bash
source venv/bin/activate
python3 clasificador.py -j/--json archivo_config.json -m/--mode train/test [-v/--verbose] [-c/--cpu valor]
```

* -j o --json, será la ruta al archivo de configuración de tipo JSON, del cual más abajo se encuentra una plantilla.
* -m o --mode, servirá para especificar el modo en el que se quiere utilizar la plantilla. Train, para generar modelos 
(En el JSON deberán estar indicados 2 archivos csv, el train y el dev). Test, para probar el modelo generado 
(En el JSON deberá especificarse en data_file1 la ruta al archivo csv a predecir y en el campo model deberá especificarse
la ruta para el archivo .sav)
* -v o --verbose, durante el Train mostrará información relativa al modelo generado, así como, la matriz de confusión 
, entre otros datos.
* -c o --cpu, servirá para especificar el número de núcleos que se quieren utilizar durante la ejecución.

Los archivos generado durante la ejecución del programa serán guardados en la carpeta output/ en la misma ubicación donde se encuentre el clasificador.py.

## ⚙ Config

```json
{
    "data_file1": "ruta_archivoTRAINTEST.csv",
    "data_file2": "ruta_archivoDEV.csv",
    "algorithm": "kNN/decision_tree/random_forest/naive_bayes/logistic_regression",
    "prediction": "columna_a_predecir",
    "f_score": "macro/micro/weighted",
    "model": "ruta_al_modelo.sav",
    "sep": ";",
    "preprocessing": {
        "pnn": true/false,
        "sampling": "none/oversampling/undersampling/SMOTE/ADASYN",
        "scaler": "minmax/maxabs/zscore/standard",
        "unique_category_threshold": value,
        "text_process": "tf-idf/bow",
        "ngram_range": [value1, value2],
        "language": "language",
        "impute_num": "mean/mode/median/constant/delete",
        "impute_cat": "mode/constant/delete",
        "drop_features": ["campo1", "campo2", ..., "campoN"]
    },
    "kNN": {
        "k": {
            "kValues" : [value1, value2, ..., valueN],
            "valueMin": valueMin,
            "valueMax": valueMax,
            "step": valueStep
        },
        "weights": ["uniform","distance"],
        "p": [1,2]
    },
    "decision_tree": {
        "max_depth": [value1, value2, ..., valueN],
        "min_samples_split": [value1, value2, ..., valueN],
        "min_samples_leaf": [value1, value2, ..., valueN],
        "criterion": ["gini", "entropy"]
    },
    "random_forest": {
        "n_estimators": [value1, value2, ..., valueN],
        "max_depth": [value1, value2, ..., valueN],
        "min_samples_split": [value1, value2, ..., valueN],
        "min_samples_leaf": [value1, value2, ..., valueN],
        "criterion": ["gini", "entropy"]
    },
    "naive_bayes": {
        "alpha": [value1, value2, ..., valueN],
        "fit_prior": [true, false]
    },
    "logistic_regression": {
        "C": [value1, value2, ..., valueN],
        "l1_ratio": [0,1],
        "solver": ["lbfgs", "saga"],
        "max_iter" : [value1, value2, ..., valueN]
    }
}
```

**EXPLICACIÓN CAMPOS AMBÍGUOS**
* "pnn" representa en True la utilización de las clases Positive, Neutral y Negative.
* "unique_category_threshold" si una columna tiene más valores únicos que este umbral, se trata como texto en lugar de variable categórica
* "drop_features" columnas que no se quieren tener en cuenta.
* "kValues" para valores específicos o "valueMin", "valueMax" y "step" para rango.
* "fit_prior" si se aprenden o no las probabilidades a priori de las clases según su frecuencia en los datos.
* "l1_ratio" 0 para l2 y 1 para l1.

### EJEMPLO PARA GENERAR EL MEJOR MODELO

```bash
source venv/bin/activate
python3 clasificador.py -j config.json -m train -v
```

```json
{
    "data_file1": "../Datos/Spotify_TrainGen.csv",
    "data_file2": "../Datos/Spotify_Dev.csv",
    "algorithm": "naive_bayes",
    "prediction": "score",
    "f_score": "macro",
    "model": "ruta_al_modelo.sav",
    "sep": ";",
    "preprocessing": {
        "pnn": true,
        "sampling": "none",
        "scaler": "zscore",
        "unique_category_threshold": 10,
        "text_process": "tf-idf",
        "ngram_range": [1, 3],
        "language": "english",
        "impute_num": "median",
        "impute_cat": "mode",
        "drop_features": ["reviewId", "date"]
    },
    "naive_bayes": {
        "alpha": [0.5],
        "fit_prior": [false]
    }
}
```

### EJEMPLO PARA PROBAR EL MEJOR MODELO

```bash
source venv/bin/activate
python3 clasificador.py -j config.json -m test
```

```json
{
    "data_file1": "../Datos/Spotify_Test.csv",
    "prediction": "score",
    "model": "output/Modelonaive_bayes.sav",
    "sep": ";"
    "preprocessing": {}
}
```
