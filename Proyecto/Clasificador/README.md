# SAD_Proyecto

## 🧾 Preparación del entorno

La versión de Python a utilizar es **Python 3.12.3**

La primera vez que ejecutes el proyecto, prepara el entorno con:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Ejecución del entorno

Para utilizar la plantilla de entrenamiento:

```bash
source venv/bin/activate
python3 clasificador.py --json archivo_config.json --mode train/test [-v/--verbose] [-c/--cpu valor]
```

## ⚙ Config

```json
{
    "data_file": "archivo.csv",
    "algorithm": "kNN/decision_tree/random_forest/naive_bayes/logistic_regression",
    "prediction": "columna_a_predecir",
    "f_score": "macro/micro/weighted",
    "model": "modelo.pkl",
    "preprocessing": {
        "pnn": true,
        "sampling": "oversampling/undersampling/SMOTE",
        "scaler": "minmax/maxabs/zscore/standard",
        "unique_category_threshold": value,
        "text_process": "tf-idf/bow",
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