# -*- coding: utf-8 -*-
import sys
import argparse
import pandas as pd
import json
import os
from colorama import Fore

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Variables globales
global data
global args

def parse_args():
    # Guarda en args los argumentos del json
    parser = argparse.ArgumentParser(description="Bloque 2: NLP y TF-IDF")
    parser.add_argument("-j", "--json", help="Archivo de configuración JSON", required=True)
    args = parser.parse_args()

    try:
        # Cargamos el JSON y actualizamos args con sus valores
        with open(args.json, 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(args, key, value)
    except FileNotFoundError:
        print(Fore.RED + f"Error: No se encontró el archivo {args.json}" + Fore.RESET)
        sys.exit(1)
    return args

def load_and_filter_data():
    """Carga el CSV y filtra por el sentimiento deseado."""
    global data
    try:
        # Usamos sep=';' como descubrimos en el paso anterior
        df = pd.read_csv(args.data_file, encoding='utf-8', sep=';')
        
        if args.sentiment_column not in df.columns or args.text_column not in df.columns:
            print(Fore.RED + "Error: Faltan columnas en el CSV." + Fore.RESET)
            sys.exit(1)

        # Filtramos solo las opiniones del sentimiento objetivo
        data = df[df[args.sentiment_column] == args.target_sentiment].copy()
        
        if data.empty:
            print(Fore.RED + "Aviso: No se encontraron opiniones." + Fore.RESET)
            sys.exit(1)
            
        print(Fore.GREEN + f"ÉXITO: Se han cargado {len(data)} opiniones de tipo '{args.target_sentiment}'." + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error fatal al cargar los datos." + Fore.RESET)
        print(e)
        sys.exit(1)

def simplify_text():
    """Limpia el texto usando NLTK (minúsculas, sin puntuación, sin stopwords, lematizado)."""
    global data
    print("\n- Limpiando y simplificando el texto (esto puede tardar unos segundos)...")
    
    # Descargamos las herramientas silenciosamente
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    stop_words = set(stopwords.words(args.language))
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+') # Solo extrae caracteres alfanuméricos

    def clean_text(text):
        text = str(text).lower()
        tokens = tokenizer.tokenize(text)
        # Filtramos números y palabras vacías, y lematizamos
        tokens = [lemmatizer.lemmatize(word) for word in tokens if not word.isnumeric() and word not in stop_words]
        return " ".join(tokens)

    # Creamos una nueva columna con el texto ya listo
    data['texto_limpio'] = data[args.text_column].fillna("").apply(clean_text)
    
    print(Fore.GREEN + "Texto simplificado con éxito." + Fore.RESET)
    print("\nCompara el antes y el después:")
    print(data[[args.text_column, 'texto_limpio']].head(2))

def process_tfidf():
    """Convierte el texto en matriz TF-IDF."""
    global data
    print("\n- Aplicando vectorización TF-IDF...")
    
    # max_df=0.85: ignora palabras que aparecen en más del 85% de los comentarios
    # min_df=5: ignora palabras que aparecen en menos de 5 comentarios (suelen ser erratas)
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=5)
    X = vectorizer.fit_transform(data['texto_limpio'])
    
    print(Fore.GREEN + f"Matriz TF-IDF creada. Tamaño: {X.shape}" + Fore.RESET)
    print(f"(Tenemos {X.shape[0]} opiniones y un vocabulario de {X.shape[1]} palabras clave únicas)")
    
    return X, vectorizer

if __name__ == "__main__":
    print(Fore.CYAN + "=== Iniciando Clustering: Bloque 1 y 2 ===" + Fore.RESET)
    args = parse_args()
    
    # Bloque 1
    load_and_filter_data()
    
    # Bloque 2
    simplify_text()
    X, vectorizer = process_tfidf()