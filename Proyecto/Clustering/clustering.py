# -*- coding: utf-8 -*-
import sys
import argparse
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from colorama import Fore
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import gensim.corpora as corpora
from gensim.models import LdaModel, CoherenceModel
import matplotlib.pyplot as plt
from gensim.models import Phrases

# Variables globales
global data
global args
global num_mejor_k
global mejor_k 

def parse_args():
    # Guarda en args los argumentos del json
    parser = argparse.ArgumentParser(description=" NLP y TF-IDF")
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

def load_and_filter_data(sentiment):
    """Carga el CSV y filtra por el sentimiento deseado."""
    global data
    try:
        df = pd.read_csv(args.data_file, encoding='utf-8', sep=';')
        
        if args.sentiment_column not in df.columns or args.text_column not in df.columns:
            print(f"Error: Faltan columnas requeridas en el archivo CSV.")
            print(f"El programa buscaba: '{args.text_column}' y '{args.sentiment_column}'")
            print(f"Columnas que realmente tiene tu CSV: {list(df.columns)}")
            sys.exit(1)

        # Mapeo de la escala 1-5 a las categorías de texto
        if sentiment == "negative":
            data = df[df[args.sentiment_column].isin([1, 2])].copy()
        elif sentiment == "neutral":
            data = df[df[args.sentiment_column] == 3].copy()
        elif sentiment == "positive":
            data = df[df[args.sentiment_column].isin([4, 5])].copy()
        else:
            data = pd.DataFrame() # Por si acaso llega un sentimiento no reconocido
        
        if data.empty:
            print(f"Aviso: No se encontraron registros para la categoría '{sentiment}'.")
            return
            
        print(f"Carga exitosa: {len(data)} registros de la categoría '{sentiment}'.")
    except Exception as e:
        print("Error al procesar el archivo de datos.")
        print(e)
        data = pd.DataFrame()
        return

def simplify_text():
    """Limpieza de datos: tokenización, eliminación de stopwords, lematización y filtrado de números."""
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
        return tokens

    # Creamos una nueva columna con el texto ya listo
    data['tokens'] = data[args.text_column].fillna("").apply(clean_text)
    
    print(Fore.GREEN + "Texto simplificado con éxito." + Fore.RESET)

def prepare_gensim_corpus():
    """Crea el Diccionario y el Bag of Words que necesita LDA."""
    global data
    modo_ngram = getattr(args, 'n_gram', 'unigram').lower()
    print("\n- Preparando Diccionario y Corpus para LDA...")
    
    if modo_ngram in ['bigram', 'trigram']:
        # 1. Creamos y aplicamos Bigramas
        bigram_phrases = Phrases(data['tokens'], min_count=5, threshold=10)
        data['tokens'] = [bigram_phrases[doc] for doc in data['tokens']]
        
        if modo_ngram == 'trigram':
            # 2. Si piden Trigramas, pasamos el modelo otra vez sobre los bigramas
            trigram_phrases = Phrases(data['tokens'], min_count=5, threshold=10)
            data['tokens'] = [trigram_phrases[doc] for doc in data['tokens']]
            print(Fore.GREEN + "Generación de Trigramas completada." + Fore.RESET)
        else:
            print(Fore.GREEN + "Generación de Bigramas completada." + Fore.RESET)
    else:
        print(Fore.GREEN + "Generación de Unigramas (sin alterar palabras) completada." + Fore.RESET)

    # Crea un diccionario con todas las palabras únicas
    id2word = corpora.Dictionary(data['tokens'])
    
    # Filtra palabras muy raras (aparecen en menos de 5 reviews) o muy comunes (en más del 85%)
    id2word.filter_extremes(no_below=5, no_above=0.85)
    
    # Convierte cada reseña en una bolsa de palabras (Bag of Words) con su ID y frecuencia
    corpus = [id2word.doc2bow(text) for text in data['tokens']]
    
    print(Fore.GREEN + f"Diccionario creado con {len(id2word)} palabras clave únicas." + Fore.RESET)
    return id2word, corpus

def run_final_model(id2word, corpus, sentiment, optimal_k):
    """Entrena el modelo final y exporta los resultados clasificados."""
    global data
    
    print(Fore.CYAN + f"\n=== Iniciando entrenamiento del modelo final (K={optimal_k}) ===" + Fore.RESET)
    
    # Entrenar el modelo final
    final_model = LdaModel(corpus=corpus, num_topics=optimal_k, id2word=id2word, random_state=42, passes=15)
    
    # Imprimir las palabras clave de cada tópico
    print(Fore.YELLOW + "\nPalabras clave por tópico:" + Fore.RESET)
    for topic_num, words in final_model.print_topics(num_words=10):
        clean_words = [word.split('*')[1].replace('"', '') for word in words.split(' + ')]
        print(f"Tópico {topic_num}: {', '.join(clean_words)}")

    # Asignar el tópico dominante y su porcentaje a cada registro
    print("\n- Clasificando registros y determinando probabilidades...")
    
    def get_dominant_topic_info(bow):
        topic_probs = final_model.get_document_topics(bow)
        best_topic = sorted(topic_probs, key=lambda x: x[1], reverse=True)[0]
        
        topic_id = best_topic[0]
        topic_percentage = round(best_topic[1] * 100, 2)
        
        return topic_id, f"{topic_percentage}%"

    data['Topico_Dominante'], data['Porcentaje_Similitud'] = zip(*[get_dominant_topic_info(bow) for bow in corpus])
    
    # Exportar los resultados
    if not os.path.exists('output'):
        os.makedirs('output')
        
    modo_ngram = getattr(args, 'n_gram', 'unigram').lower()
    folder_path = os.path.join('output', args.company_name, modo_ngram)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    output_csv = os.path.join(folder_path, f'Clasificacion_{sentiment}.csv')
    columnas_finales = [args.text_column, args.sentiment_column, 'Topico_Dominante', 'Porcentaje_Similitud']
    data[columnas_finales].to_csv(output_csv, sep=';', index=False, encoding='utf-8')
    
    print(Fore.GREEN + f"\nProceso finalizado. Exportación generada en: {output_csv}" + Fore.RESET)


def calculate_lda_coherence(id2word, corpus, sentiment):
    """Ejecuta LDA varias veces y grafica la coherencia C_V para encontrar el K óptimo."""
    global data

    print(Fore.CYAN + "\nCalculando métrica de coherencia LDA. Este proceso requiere tiempo..." + Fore.RESET)
    
    coherence_values = []
    K_rango = range(args.lda["k_min"], args.lda["k_max"] + 1, args.lda["step"])
    
    mejor_k_valor = 0
    num_mejor_k = 0

    for k in K_rango:
        # Entrenamos el modelo LDA
        model = LdaModel(corpus=corpus, num_topics=k, id2word=id2word, random_state=42, passes=10)
        
        # Calculamos la métrica C_v (Coherencia)
        coherencemodel = CoherenceModel(model=model, texts=data['tokens'], dictionary=id2word, coherence='c_v')
        coherencia = coherencemodel.get_coherence()
        if num_mejor_k == 0 or coherencia > mejor_k_valor:
            mejor_k_valor = coherencia
            num_mejor_k = k
        
        coherence_values.append(coherencia)
        print(f"  -> Terminado LDA k={k} | Coherencia (C_V): {coherencia:.4f}")
        
    # Dibujamos y guardamos el gráfico
    plt.figure(figsize=(8, 5))
    plt.plot(K_rango, coherence_values, marker='o', linestyle='-', color='g')
    plt.title(f'Gráfico de Coherencia LDA ({sentiment.capitalize()})')
    plt.xlabel('Número de Tópicos (k)')
    plt.ylabel('Coherencia C_V')
    plt.grid(True)
    
    if not os.path.exists('output'):
        os.makedirs('output')
        
    modo_ngram = getattr(args, 'n_gram', 'unigram').lower()
    folder_path = os.path.join('output', args.company_name, modo_ngram)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    ruta_grafico = os.path.join(folder_path, f'grafico_coherencia_{sentiment}.png')
    plt.savefig(ruta_grafico)
    print(Fore.GREEN + f"\nGráfico de coherencia guardado en: {ruta_grafico}" + Fore.RESET)

    return num_mejor_k

if __name__ == "__main__":
    print(Fore.CYAN + "Iniciando Clustering" + Fore.RESET)
    args = parse_args()
    

    for sentiment in args.target_sentiment:
        print(f"\n--- Procesando datos para el sentimiento: {sentiment.upper()} ---")
        
        # 1. Cargar datos
        load_and_filter_data(sentiment)
        # Si el DataFrame está vacío, pasamos al siguiente sentimiento
        if data.empty:
            continue 
        # 2. Limpiar texto
        simplify_text()
        # 3. Preparar formato LDA
        id2word, corpus = prepare_gensim_corpus()
        # 4. Calcular y graficar K óptimo
        k_ganador = calculate_lda_coherence(id2word, corpus, sentiment)
        # 5. Entrenar modelo final y exportar resultados
        run_final_model(id2word, corpus, sentiment, k_ganador)