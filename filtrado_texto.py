import os
import googleapiclient.discovery
import pandas as pd
import numpy as np
import sys
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re


# Asegúrate de descargar los recursos necesarios de NLTK antes de usarlo.
# Ejecuta esto una vez en tu entorno de Python para descargar los recursos:
nltk.download('punkt')
# nltk.download('stopwords')

# Lista de palabras vacías (stopwords) en español
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# Crear un objeto para realizar el stemming (extracción de raíces) de las palabras
stemmer = SnowballStemmer('spanish')

# Asegúrate de reemplazar 'TU_CLAVE_DE_API' con la clave de API que obtuviste.
api_key = 'AIzaSyCH5zNN2D-UKZVQDRLMwgp09hkEBaLLFWA'
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

def filtrar_palabras(texto):
    # Eliminar emojis usando expresiones regulares
    texto_sin_emojis = re.sub(r'[^\x00-\x7F]+', '', texto)

    # Tokenización
    tokens = word_tokenize(texto_sin_emojis, language='spanish')

    # Filtrar las palabras vacías (stopwords)
    palabras_filtradas = [palabra.lower() for palabra in tokens if palabra.lower() not in stop_words]

    # Eliminar signos de puntuación excesivos usando expresiones regulares
    texto_filtrado = ' '.join(palabras_filtradas)
    texto_filtrado = re.sub(r'[^\w\s]', ' ', texto_filtrado)

    # Aplicar stemming (extracción de raíces) a las palabras filtradas
    palabras_stemmed = [stemmer.stem(palabra) for palabra in word_tokenize(texto_filtrado, language='spanish')]

    # Unir las palabras filtradas y stemmizadas en un solo texto
    texto_filtrado = ' '.join(palabras_stemmed)

    return texto_filtrado

def analizar_sentimiento(texto):
    blob = TextBlob(texto)
    polaridad = blob.sentiment.polarity

    if polaridad >= 0.001:
        return "Positivo"
    elif polaridad < -0.2:
        return "Negativo"
    else:
        return "Neutro"

def get_video_comments(video_id):
    comments_with_sentiment = []
    next_page_token = None

    while True:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            pageToken=next_page_token,
            maxResults=100
        ).execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comment_filtrado = filtrar_palabras(comment)
            sentiment = analizar_sentimiento(comment)
            print(f"Comentario original: {comment}")
            print(f"Comentario filtrado: {comment_filtrado}")
            print(f"Sentimiento: {sentiment}\n")
            comments_with_sentiment.append((comment_filtrado, sentiment))

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments_with_sentiment

video_id = 'VoMyUgI-5NI'
comments_with_sentiment = get_video_comments(video_id)

# Crear un DataFrame con los comentarios filtrados y las etiquetas
df = pd.DataFrame(comments_with_sentiment, columns=['Comentario Filtrado', 'Etiqueta'])

# Guardar los comentarios y las etiquetas en un archivo CSV
df.to_csv('comentarios_con_etiquetas.csv', index=False, encoding='utf-8')
