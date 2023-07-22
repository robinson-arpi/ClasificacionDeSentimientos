import os
import googleapiclient.discovery
import pandas as pd
import numpy as np
import sys
from textblob import TextBlob

# Asegúrate de reemplazar 'TU_CLAVE_DE_API' con la clave de API que obtuviste.
api_key = 'AIzaSyCH5zNN2D-UKZVQDRLMwgp09hkEBaLLFWA'
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

def get_video_comments(video_id):

    if sys.stdout.encoding != 'utf-8':
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

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
            sentiment = analizar_sentimiento(comment)  # Análisis de sentimiento utilizando TextBlob
            print(f"Comentario: {comment}")
            print(f"Sentimiento: {sentiment}\n")
            comments_with_sentiment.append((comment, sentiment))

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments_with_sentiment


def analizar_sentimiento(texto):
    blob = TextBlob(texto)
    polaridad = blob.sentiment.polarity

    if polaridad > 0:
        return "Positivo"
    elif polaridad < 0:
        return "Negativo"
    else:
        return "Neutro"


video_id = '439ujyWOClE'
comments_with_sentiment = get_video_comments(video_id)

# Guardar los comentarios y los sentimientos en un archivo de texto
with open('comentarios_con_sentimiento.txt', 'w', encoding='utf-8') as file:
    for comment, sentiment in comments_with_sentiment:
        file.write(f"Comentario: {comment}\n")
        file.write(f"Sentimiento: {sentiment}\n\n")
