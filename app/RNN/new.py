import os
import googleapiclient.discovery
import pandas as pd
import numpy as np
import sys
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Embedding, LSTM, Dense
# from keras.utils import to_categorical

# Asegúrate de reemplazar 'TU_CLAVE_DE_API' con la clave de API que obtuviste.
api_key = 'AIzaSyCH5zNN2D-UKZVQDRLMwgp09hkEBaLLFWA'
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

def get_video_comments(video_id):

    if sys.stdout.encoding != 'utf-8':
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

    comments = []
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
            print(comment)
            comments.append(comment)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments


video_id = 'VoMyUgI-5NI'
comments = get_video_comments(video_id)


# print(comments)