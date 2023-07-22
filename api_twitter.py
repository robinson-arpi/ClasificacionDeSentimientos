import tweepy
from textblob import TextBlob
import pandas as pd

# bearer token AAAAAAAAAAAAAAAAAAAAAC%2FFowEAAAAAtGQQH8YLmRE4yNNi5DlHbSwTD7M%3DhBMhhpCBbnfDrQS5CcfsyNmLhiTzIGGdrX9WsKtrhYSchJygkF
# Configuración de las claves de acceso de la API de Twitter
consumer_key = 'luXNhrLInuRiTGt7WN2jLG15n'
consumer_secret = 'tvsVTatbIKvHJgwJvNzgZcpQgfpy9f2xvjgl0knLuWkJGGsXiB'
access_token = '707966186202517504-jG3ISML7Zzex3LbcIo4Dn2gBnWGIlPl'
access_token_secret = 'xf1rXo4UaSO79xTd37EgtYVrjk5GCNeEl8YXIRuMYgGb8'

# Autenticación con la API de Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Palabras clave a buscar
keywords = "Ecuador lasso OR elecciones OR seguridad OR correa"

# Recopilación de tweets con las palabras clave
tweets_data = []
for tweet in tweepy.Cursor(api.search, q=keywords, lang='es', tweet_mode='extended').items(100):
    tweets_data.append({
        'Texto': tweet.full_text,
        'Fecha': tweet.created_at,
        'Usuario': tweet.user.screen_name,
        'Seguidores': tweet.user.followers_count
    })

# Convertir la lista de tweets en un DataFrame de pandas
df = pd.DataFrame(tweets_data)

# Guardar los datos en un archivo CSV
df.to_csv('tweets_data.csv', index=False)

print("Tweets guardados exitosamente en tweets_data.csv")