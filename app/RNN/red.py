import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Leer el archivo CSV
file_path = 'ClasificacionDeSentimientos/app/RNN/Comentarios1.csv'
df = pd.read_csv(file_path)

# Paso 2: Preprocesar los datos
opinions = df['comentario'].values
labels = df['etiqueta'].values

# Mapear las etiquetas a valores enteros no negativos (0, 1, 2)
label_mapping = {1: 2, 0: 1, -1: 0}
labels = [label_mapping[label] for label in labels]

# Paso 3: Tokenizar los textos
num_words = 10000
tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
tokenizer.fit_on_texts(opinions)
sequences = tokenizer.texts_to_sequences(opinions)

# Paso 4: Padding (rellenar secuencias para que todas tengan la misma longitud)
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Paso 5: Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.05, random_state=42)

# Convertir X_train, y_train, X_test y y_test a numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Reshape X_train y X_test si es necesario
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Paso 6: Construir el modelo RNN
model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=64, input_length=max_sequence_length))
model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.3, return_sequences=True)) # return_sequences=True para poder apilar otra capa LSTM
model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.3)) # No necesitamos return_sequences=True para la última capa LSTM
model.add(Dense(32, activation='relu')) # Capa densa adicional
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Paso 7: Entrenar el modelo
epochs = 30
batch_size = 64
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Guardar el modelo
model.save('ClasificacionDeSentimientos/app/RNN/modelo.h5')
print("Modelo guardado correctamente.")
