import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight

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
word_index = tokenizer.word_index

# Cargar los embeddings
embedding_dim = 300
embeddings_index = {}
with open('ClasificacionDeSentimientos/app/RNN/cc.es.300.vec', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Crear la matriz de embeddings
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i < num_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

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

# # Crear un objeto SMOTE
# smote = SMOTE()

# # Aplicar SMOTE a los datos para tratar de balancearlos
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


# Paso 6: Construir el modelo RNN
model = Sequential()
model.add(Embedding(num_words, embedding_dim, embeddings_initializer=Constant(embedding_matrix),
                    input_length=max_sequence_length, trainable=True))
model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.3, return_sequences=True)) # return_sequences=True para poder apilar otra capa LSTM
model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.3)) # No necesitamos return_sequences=True para la última capa LSTM
model.add(Dense(32, activation='relu')) # Capa densa adicional
model.add(Dense(3, activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Calcular los pesos de las clases
class_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Paso 7: Entrenar el modelo
epochs = 30
batch_size = 32
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, sample_weight=class_weights)


# # Paso 8: Evaluar el modelo
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Loss: {loss}, Accuracy: {accuracy}')

# Paso 9: Guardar el modelo entrenado
model.save('ClasificacionDeSentimientos/app/RNN/modelo.h5')
print("Modelo guardado correctamente.")

# # Obtén las predicciones del modelo para el conjunto de pruebas
# y_pred = model.predict(X_test)
# # Convertir las predicciones a etiquetas
# y_pred_classes = np.argmax(y_pred, axis=1)

# # Crear la matriz de confusión
# cm = confusion_matrix(y_test, y_pred_classes)

# Dibujar la matriz de confusión usando seaborn
# plt.figure(figsize=(10,7))
# sns.heatmap(cm, annot=True, fmt='d')
# plt.xlabel('Predicho')
# plt.ylabel('Real')
# plt.show()

# # Imprimir la matriz de confusión
# cm_df = pd.DataFrame(cm, 
#                      index = ['Negativo', 'Neutro', 'Positivo'], 
#                      columns = ['Negativo', 'Neutro', 'Positivo'])

# # Imprimir el DataFrame
# print('\n Matriz de confusión:\n')
# print(cm_df)
# print()


# def predict_sentiment(model, tokenizer, text):
#     # Tokenizar el texto
#     sequence = tokenizer.texts_to_sequences([text])

#     # Rellenar la secuencia
#     sequence = pad_sequences(sequence, maxlen=max_sequence_length)

#     # Realizar la predicción
#     prediction = model.predict(sequence)

#     # Obtener la clase con la probabilidad más alta
#     class_pred = np.argmax(prediction)

#     # Mapear la etiqueta de clase de nuevo a la etiqueta original
#     label_mapping_inverse = {2: 1, 1: 0, 0: -1}
#     class_pred = label_mapping_inverse[class_pred]

#     # Imprimir la predicción
#     if class_pred == 1:
#         print('La frase: \"', text, '\" tiene un sentimiento positivo.')
#     elif class_pred == 0:
#         print('La frase: \"', text, '\" tiene un sentimiento neutro.')
#     else:
#         print('La frase: \"', text, '\" tiene un sentimiento negativo.')


# # Prueba la función con nuevos comentarios
# predict_sentiment(model, tokenizer, 'No ha hecho nada por el Ecuador')
# predict_sentiment(model, tokenizer, 'Solo ha buscado su propio beneficio')
# predict_sentiment(model, tokenizer, 'El correato va a regresar')
# predict_sentiment(model, tokenizer, 'No tengo ninguna opinión sobre Lasso')
# print()