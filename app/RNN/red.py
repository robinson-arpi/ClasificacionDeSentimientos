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
file_path = 'ClasificacionDeSentimientos/app/RNN/Opiniones.csv'
df = pd.read_csv(file_path)

# Paso 2: Preprocesar los datos
opinions = df['opinion'].values
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
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

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
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Paso 7: Entrenar el modelo
epochs = 30
batch_size = 64
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Paso 8: Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Obtén las predicciones del modelo para el conjunto de pruebas
y_pred = model.predict(X_test)
# Convertir las predicciones a etiquetas
y_pred_classes = np.argmax(y_pred, axis=1)

# Crear la matriz de confusión
cm = confusion_matrix(y_test, y_pred_classes)

# Dibujar la matriz de confusión usando seaborn
# plt.figure(figsize=(10,7))
# sns.heatmap(cm, annot=True, fmt='d')
# plt.xlabel('Predicho')
# plt.ylabel('Real')
# plt.show()

# Imprimir la matriz de confusión
cm_df = pd.DataFrame(cm, 
                     index = ['Negativo', 'Neutro', 'Positivo'], 
                     columns = ['Negativo', 'Neutro', 'Positivo'])

# Imprimir el DataFrame
print('\n Matriz de confusión:\n')
print(cm_df)
print()

def predict_sentiment(model, tokenizer, text):
    # Tokenizar el texto
    sequence = tokenizer.texts_to_sequences([text])

    # Rellenar la secuencia
    sequence = pad_sequences(sequence, maxlen=max_sequence_length)

    # Realizar la predicción
    prediction = model.predict(sequence)

    # Obtener la clase con la probabilidad más alta
    class_pred = np.argmax(prediction)

    # Mapear la etiqueta de clase de nuevo a la etiqueta original
    label_mapping_inverse = {2: 1, 1: 0, 0: -1}
    class_pred = label_mapping_inverse[class_pred]

    # Imprimir la predicción
    if class_pred == 1:
        print('La frase: \"', text, '\" tiene un sentimiento positivo.')
    elif class_pred == 0:
        print('La frase: \"', text, '\" tiene un sentimiento neutro.')
    else:
        print('La frase: \"', text, '\" tiene un sentimiento negativo.')

# Prueba la función con una nueva frase
predict_sentiment(model, tokenizer, 'El presidente viajará a China')
predict_sentiment(model, tokenizer, 'El presidente ha descepcionado al pueblo')
predict_sentiment(model, tokenizer, 'El presidente no ha cumplido con las obras ofrecidas')
predict_sentiment(model, tokenizer, 'El presidente es muy buen presidente')