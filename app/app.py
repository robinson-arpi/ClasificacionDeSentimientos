from flask import Flask, render_template, request, jsonify
import pandas as pd
from RNN.filtrado_texto import get_video_comments
from keras.models import load_model
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Carga del modelo entrenado
loaded_model = load_model('ClasificacionDeSentimientos/app/RNN/modelo.h5')

file_path = 'ClasificacionDeSentimientos/app/RNN/Comentarios1.csv'
df = pd.read_csv(file_path)
opinions = df['comentario'].values
num_words = 10000
tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
tokenizer.fit_on_texts(opinions)
sequences = tokenizer.texts_to_sequences(opinions)
max_sequence_length = max(len(seq) for seq in sequences)

app = Flask(__name__)

# Función para convertir el objeto Counter a un diccionario
def counter_to_dict(counter):
    return dict(counter)

# Ruta para cargar la página inicial
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar los datos y devolver el JSON con los resultados
@app.route('/procesar', methods=['POST'])
def procesar():
    # Obtener los datos del formulario en el HTML
    id_video = request.form.get('videoId')

    # obtiene los comentarios del video
    comentarios_video = get_video_comments(id_video)

    total = len(comentarios_video)

    #Listas
    l_positivos = []
    l_negativos = []
    l_neutros = []


    # Uso del modelo para predicciones
    for x in comentarios_video:
        sequence = tokenizer.texts_to_sequences([x[1]])
        sequence = pad_sequences(sequence, maxlen=max_sequence_length)
        prediction = loaded_model.predict(sequence)
        class_pred = np.argmax(prediction)
        label_mapping_inverse = {2: 1, 1: 0, 0: -1}
        class_pred = label_mapping_inverse[class_pred]

        print(x[0], class_pred)

        if class_pred == 1:
            l_positivos.append(x[0])
        elif class_pred == 0:
            l_neutros.append(x[0])
        else:
            l_negativos.append(x[0])

    t_pos = len(l_positivos)
    t_neg = len(l_negativos)
    t_neu = len(l_neutros)

    if t_pos > 10: l_positivos[:10]
    if t_neg > 10: l_negativos[:10]

    # Convertir los datos numéricos a enteros para que puedan ser serializados a JSON
    resultado_json = {
        'positivo': t_pos,
        'negativo': t_neg,
        'neutro': t_neu,
        'lista_positivos': l_positivos,
        'lista_negativos': l_negativos
    }

    # Convertir el JSON a formato de cadena y devolverlo como respuesta
    return jsonify(resultado_json)

if __name__ == '__main__':
    app.run(debug=True)
