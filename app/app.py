from flask import Flask, render_template, request, jsonify
import pandas as pd
from textblob import TextBlob
from collections import Counter

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
    textos = request.form.get('textos')
    etiquetas = request.form.get('etiquetas').split(',')

    # Crear un DataFrame con los datos ingresados
    data = {'Texto': textos.split(','), 'Sentimiento': etiquetas}
    df = pd.DataFrame(data)

    # Realizar el análisis de sentimientos y contar las palabras
    sentimiento_counts = df['Sentimiento'].value_counts()
    palabras_counter = Counter(" ".join(df['Texto']).split())

    # Convertir los datos numéricos a enteros para que puedan ser serializados a JSON
    resultado_json = {
        'positivo': int(sentimiento_counts.get('bueno', 0)),
        'negativo': int(sentimiento_counts.get('malo', 0)),
        'palabras_contadas': counter_to_dict(palabras_counter.most_common(5))
        #'lista_textos': lista_textos_etiquetas
    }

    # Convertir el JSON a formato de cadena y devolverlo como respuesta
    return jsonify(resultado_json)

if __name__ == '__main__':
    app.run(debug=True)
