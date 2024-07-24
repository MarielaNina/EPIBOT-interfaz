from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import spacy
import unidecode  # Para quitar las tildes
from keras.models import load_model

app = Flask(__name__)

# Cargar el modelo de spaCy para español
nlp = spacy.load('es_core_news_sm')

# Cargar los datos de intenciones y el modelo entrenado
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Lista de stopwords personalizada
stopwords = ["y", "de", "pero", "las", "la", "le", "los", "un", "una", "unos", "unas", "a", "en", "que", "con", "por", "para"]

# Pasamos las palabras de la oración a su forma raíz
def clean_up_sentence(sentence):
    doc = nlp(sentence)
    # Eliminar tildes y convertir a minúsculas
    sentence_words = [unidecode.unidecode(token.lemma_.lower()) for token in doc if token.text.lower() not in stopwords]
    return sentence_words

# Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

# Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.argmax(res)
    category = classes[max_index]
    return category

# Obtenemos una respuesta aleatoria
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chat():
    user_input = request.form['msg']
    category = predict_class(user_input)
    response = get_response(category, intents)
    return jsonify({'response': response})

if __name__ == '__main__':
        app.run(debug=True, port=5001) 
