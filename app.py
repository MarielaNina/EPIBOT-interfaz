from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import spacy
import unidecode
from tensorflow.keras.models import load_model
import re
import logging

app = Flask(__name__)

# Configuración del logging
# Cargar recursos una sola vez al inicio
try:
    nlp = spacy.load('es_core_news_sm')
    with open('intents.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.h5', compile=False)
except Exception as e:
    logging.error(f"Error al cargar recursos: {e}")
    # Maneja el error apropiadamente, tal vez configurando una bandera para verificar antes de usar estos recursos

stopwords = ["y", "de", "pero", "las", "la", "le", "los", "un", "una", "unos", "unas", "a", "en", "que", "con", "por", "para"]

def is_meaningful(sentence):
    doc = nlp(sentence)
    tokens = [token for token in doc if token.is_alpha and token.text.lower() not in stopwords]
    return len(tokens) > 0

def clean_up_sentence(sentence):
    doc = nlp(sentence)
    sentence_words = [unidecode.unidecode(token.lemma_.lower()) for token in doc if token.text.lower() not in stopwords]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence, threshold=0.3):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.argmax(res)
    confidence = res[max_index]
    if confidence >= threshold:
        return classes[max_index], confidence
    else:
        return None, confidence

def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i["tag"] == tag:
            return random.choice(i['responses'])
    return "Con gusto te ayudaría, pero necesito un poco más de información para entender tu consulta. ¿Puedes ser más específico, por favor?"

def clean_input(user_input):
    cleaned_input = re.sub(r'[^\w\s]', '', user_input)
    return cleaned_input

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chat():
    user_input = request.form['msg']
    user_input = clean_input(user_input)
    if not is_meaningful(user_input):
        response = "Con gusto te ayudaría, pero necesito un poco más de información para entender tu consulta. ¿Puedes ser más específico, por favor?"
    else:
        try:
            category, confidence = predict_class(user_input)
            if category is None:
                response = "Con gusto te ayudaría, pero necesito un poco más de información para entender tu consulta. ¿Puedes ser más específico, por favor?"
            else:
                response = get_response(category, intents)
        except Exception as e:
            logging.error(f"Error al procesar la entrada: {e}")
            response = "Lo siento, ha ocurrido un error al procesar tu solicitud. Por favor, intenta de nuevo más tarde."
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True,port=5001)  # Cambiado a False para producción






