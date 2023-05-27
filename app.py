import json
from flask import Flask, request, jsonify
from keras.models import load_model
from extract import class_prediction, get_response


model = load_model('model.h5')
intents = json.loads(open('intents.json', encoding='utf-8').read())


app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return "Bem vindo a API Nine Nine"


@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    message = data['message']

    ints = class_prediction(message, model)
    res = get_response(ints, intents)

    return jsonify({'response': res})

if __name__ == '__main__':
    app.run(debug=True)
