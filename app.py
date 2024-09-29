from flask import Flask, render_template, request
import joblib
import json
import random
import os

app = Flask(__name__)

# Load the trained model and vectorizer
model_path = os.path.join('model', 'chatbot_model.pkl')
vectorizer_path = os.path.join('model', 'vectorizer.pkl')
intents_path = os.path.join('dataset', 'intents1.json')

try:
    best_model = joblib.load(model_path)
except FileNotFoundError:
    print(f"Error: File not found '{model_path}'")

try:
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError:
    print(f"Error: File not found '{vectorizer_path}'")

try:
    with open(intents_path, 'r') as f:
        intents = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found '{intents_path}'")

def chatbot_response(user_input):
    input_text = vectorizer.transform([user_input])
    predicted_intent = best_model.predict(input_text)[0]

    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            break

    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = chatbot_response(user_input)
    return response

if __name__ == '__main__':
    app.run(debug=True)
