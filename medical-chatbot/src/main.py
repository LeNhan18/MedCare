import os
from flask import Flask, request, jsonify
from models.symptom_classifier import SymptomClassifier
from models.drug_recommender import DrugRecommender
from chatbot.conversation_handler import ConversationHandler

app = Flask(__name__)

# Initialize models
symptom_classifier = SymptomClassifier()
drug_recommender = DrugRecommender()
conversation_handler = ConversationHandler(symptom_classifier, drug_recommender)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    response = conversation_handler.handle_conversation(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)