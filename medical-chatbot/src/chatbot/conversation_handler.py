from flask import request, jsonify
from src.models.symptom_classifier import SymptomClassifier
from src.models.drug_recommender import DrugRecommender

class ConversationHandler:
    def __init__(self):
        self.classifier = SymptomClassifier()
        self.recommender = DrugRecommender()

    def handle_message(self, message):
        symptoms = self.classifier.classify_symptoms(message)
        medications = self.recommender.recommend_drugs(symptoms)
        response = {
            "symptoms": symptoms,
            "medications": medications
        }
        return response

    def process_request(self):
        data = request.get_json()
        user_message = data.get("message", "")
        response = self.handle_message(user_message)
        return jsonify(response)