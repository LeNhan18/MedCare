from models.symptom_classifier import SymptomClassifier
from models.drug_recommender import DrugRecommender

class ResponseGenerator:
    def __init__(self):
        self.symptom_classifier = SymptomClassifier()
        self.drug_recommender = DrugRecommender()

    def generate_response(self, user_input):
        symptoms = self.symptom_classifier.classify_symptoms(user_input)
        if not symptoms:
            return "I'm sorry, I couldn't identify any symptoms from your input."

        medications = self.drug_recommender.recommend_drugs(symptoms)
        if not medications:
            return "I couldn't find any medications for the identified symptoms."

        response = f"Based on your symptoms: {', '.join(symptoms)}, I recommend the following medications: {', '.join(medications)}."
        return response