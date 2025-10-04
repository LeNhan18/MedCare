from symptom_classifier import SymptomClassifier
from drug_recommender import DrugRecommender

class ChatbotModel:
    def __init__(self):
        self.symptom_classifier = SymptomClassifier()
        self.drug_recommender = DrugRecommender()

    def get_medication_suggestion(self, user_input):
        symptoms = self.symptom_classifier.classify_symptoms(user_input)
        medications = self.drug_recommender.recommend_drugs(symptoms)
        return medications

    def respond_to_query(self, user_input):
        medications = self.get_medication_suggestion(user_input)
        if medications:
            response = f"Based on your symptoms, I suggest the following medications: {', '.join(medications)}."
        else:
            response = "I'm sorry, I couldn't find any medication suggestions for your symptoms."
        return response