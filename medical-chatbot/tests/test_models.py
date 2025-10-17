import pytest
from src.models.symptom_classifier import SymptomClassifier
from src.models.drug_recommender import DrugRecommender
from src.models.chatbot_model import ChatbotModel

def test_symptom_classifier():
    classifier = SymptomClassifier()
    test_input = "I have a headache and fever."
    predicted_symptom = classifier.predict(test_input)
    assert predicted_symptom in ["headache", "fever", "flu", "cold"]  # Example symptoms

def test_drug_recommender():
    recommender = DrugRecommender()
    test_symptom = "headache"
    recommended_drugs = recommender.recommend(test_symptom)
    assert isinstance(recommended_drugs, list)
    assert len(recommended_drugs) > 0  # Ensure some drugs are recommended

def test_chatbot_model():
    chatbot = ChatbotModel()
    test_input = "I feel sick."
    response = chatbot.get_response(test_input)
    assert isinstance(response, str)
    assert len(response) > 0  # Ensure a response is generated