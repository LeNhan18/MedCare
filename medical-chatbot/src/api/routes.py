from flask import Blueprint, request, jsonify
from src.models.symptom_classifier import SymptomClassifier
from src.models.drug_recommender import DrugRecommender

api = Blueprint('api', __name__)

symptom_classifier = SymptomClassifier()
drug_recommender = DrugRecommender()

@api.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms', [])
    
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400
    
    classified_symptoms = symptom_classifier.classify(symptoms)
    medications = drug_recommender.recommend(classified_symptoms)
    
    return jsonify({
        'classified_symptoms': classified_symptoms,
        'medications': medications
    })