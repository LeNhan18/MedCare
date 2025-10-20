"""
Train NER Model with Enhanced Dataset
"""

import json
import os
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
import joblib

def load_data(file_path):
    """Load training data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def prepare_features_and_labels(data):
    """Prepare features and labels for CRF training"""
    features = []
    labels = []
    
    for sample in data:
        tokens = sample['tokens']
        labels.append(sample['labels'])
        
        sample_features = []
        for i, token in enumerate(tokens):
            feature = {
                'word.lower()': token.lower(),
                'word.isupper()': token.isupper(),
                'word.istitle()': token.istitle(),
                'word.isdigit()': token.isdigit(),
                'BOS': i == 0,  # Beginning of sentence
                'EOS': i == len(tokens) - 1  # End of sentence
            }
            sample_features.append(feature)
        
        features.append(sample_features)
    
    return features, labels

def train_crf(X_train, y_train):
    """Train CRF model"""
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    return crf

def evaluate_model(crf, X_test, y_test):
    """Evaluate CRF model"""
    y_pred = crf.predict(X_test)
    report = flat_classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)

def main():
    # Paths
    enhanced_data_path = 'z:/MedCare/medical-chatbot/data/processed/ner_training_data_enhanced.json'
    model_output_path = 'z:/MedCare/medical-chatbot/data/models/simple_ner_model.joblib'
    
    # Load data
    print("Loading enhanced training data...")
    data = load_data(enhanced_data_path)
    print(f"Loaded {len(data)} samples")
    
    # Prepare features and labels
    print("Preparing features and labels...")
    X, y = prepare_features_and_labels(data)
    
    # Split data
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train model
    print("Training CRF model...")
    crf = train_crf(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(crf, X_test, y_test)
    
    # Save model
    print("Saving model...")
    joblib.dump({'model': crf}, model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    main()