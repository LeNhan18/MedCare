"""
Test NER Model với câu tiếng Việt
"""

import json
import joblib
from sklearn_crfsuite import CRF

def prepare_features(tokens):
    """Prepare features for a single sentence"""
    features = []
    for i, token in enumerate(tokens):
        feature = {
            'word.lower()': token.lower(),
            'word.isupper()': token.isupper(),
            'word.istitle()': token.istitle(),
            'word.isdigit()': token.isdigit(),
            'BOS': i == 0,  # Beginning of sentence
            'EOS': i == len(tokens) - 1  # End of sentence
        }
        features.append(feature)
    return features

def predict_ner(model, text):
    """Predict NER tags for a Vietnamese text"""
    # Simple tokenization (split by spaces)
    tokens = text.split()
    features = prepare_features(tokens)
    
    # Predict labels
    predicted_labels = model.predict([features])[0]
    
    # Format output
    result = []
    for token, label in zip(tokens, predicted_labels):
        result.append((token, label))
    
    return result

def extract_entities(predictions):
    """Extract entities from BIO predictions"""
    entities = []
    current_entity = None
    current_text = []
    
    for token, label in predictions:
        if label.startswith('B-'):
            # Start of new entity
            if current_entity:
                entities.append((current_entity, ' '.join(current_text)))
            current_entity = label[2:]  # Remove 'B-'
            current_text = [token]
        elif label.startswith('I-') and current_entity:
            # Continuation of entity
            current_text.append(token)
        else:
            # End of entity or no entity
            if current_entity:
                entities.append((current_entity, ' '.join(current_text)))
                current_entity = None
                current_text = []
    
    # Don't forget the last entity
    if current_entity:
        entities.append((current_entity, ' '.join(current_text)))
    
    return entities

def main():
    # Load model
    print("Đang tải model...")
    model_data = joblib.load('z:/MedCare/medical-chatbot/data/models/simple_ner_model.joblib')
    model = model_data['model']
    
    # Test sentences
    test_sentences = [
        "Tôi bị đau đầu và sốt cao từ 3 ngày nay",
        "Con tôi 5 tuổi bị ho và chảy nước mũi",
        "Bà ngoại 70 tuổi bị đau khớp ở chân",
        "Uống paracetamol để giảm đau và hạ sốt",
        "Bệnh nhân bị viêm phổi cần dùng kháng sinh",
        "Đau bụng dưới kèm theo buồn nôn"
    ]
    
    print("Testing NER model với câu tiếng Việt:\n")
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"Câu {i}: {sentence}")
        
        # Predict
        predictions = predict_ner(model, sentence)
        print("Predictions:", predictions)
        
        # Extract entities
        entities = extract_entities(predictions)
        print("Entities found:")
        for entity_type, entity_text in entities:
            print(f"  {entity_type}: {entity_text}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()