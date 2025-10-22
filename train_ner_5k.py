#!/usr/bin/env python3
"""
Train và test NER model với 5K dataset
"""

import pandas as pd
import json
import joblib
import os
from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

def load_ner_data(csv_path):
    """Load NER data từ CSV 5K"""
    print(f"📄 Loading NER data from: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    print(f"✅ Loaded {len(df)} samples")
    
    sentences = []
    labels = []
    
    for _, row in df.iterrows():
        text = row['text'].strip('"')
        entities_str = row['entities'].strip('"')
        
        try:
            entities = json.loads(entities_str)
        except:
            continue
            
        # Tokenize (simple split by spaces)
        tokens = text.split()
        
        # Initialize BIO tags
        bio_tags = ['O'] * len(tokens)
        
        # Convert entity annotations to BIO format
        for entity in entities:
            entity_text = entity['text']
            entity_label = entity['label']
            start_pos = entity['start']
            end_pos = entity['end']
            
            # Find entity in tokens
            entity_tokens = entity_text.split()
            
            # Find position in token list
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if ' '.join(tokens[i:i+len(entity_tokens)]) == entity_text:
                    # Mark BIO tags
                    bio_tags[i] = f'B-{entity_label}'
                    for j in range(1, len(entity_tokens)):
                        if i + j < len(bio_tags):
                            bio_tags[i + j] = f'I-{entity_label}'
                    break
        
        sentences.append(tokens)
        labels.append(bio_tags)
    
    return sentences, labels

def prepare_features(sentence):
    """Prepare features for CRF"""
    features = []
    for i, word in enumerate(sentence):
        feature = {
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'word.isalpha()': word.isalpha(),
            'word.length': len(word),
            'BOS': i == 0,
            'EOS': i == len(sentence) - 1,
        }
        
        # Add prefix/suffix features
        if len(word) >= 2:
            feature['prefix-2'] = word[:2]
            feature['suffix-2'] = word[-2:]
        if len(word) >= 3:
            feature['prefix-3'] = word[:3]
            feature['suffix-3'] = word[-3:]
            
        # Previous and next word features
        if i > 0:
            feature['prev_word'] = sentence[i-1].lower()
        if i < len(sentence) - 1:
            feature['next_word'] = sentence[i+1].lower()
            
        features.append(feature)
    
    return features

def prepare_dataset(sentences, labels):
    """Prepare features and labels for all sentences"""
    X = []
    y = []
    
    for sentence, sentence_labels in zip(sentences, labels):
        if len(sentence) == len(sentence_labels):
            features = prepare_features(sentence)
            X.append(features)
            y.append(sentence_labels)
    
    return X, y

def train_ner_model(X_train, y_train):
    """Train CRF NER model"""
    print("🔥 Training NER model...")
    
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    crf.fit(X_train, y_train)
    print("✅ Training completed!")
    
    return crf

def evaluate_model(model, X_test, y_test):
    """Evaluate NER model"""
    print("📊 Evaluating model...")
    
    y_pred = model.predict(X_test)
    
    # Flatten for sklearn metrics
    y_true_flat = []
    y_pred_flat = []
    
    for true_labels, pred_labels in zip(y_test, y_pred):
        y_true_flat.extend(true_labels)
        y_pred_flat.extend(pred_labels)
    
    # Get unique labels
    labels = sorted(set(y_true_flat + y_pred_flat))
    labels = [l for l in labels if l != 'O']  # Remove 'O' for entity-focused metrics
    
    print("\n📋 Classification Report:")
    print(classification_report(y_true_flat, y_pred_flat, labels=labels, zero_division=0))
    
    # Calculate entity-level accuracy
    correct_entities = 0
    total_entities = 0
    
    for true_labels, pred_labels in zip(y_test, y_pred):
        true_entities = extract_entities_from_bio(true_labels)
        pred_entities = extract_entities_from_bio(pred_labels)
        
        total_entities += len(true_entities)
        for entity in true_entities:
            if entity in pred_entities:
                correct_entities += 1
    
    if total_entities > 0:
        entity_accuracy = correct_entities / total_entities
        print(f"\n✅ Entity-level accuracy: {entity_accuracy:.3f}")
    
    return y_pred

def extract_entities_from_bio(bio_tags):
    """Extract entities from BIO tags"""
    entities = []
    current_entity = None
    start_idx = 0
    
    for i, tag in enumerate(bio_tags):
        if tag.startswith('B-'):
            if current_entity:
                entities.append((current_entity, start_idx, i-1))
            current_entity = tag[2:]
            start_idx = i
        elif tag.startswith('I-'):
            continue
        else:  # 'O' tag
            if current_entity:
                entities.append((current_entity, start_idx, i-1))
                current_entity = None
    
    if current_entity:
        entities.append((current_entity, start_idx, len(bio_tags)-1))
    
    return entities

def predict_ner(model, text):
    """Predict NER for new text"""
    tokens = text.split()
    features = prepare_features(tokens)
    predicted_labels = model.predict([features])[0]
    
    result = []
    for token, label in zip(tokens, predicted_labels):
        result.append((token, label))
    
    return result

def test_model(model):
    """Test model với sample sentences"""
    test_sentences = [
        "Tôi 25 tuổi bị đau đầu và sốt cao",
        "Con tôi 5 tuổi bị ho và chảy nước mũi",
        "Uống paracetamol 500mg để giảm đau",
        "Bệnh nhân 45 tuổi bị viêm phổi cần dùng kháng sinh",
        "Cô ấy bị đau lưng mãn tính từ 2 tuần nay",
        "Dùng insulin 10 đơn vị trước bữa ăn"
    ]
    
    print("\n🧪 Testing NER model:")
    print("=" * 60)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. Text: {sentence}")
        
        predictions = predict_ner(model, sentence)
        
        # Extract entities
        entities = []
        current_entity = None
        current_text = []
        
        for token, label in predictions:
            if label.startswith('B-'):
                if current_entity:
                    entities.append((current_entity, ' '.join(current_text)))
                current_entity = label[2:]
                current_text = [token]
            elif label.startswith('I-') and current_entity:
                current_text.append(token)
            else:
                if current_entity:
                    entities.append((current_entity, ' '.join(current_text)))
                    current_entity = None
                    current_text = []
        
        if current_entity:
            entities.append((current_entity, ' '.join(current_text)))
        
        print("   Entities:")
        for entity_type, entity_text in entities:
            print(f"     • {entity_type}: {entity_text}")

def main():
    # Paths
    data_5k_path = 'e:/MedCare/data/ner_training_dataset_5k.csv'
    data_original_path = 'e:/MedCare/data/ner_training_dataset.csv'
    model_path = 'e:/MedCare/medical-chatbot/data/models/ner_model_5k.joblib'
    
    # Try 5K dataset first
    if os.path.exists(data_5k_path):
        print("🔥 Using 5K NER dataset")
        sentences, labels = load_ner_data(data_5k_path)
    elif os.path.exists(data_original_path):
        print("📄 Fallback to original dataset")
        sentences, labels = load_ner_data(data_original_path)
    else:
        print("❌ No dataset found!")
        return
    
    print(f"📊 Total samples: {len(sentences)}")
    
    # Prepare features
    print("🔧 Preparing features...")
    X, y = prepare_dataset(sentences, labels)
    print(f"✅ Prepared {len(X)} samples with features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    model = train_ner_model(X_train, y_train)
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    # Save model
    print(f"💾 Saving model to: {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        'model': model,
        'type': 'CRF_NER',
        'training_samples': len(sentences)
    }, model_path)
    
    # Test with examples
    test_model(model)
    
    print(f"\n🎉 NER model ready! Trained on {len(sentences)} samples")

if __name__ == "__main__":
    main()