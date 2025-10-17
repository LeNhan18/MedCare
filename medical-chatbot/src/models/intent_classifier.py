#!/usr/bin/env python3
"""
Medical Chatbot - Intent Classification Model
Phân loại ý định người dùng từ câu hỏi tiếng Việt

Intent Categories:
- symptom_inquiry: Hỏi về triệu chứng ("Tôi bị đau đầu")
- drug_question: Hỏi về thuốc ("Paracetamol có tác dụng gì?") 
- emergency: Tình huống khẩn cấp ("Tôi bị đau ngực dữ dội")
- dosage_question: Hỏi về liều lượng ("Uống bao nhiêu viên?")
- side_effects: Hỏi về tác dụng phụ ("Thuốc này có tác dụng phụ không?")
- general_health: Câu hỏi sức khỏe tổng quát ("Làm sao để khỏe mạnh?")
- greeting: Chào hỏi ("Xin chào", "Hello")
- unknown: Không xác định được
"""

import pandas as pd
import numpy as np
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import json
from datetime import datetime
import os

class MedicalIntentClassifier:
    def __init__(self, model_path=None):
        """
        Initialize Intent Classifier for Vietnamese medical queries
        
        Args:
            model_path (str): Path to saved model file
        """
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.intent_labels = [
            'symptom_inquiry',
            'drug_question', 
            'emergency',
            'dosage_question',
            'side_effects',
            'general_health',
            'greeting',
            'unknown'
        ]
        
        # Emergency keywords for high-priority detection
        self.emergency_keywords = [
            'cấp cứu', 'emergency', 'nguy hiểm', 'nguy kịch',
            'đau ngực dữ dội', 'khó thở', 'mất ý thức', 
            'xuất huyết', 'chảy máu nhiều', 'sốt cao',
            'co giật', 'đột quỵ', 'tim đập nhanh',
            'hôn mê', 'ngộ độc', 'dị ứng nặng'
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
    def build_model(self):
        """Xây dựng model BERT cho phân loại intent"""
        # Load pre-trained BERT
        if self.model_type == "phobert":
            model_name = "vinai/phobert-base"
        elif self.model_type == "distilbert":
            model_name = "distilbert-base-multilingual-cased"
        else:
            model_name = "bert-base-multilingual-cased"
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = TFAutoModel.from_pretrained(model_name)
        
        # Input layers
        input_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name="attention_mask")
        
        # BERT embeddings
        bert_output = bert_model(input_ids, attention_mask=attention_mask)
        
        # Classification head
        pooled_output = bert_output.pooler_output
        dropout = tf.keras.layers.Dropout(0.3)(pooled_output)
        output = tf.keras.layers.Dense(
            len(self.classes), 
            activation='softmax', 
            name='intent_output'
        )(dropout)
        
        self.model = tf.keras.Model(
            inputs=[input_ids, attention_mask], 
            outputs=output
        )
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def preprocess_text(self, text):
        """Tiền xử lý văn bản đầu vào"""
        if isinstance(text, list):
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='tf'
            )
        else:
            encoding = self.tokenizer(
                [text],
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='tf'
            )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
    
    def predict(self, text):
        """Dự đoán intent từ văn bản"""
        if self.model is None:
            raise ValueError("Model chưa được load. Hãy gọi load_model() trước.")
        
        # Preprocess
        inputs = self.preprocess_text(text)
        
        # Predict
        predictions = self.model.predict([inputs['input_ids'], inputs['attention_mask']])
        
        # Get results
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_intent = self.classes[predicted_class_idx]
        
        return {
            'intent': predicted_intent,
            'confidence': confidence,
            'all_probabilities': {
                self.classes[i]: float(predictions[0][i]) 
                for i in range(len(self.classes))
            },
            'is_confident': confidence >= self.confidence_threshold
        }
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, epochs=5):
        """Huấn luyện model"""
        if self.model is None:
            self.build_model()
        
        # Preprocess training data
        train_inputs = self.preprocess_text(train_texts)
        
        if val_texts is not None and val_labels is not None:
            val_inputs = self.preprocess_text(val_texts)
            validation_data = ([val_inputs['input_ids'], val_inputs['attention_mask']], val_labels)
        else:
            validation_data = None
        
        # Train
        history = self.model.fit(
            [train_inputs['input_ids'], train_inputs['attention_mask']],
            train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=16,
            verbose=1
        )
        
        return history
    
    def save_model(self, path=None):
        """Lưu model"""
        if path is None:
            path = self.model_config['path']
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
        # Save tokenizer
        tokenizer_path = path.replace('.h5', '_tokenizer')
        self.tokenizer.save_pretrained(tokenizer_path)
    
    def load_model(self, path=None):
        """Load model đã huấn luyện"""
        if path is None:
            path = self.model_config['path']
        
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path)
            
            # Load tokenizer
            tokenizer_path = path.replace('.h5', '_tokenizer')
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                # Fallback to default tokenizer
                if self.model_type == "phobert":
                    model_name = "vinai/phobert-base"
                elif self.model_type == "distilbert":
                    model_name = "distilbert-base-multilingual-cased"
                else:
                    model_name = "bert-base-multilingual-cased"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            print(f"Model file không tồn tại: {path}")
            self.build_model()


# Example usage và test
if __name__ == "__main__":
    # Test examples
    test_examples = [
        "Tôi bị đau đầu",
        "Thuốc Panadol có tác dụng gì?", 
        "Tôi có thể uống paracetamol khi bị cảm không?",
        "Bệnh tiểu đường có nguy hiểm không?",
        "Xin chào bạn"
    ]
    
    expected_intents = [
        "triệu_chứng",
        "tra_cứu_thuốc", 
        "tư_vấn_sử_dụng",
        "thông_tin_bệnh",
        "khác"
    ]
    
    # Initialize classifier
    classifier = IntentClassifier()
    classifier.build_model()
    
    print("Model Intent Classifier đã được khởi tạo!")
    print(f"Classes: {classifier.classes}")
    print(f"Model summary:")
    classifier.model.summary()