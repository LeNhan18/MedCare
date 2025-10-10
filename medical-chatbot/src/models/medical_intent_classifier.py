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
import json
import random
import os
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
    
    def create_training_data(self):
        """
        Tạo dữ liệu training từ dataset y tế thực tế (CSV format)
        """
        training_data = []
        
        try:
            # Đọc dữ liệu từ file CSV với encoding tự động detect
            try:
                df = pd.read_csv('data/medical_dataset_training.csv', encoding='utf-8')
            except UnicodeDecodeError:
                print("UTF-8 failed, trying latin-1...")
                df = pd.read_csv('data/medical_dataset_training.csv', encoding='latin-1')
            
            print(f"Đã tải {len(df)} bản ghi từ dataset y tế (CSV)")
            
            # Tạo training examples từ dữ liệu thực
            for _, row in df.iterrows():
                drug_name = row.get('drug_name', '')
                condition_vi = row.get('medical_condition_vi', '')  # CSV uses medical_condition_vi
                condition_en = row.get('medical_condition', '')     # CSV uses medical_condition
                side_effects_vi = row.get('side_effects_vi', '')
                description_vi = row.get('medical_condition_description_vi', '')
                
                # Tạo các câu hỏi về triệu chứng từ condition_vi (nhiều hơn để đảm bảo đủ data)
                if condition_vi and str(condition_vi) not in ['', 'nan', 'NaN', 'None']:
                    # Các template cho symptom_inquiry
                    symptom_templates = [
                        f"Tôi bị {condition_vi.lower()}",
                        f"Có thuốc nào chữa {condition_vi.lower()} không",
                        f"Triệu chứng {condition_vi.lower()} là gì",
                        f"Làm sao để điều trị {condition_vi.lower()}",
                        f"Tôi có dấu hiệu {condition_vi.lower()}"
                    ]
                    
                    # Tạo nhiều samples hơn - lấy tất cả templates với xác suất 30%
                    if random.random() < 0.3:  # 30% chance to create samples
                        for template in symptom_templates:
                            training_data.append((template, 'symptom_inquiry'))
                
                # Tạo các câu hỏi về thuốc từ drug_name
                if drug_name and str(drug_name) not in ['', 'nan', 'NaN', 'None']:
                    drug_templates = [
                        f"Thuốc {drug_name} có tác dụng gì",
                        f"Cách sử dụng thuốc {drug_name}",
                        f"Liều lượng thuốc {drug_name} như thế nào",
                        f"Thuốc {drug_name} dùng để làm gì",
                        f"Có nên uống thuốc {drug_name} không"
                    ]
                    
                    # Tạo nhiều samples với xác suất 40%
                    if random.random() < 0.4:  # 40% chance
                        selected_drug_templates = random.sample(drug_templates, min(3, len(drug_templates)))
                        for template in selected_drug_templates:
                            training_data.append((template, 'drug_question'))
                
                # Tạo câu hỏi về tác dụng phụ
                if side_effects_vi and str(side_effects_vi) not in ['', 'nan', 'NaN', 'None'] and len(str(side_effects_vi)) > 20:
                    side_effect_templates = [
                        f"Thuốc {drug_name} có tác dụng phụ gì",
                        f"Tác dụng phụ của thuốc {drug_name}",
                        f"Uống {drug_name} có hại gì không",
                        f"Thuốc {drug_name} có an toàn không"
                    ]
                    
                    # Tăng xác suất tạo side effects samples
                    if random.random() < 0.4:  # 40% chance
                        selected_side_templates = random.sample(side_effect_templates, min(2, len(side_effect_templates)))
                        for template in selected_side_templates:
                            training_data.append((template, 'side_effects'))
                        
                # Tạo câu hỏi về liều lượng
                if drug_name and str(drug_name) not in ['', 'nan', 'NaN', 'None']:
                    dosage_templates = [
                        f"Liều lượng thuốc {drug_name}",
                        f"Uống {drug_name} bao nhiêu viên một lần",
                        f"Cách dùng thuốc {drug_name} đúng cách",
                        f"Một ngày uống {drug_name} mấy lần"
                    ]
                    
                    # Tăng xác suất tạo dosage samples
                    if random.random() < 0.4:  # 40% chance
                        selected_dosage = random.choice(dosage_templates)
                        training_data.append((selected_dosage, 'dosage_question'))
            
            # Thêm các ví dụ cứng cho emergency, general_health, greeting
            emergency_examples = [
                ("Cấp cứu! Tôi bị đau ngực dữ dội", 'emergency'),
                ("Khẩn cấp: bé bị sốt cao 40 độ", 'emergency'), 
                ("Gọi bác sĩ ngay! Tôi không thở được", 'emergency'),
                ("Cần cấp cứu: bị ngộ độc thực phẩm", 'emergency'),
                ("Khẩn cấp! Bị chảy máu không cầm được", 'emergency'),
                ("Cấp cứu: có người bị ngất xỉu", 'emergency'),
                ("Gọi 115! Có tai nạn xe máy", 'emergency'),
                ("Khẩn cấp: bé nuốt phải thuốc", 'emergency'),
                ("Cần bác sĩ gấp: đau bụng dữ dội", 'emergency'),
                ("Cấp cứu! Bị dị ứng thuốc nghiêm trọng", 'emergency')
            ]
            
            health_examples = [
                ("Làm sao để tăng sức đề kháng", 'general_health'),
                ("Chế độ ăn uống lành mạnh", 'general_health'),
                ("Tập thể dục như thế nào để khỏe", 'general_health'),
                ("Cách phòng ngừa bệnh tật", 'general_health'),
                ("Giữ gìn sức khỏe ở tuổi cao", 'general_health'),
                ("Lời khuyên sống khỏe mạnh", 'general_health'),
                ("Cách giảm stress hiệu quả", 'general_health'), 
                ("Ngủ đủ giấc quan trọng ra sao", 'general_health'),
                ("Tác hại của thuốc lá đến sức khỏe", 'general_health'),
                ("Cách tăng cường miễn dịch tự nhiên", 'general_health')
            ]
            
            greeting_examples = [
                ("Xin chào", 'greeting'),
                ("Chào bác sĩ", 'greeting'), 
                ("Hi", 'greeting'),
                ("Hello", 'greeting'),
                ("Chào em", 'greeting'),
                ("Tôi cần tư vấn", 'greeting'),
                ("Cho tôi hỏi", 'greeting'),
                ("Bạn có thể giúp tôi không", 'greeting'),
                ("Tôi có thể hỏi gì đó được không", 'greeting')
            ]
            
            unknown_examples = [
                ("xyz abc", 'unknown'),
                ("12345", 'unknown'), 
                ("không hiểu", 'unknown'),
                ("???", 'unknown'),
                ("haha hihi", 'unknown'),
                ("test test", 'unknown'),
                ("random text", 'unknown'),
                ("asdfgh", 'unknown'),
                ("........", 'unknown'),
                ("", 'unknown')
            ]
            
            # Thêm các ví dụ cứng
            training_data.extend(emergency_examples)
            training_data.extend(health_examples)
            training_data.extend(greeting_examples)
            training_data.extend(unknown_examples)
            
            print(f"Đã tạo {len(training_data)} mẫu training từ dataset thực tế")
            return training_data
            
        except FileNotFoundError:
            print("Không tìm thấy file medical_dataset_training.csv, sử dụng dữ liệu mặc định")
            # Fallback to default data if file not found
            return self._create_fallback_training_data()
        except Exception as e:
            print(f"Lỗi khi đọc dataset CSV: {e}")
            return self._create_fallback_training_data()
    
    def _create_fallback_training_data(self):
        """
        Dữ liệu dự phòng nếu không đọc được file JSON
        """
        fallback_data = [
            ("Tôi bị đau đầu", 'symptom_inquiry'),
            ("Thuốc paracetamol có tác dụng gì", 'drug_question'),
            ("Cấp cứu! Đau ngực dữ dội", 'emergency'),
            ("Liều lượng aspirin", 'dosage_question'),
            ("Thuốc này có tác dụng phụ gì", 'side_effects'),
            ("Cách giữ gìn sức khỏe", 'general_health'),
            ("Xin chào bác sĩ", 'greeting'),
            ("???", 'unknown')
        ]
        return fallback_data
    
    def preprocess_text(self, text):
        """
        Tiền xử lý văn bản tiếng Việt
        
        Args:
            text (str): Raw text input
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove special characters nhưng giữ lại dấu câu quan trọng
        text = re.sub(r'[^\w\s\.\?\!]', ' ', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def detect_emergency(self, text):
        """
        Detect emergency situations với high priority
        
        Args:
            text (str): User input text
            
        Returns:
            bool: True nếu phát hiện tình huống khẩn cấp
        """
        text_lower = text.lower()
        
        for keyword in self.emergency_keywords:
            if keyword in text_lower:
                return True
                
        # Additional logic for emergency detection
        emergency_patterns = [
            r'sốt.*4[0-9].*độ',  # Sốt >= 40 độ
            r'đau.*ngực.*dữ.*dội',
            r'không.*thể.*thở',
            r'mất.*ý.*thức',
            r'chảy.*máu.*nhiều'
        ]
        
        for pattern in emergency_patterns:
            if re.search(pattern, text_lower):
                return True
                
        return False
    
    def train(self, save_path=None):
        """
        Train Intent Classification model
        
        Args:
            save_path (str): Path to save trained model
        """
        print("🤖 Training Medical Intent Classifier...")
        
        # Create training data
        training_data = self.create_training_data()
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data, columns=['text', 'intent'])
        
        print(f"📊 Training data: {len(df)} samples")
        print(f"📋 Intent distribution:")
        print(df['intent'].value_counts())
        
        # Preprocess texts
        df['text_clean'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        X = df['text_clean']
        y = df['intent']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline với TF-IDF + Logistic Regression
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=None  # Không dùng stop words cho tiếng Việt
            )),
            ('classifier', LogisticRegression(
                random_state=42,
                max_iter=1000
            ))
        ])
        
        # Train model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.pipeline.score(X_train, y_train)
        test_score = self.pipeline.score(X_test, y_test)
        
        print(f"✅ Training accuracy: {train_score:.3f}")
        print(f"✅ Testing accuracy: {test_score:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X, y, cv=5)
        print(f"✅ Cross-validation accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Detailed classification report
        y_pred = self.pipeline.predict(X_test)
        print("\n Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model nếu có path
        if save_path:
            self.save_model(save_path)
            print(f" Model saved to: {save_path}")
            
        return test_score  # Return single accuracy score for compatibility
    
    def predict(self, text, return_confidence=False):
        """
        Predict intent từ user input
        
        Args:
            text (str): User input text
            return_confidence (bool): Return confidence scores
            
        Returns:
            str or tuple: Intent label hoặc (intent, confidence_dict)
        """
        if not self.pipeline:
            raise ValueError("Model chưa được train! Gọi train() trước.")
            
        # Emergency detection first
        if self.detect_emergency(text):
            if return_confidence:
                return 'emergency', {'emergency': 1.0}
            return 'emergency'
            
        # Preprocess input
        text_clean = self.preprocess_text(text)
        
        if not text_clean:
            if return_confidence:
                return 'unknown', {'unknown': 1.0}
            return 'unknown'
            
        # Predict intent
        intent = self.pipeline.predict([text_clean])[0]
        
        if return_confidence:
            # Get confidence scores
            probabilities = self.pipeline.predict_proba([text_clean])[0]
            confidence_dict = dict(zip(self.pipeline.classes_, probabilities))
            return intent, confidence_dict
            
        return intent
    
    def predict_intent(self, text):
        """
        Alias cho predict() để tương thích
        """
        return self.predict(text)
        
    def get_confidence(self, text):
        """
        Get confidence score cho predicted intent
        """
        _, confidence_dict = self.predict(text, return_confidence=True)
        return max(confidence_dict.values())
    
    def batch_predict(self, texts):
        """
        Predict intents cho nhiều texts
        
        Args:
            texts (list): List of input texts
            
        Returns:
            list: List of predicted intents
        """
        results = []
        
        for text in texts:
            intent = self.predict(text)
            results.append(intent)
            
        return results
    
    def save_model(self, file_path):
        """
        Save trained model
        
        Args:
            file_path (str): Path to save model
        """
        model_data = {
            'pipeline': self.pipeline,
            'intent_labels': self.intent_labels,
            'emergency_keywords': self.emergency_keywords,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        joblib.dump(model_data, file_path)
    
    def load_model(self, file_path):
        """
        Load saved model
        
        Args:
            file_path (str): Path to saved model
        """
        model_data = joblib.load(file_path)
        
        self.pipeline = model_data['pipeline']
        self.intent_labels = model_data['intent_labels'] 
        self.emergency_keywords = model_data['emergency_keywords']
        
        print(f"✅ Model loaded from: {file_path}")
        print(f"📅 Model version: {model_data.get('version', 'Unknown')}")
        print(f"🕒 Trained at: {model_data.get('timestamp', 'Unknown')}")

def main():
    """
    Demo và test Intent Classifier
    """
    print("🏥 Medical Intent Classifier - Demo")
    print("=" * 50)
    
    # Initialize classifier
    classifier = MedicalIntentClassifier()
    
    # Train model
    results = classifier.train()
    
    # Test examples
    test_cases = [
        "Tôi bị đau đầu dữ dội",
        "Paracetamol có tác dụng phụ gì không?",
        "Đau ngực không thể thở được",
        "Uống thuốc này bao nhiêu viên?",
        "Chào bạn",
        "Làm sao để khỏe mạnh?",
        "xyz random text"
    ]
    
    print("\n🧪 Testing with sample inputs:")
    print("-" * 50)
    
    for text in test_cases:
        intent, confidence = classifier.predict(text, return_confidence=True)
        max_conf = max(confidence.values())
        
        print(f"Input: '{text}'")
        print(f"Intent: {intent} (confidence: {max_conf:.3f})")
        print()

if __name__ == "__main__":
    main()