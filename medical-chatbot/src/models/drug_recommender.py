import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DrugRecommender:
    def __init__(self, symptom_data, drug_data):
        self.symptom_data = symptom_data
        self.drug_data = drug_data
        self.model = None
        self.label_encoder = LabelEncoder()
        self.prepare_data()

    def prepare_data(self):
        self.drug_data['label'] = self.label_encoder.fit_transform(self.drug_data['drug_name'])
        self.X = self.symptom_data['symptoms']
        self.y = self.drug_data['label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=1000, output_dim=64))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(64))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(len(self.label_encoder.classes_), activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, epochs=10, batch_size=32):
        self.X_train = pad_sequences(self.X_train, padding='post')
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

    """
Model 3: Drug Recommendation System
Mục tiêu: Gợi ý thuốc OTC dựa trên triệu chứng
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import yaml
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json

class DrugRecommender:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.model_config = config['model']['drug_recommender']
        self.embedding_dim = self.model_config['embedding_dim']
        self.top_k = self.model_config['top_k']
        self.similarity_threshold = self.model_config['similarity_threshold']
        self.model_type = self.model_config['model_type']
        
        self.sentence_model = None
        self.drug_database = None
        self.symptom_embeddings = None
        self.drug_embeddings = None
        self.model = None
        
        # Load drug database
        self.load_drug_database()
        
    def load_drug_database(self):
        """Load cơ sở dữ liệu thuốc và triệu chứng"""
        # Tạo database mẫu nếu chưa có
        drug_data_path = "data/processed/drug_symptom_mapping.csv"
        
        if not os.path.exists(drug_data_path):
            self.create_sample_drug_database(drug_data_path)
        
        self.drug_database = pd.read_csv(drug_data_path)
        print(f"Đã load {len(self.drug_database)} thuốc từ database")
        
    def create_sample_drug_database(self, path):
        """Tạo database thuốc mẫu"""
        sample_data = [
            # Thuốc giảm đau, hạ sốt
            {"drug_name": "Paracetamol", "active_ingredient": "Paracetamol", 
             "symptoms": "đau đầu, sốt, đau cơ, đau răng", 
             "medical_condition": "giảm đau, hạ sốt",
             "dosage": "500mg-1000mg mỗi 6-8 giờ",
             "contraindications": "suy gan nặng, dị ứng paracetamol",
             "age_restriction": "trên 6 tháng tuổi",
             "drug_type": "OTC"},
            
            {"drug_name": "Ibuprofen", "active_ingredient": "Ibuprofen",
             "symptoms": "đau đầu, sốt, viêm, đau cơ, đau khớp",
             "medical_condition": "giảm đau, hạ sốt, chống viêm",
             "dosage": "200mg-400mg mỗi 6-8 giờ", 
             "contraindications": "loét dạ dày, suy thận, dị ứng aspirin",
             "age_restriction": "trên 6 tháng tuổi",
             "drug_type": "OTC"},
            
            # Thuốc cảm lạnh
            {"drug_name": "Loratadine", "active_ingredient": "Loratadine",
             "symptoms": "nghẹt mũi, chảy nước mũi, hắt hơi, ngứa mũi",
             "medical_condition": "dị ứng, viêm mũi dị ứng",
             "dosage": "10mg mỗi ngày",
             "contraindications": "dị ứng loratadine, suy gan nặng", 
             "age_restriction": "trên 2 tuổi",
             "drug_type": "OTC"},
            
            {"drug_name": "Xylometazoline", "active_ingredient": "Xylometazoline HCl",
             "symptoms": "nghẹt mũi, viêm mũi", 
             "medical_condition": "thu nhỏ mạch máu mũi",
             "dosage": "2-3 lần/ngày, không quá 7 ngày",
             "contraindications": "tăng nhãn áp, tăng huyết áp nặng",
             "age_restriction": "trên 6 tuổi", 
             "drug_type": "OTC"},
            
            # Thuốc tiêu hóa
            {"drug_name": "Domperidone", "active_ingredient": "Domperidone",
             "symptoms": "buồn nôn, nôn, đầy bụng, khó tiêu",
             "medical_condition": "rối loạn tiêu hóa", 
             "dosage": "10mg, 3 lần/ngày trước ăn",
             "contraindications": "tắc ruột, xuất huyết tiêu hóa",
             "age_restriction": "trên 12 tuổi",
             "drug_type": "OTC"},
            
            {"drug_name": "Smecta", "active_ingredient": "Diosmectite",
             "symptoms": "tiêu chảy, đau bụng, khó tiêu",
             "medical_condition": "tiêu chảy cấp và mãn tính",
             "dosage": "3 gói/ngày",
             "contraindications": "tắc ruột, dị ứng thành phần",
             "age_restriction": "mọi lứa tuổi",
             "drug_type": "OTC"},
            
            # Thuốc ho
            {"drug_name": "Bromhexine", "active_ingredient": "Bromhexine HCl", 
             "symptoms": "ho có đờm, khó khạc đờm",
             "medical_condition": "long đờm",
             "dosage": "8mg, 3 lần/ngày",
             "contraindications": "loét dạ dày, dị ứng bromhexine",
             "age_restriction": "trên 6 tuổi",
             "drug_type": "OTC"},
            
            {"drug_name": "Dextromethorphan", "active_ingredient": "Dextromethorphan HBr",
             "symptoms": "ho khan, ho không đờm", 
             "medical_condition": "ức chế ho",
             "dosage": "15mg, 3-4 lần/ngày",
             "contraindications": "suy hô hấp, dùng MAOI",
             "age_restriction": "trên 6 tuổi", 
             "drug_type": "OTC"},
            
            # Thuốc ngoài da
            {"drug_name": "Betadine", "active_ingredient": "Povidone iodine",
             "symptoms": "vết thương, nhiễm trùng da, nứt nẻ",
             "medical_condition": "sát trùng",
             "dosage": "thoa 2-3 lần/ngày",
             "contraindications": "dị ứng iod, rối loạn tuyến giáp",
             "age_restriction": "mọi lứa tuổi",
             "drug_type": "OTC"},
            
            {"drug_name": "Hydrocortisone", "active_ingredient": "Hydrocortisone",
             "symptoms": "ngứa, viêm da, eczema, dị ứng da",
             "medical_condition": "chống viêm da",  
             "dosage": "thoa mỏng 2-3 lần/ngày",
             "contraindications": "nhiễm trùng da, mụn trứng cá",
             "age_restriction": "trên 2 tuổi",
             "drug_type": "OTC"}
        ]
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(sample_data)
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"Đã tạo database thuốc mẫu: {path}")
        
    def build_embedding_model(self):
        """Xây dựng model embedding"""
        if self.model_type == "embedding_similarity":
            # Sử dụng sentence transformer
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # Tạo embeddings cho symptoms và drugs
            self._create_embeddings()
            
        elif self.model_type == "neural_collaborative":
            # Neural collaborative filtering model
            self._build_ncf_model()
    
    def _create_embeddings(self):
        """Tạo embeddings cho triệu chứng và thuốc"""
        # Tạo embeddings cho triệu chứng
        symptom_texts = []
        for _, row in self.drug_database.iterrows():
            symptom_texts.append(row['symptoms'])
        
        self.symptom_embeddings = self.sentence_model.encode(symptom_texts)
        
        # Tạo embeddings cho thuốc (tên + hoạt chất + công dụng)
        drug_texts = []
        for _, row in self.drug_database.iterrows():
            drug_text = f"{row['drug_name']} {row['active_ingredient']} {row['medical_condition']}"
            drug_texts.append(drug_text)
            
        self.drug_embeddings = self.sentence_model.encode(drug_texts)
        
        print(f"Đã tạo embeddings: {self.symptom_embeddings.shape}, {self.drug_embeddings.shape}")
    
    def _build_ncf_model(self):
        """Xây dựng Neural Collaborative Filtering model"""
        # Simplified NCF model
        num_symptoms = len(self.drug_database)
        num_drugs = len(self.drug_database)
        
        # Input layers
        symptom_input = tf.keras.layers.Input(shape=(), name='symptom_id')
        drug_input = tf.keras.layers.Input(shape=(), name='drug_id')
        
        # Embedding layers
        symptom_embedding = tf.keras.layers.Embedding(
            num_symptoms, self.embedding_dim
        )(symptom_input)
        drug_embedding = tf.keras.layers.Embedding(
            num_drugs, self.embedding_dim  
        )(drug_input)
        
        # Flatten
        symptom_vec = tf.keras.layers.Flatten()(symptom_embedding)
        drug_vec = tf.keras.layers.Flatten()(drug_embedding)
        
        # Concatenate
        concat = tf.keras.layers.Concatenate()([symptom_vec, drug_vec])
        
        # Dense layers
        x = tf.keras.layers.Dense(128, activation='relu')(concat)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Output layer
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        self.model = tf.keras.Model(
            inputs=[symptom_input, drug_input],
            outputs=output
        )
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def recommend_drugs(self, symptoms_text, user_profile=None):
        """Gợi ý thuốc dựa trên triệu chứng"""
        if self.model_type == "embedding_similarity":
            return self._recommend_by_similarity(symptoms_text, user_profile)
        elif self.model_type == "neural_collaborative":
            return self._recommend_by_ncf(symptoms_text, user_profile)
    
    def _recommend_by_similarity(self, symptoms_text, user_profile):
        """Gợi ý thuốc bằng cosine similarity"""
        if self.sentence_model is None:
            self.build_embedding_model()
        
        # Encode triệu chứng đầu vào
        query_embedding = self.sentence_model.encode([symptoms_text])
        
        # Tính similarity với tất cả triệu chứng trong database
        similarities = cosine_similarity(query_embedding, self.symptom_embeddings)[0]
        
        # Lấy top-k thuốc có similarity cao nhất
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] >= self.similarity_threshold:
                drug_info = self.drug_database.iloc[idx].to_dict()
                drug_info['similarity_score'] = float(similarities[idx])
                drug_info['recommendation_reason'] = self._generate_reason(symptoms_text, drug_info)
                recommendations.append(drug_info)
        
        # Lọc theo user profile nếu có
        if user_profile:
            recommendations = self._filter_by_profile(recommendations, user_profile)
        
        return recommendations
    
    def _recommend_by_ncf(self, symptoms_text, user_profile):
        """Gợi ý thuốc bằng Neural Collaborative Filtering"""
        # Simplified implementation
        # Trong thực tế cần encode symptoms thành IDs
        recommendations = []
        
        for idx, row in self.drug_database.iterrows():
            # Predict compatibility score
            score = np.random.random()  # Placeholder
            
            if score >= self.similarity_threshold:
                drug_info = row.to_dict()
                drug_info['compatibility_score'] = float(score)
                drug_info['recommendation_reason'] = self._generate_reason(symptoms_text, drug_info)
                recommendations.append(drug_info)
        
        # Sort by score và lấy top-k
        recommendations.sort(key=lambda x: x['compatibility_score'], reverse=True)
        return recommendations[:self.top_k]
    
    def _generate_reason(self, symptoms_text, drug_info):
        """Tạo lý do gợi ý thuốc"""
        return f"Thuốc {drug_info['drug_name']} phù hợp vì có tác dụng điều trị: {drug_info['medical_condition']}"
    
    def _filter_by_profile(self, recommendations, user_profile):
        """Lọc thuốc theo thông tin người dùng"""
        filtered = []
        
        for drug in recommendations:
            # Kiểm tra độ tuổi
            if 'age' in user_profile:
                age_restriction = drug.get('age_restriction', '')
                if 'trên' in age_restriction:
                    min_age = int(age_restriction.split('trên ')[1].split(' ')[0])
                    if user_profile['age'] < min_age:
                        continue
            
            # Kiểm tra chống chỉ định
            if 'existing_conditions' in user_profile:
                contraindications = drug.get('contraindications', '').lower()
                user_conditions = [c.lower() for c in user_profile['existing_conditions']]
                
                has_contraindication = False
                for condition in user_conditions:
                    if condition in contraindications:
                        has_contraindication = True
                        break
                
                if has_contraindication:
                    drug['warning'] = f"Cảnh báo: Có thể chống chỉ định với bệnh nền của bạn"
            
            filtered.append(drug)
        
        return filtered
    
    def get_drug_info(self, drug_name):
        """Lấy thông tin chi tiết của thuốc"""
        drug_data = self.drug_database[
            self.drug_database['drug_name'].str.lower() == drug_name.lower()
        ]
        
        if not drug_data.empty:
            return drug_data.iloc[0].to_dict()
        else:
            return None
    
    def save_model(self, path=None):
        """Lưu model"""
        if path is None:
            path = self.model_config['path']
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if self.model_type == "embedding_similarity":
            # Lưu embeddings
            embeddings_data = {
                'symptom_embeddings': self.symptom_embeddings.tolist(),
                'drug_embeddings': self.drug_embeddings.tolist()
            }
            
            embeddings_path = path.replace('.h5', '_embeddings.json')
            with open(embeddings_path, 'w', encoding='utf-8') as f:
                json.dump(embeddings_data, f)
                
        elif self.model_type == "neural_collaborative":
            if self.model:
                self.model.save(path)
    
    def load_model(self, path=None):
        """Load model đã huấn luyện"""
        if path is None:
            path = self.model_config['path']
        
        if self.model_type == "embedding_similarity":
            # Load embeddings
            embeddings_path = path.replace('.h5', '_embeddings.json')
            if os.path.exists(embeddings_path):
                with open(embeddings_path, 'r', encoding='utf-8') as f:
                    embeddings_data = json.load(f)
                    self.symptom_embeddings = np.array(embeddings_data['symptom_embeddings'])
                    self.drug_embeddings = np.array(embeddings_data['drug_embeddings'])
                
                # Load sentence transformer
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            else:
                print("Embeddings không tồn tại, tạo mới...")
                self.build_embedding_model()
                
        elif self.model_type == "neural_collaborative":
            if os.path.exists(path):
                self.model = tf.keras.models.load_model(path)
            else:
                print(f"Model file không tồn tại: {path}")
                self.build_embedding_model()

    def recommend(self, symptoms):
        """Backward compatibility method"""
        return self.recommend_drugs(symptoms)