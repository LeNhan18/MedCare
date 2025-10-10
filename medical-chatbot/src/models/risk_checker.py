"""
Model 4: Risk Checking / Contraindication
Mục tiêu: Kiểm tra rủi ro và chống chỉ định khi sử dụng thuốc
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import yaml
import os
import json

class RiskChecker:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.model_config = config['model']['risk_checker']
        self.input_features = self.model_config['input_features']
        self.risk_levels = self.model_config['risk_levels']
        self.confidence_threshold = self.model_config['confidence_threshold']
        self.model = None
        self.feature_encoders = {}
        self.risk_database = None
        
        # Load risk database
        self.load_risk_database()
        
    def load_risk_database(self):
        """Load cơ sở dữ liệu về rủi ro và chống chỉ định"""
        risk_data_path = "data/processed/drug_risk_database.csv"
        
        if not os.path.exists(risk_data_path):
            self.create_sample_risk_database(risk_data_path)
        
        self.risk_database = pd.read_csv(risk_data_path)
        print(f"Đã load {len(self.risk_database)} quy tắc rủi ro từ database")
        
    def create_sample_risk_database(self, path):
        """Tạo database rủi ro mẫu"""
        sample_data = [
            # Paracetamol
            {"drug_name": "Paracetamol", "risk_factor": "age", "condition": "<6months", 
             "risk_level": "contraindicated", "description": "Không dùng cho trẻ dưới 6 tháng tuổi"},
            {"drug_name": "Paracetamol", "risk_factor": "existing_conditions", "condition": "suy gan", 
             "risk_level": "contraindicated", "description": "Chống chỉ định tuyệt đối với bệnh nhân suy gan"},
            {"drug_name": "Paracetamol", "risk_factor": "current_drugs", "condition": "warfarin", 
             "risk_level": "caution", "description": "Tương tác với warfarin, cần theo dõi INR"},
            
            # Ibuprofen  
            {"drug_name": "Ibuprofen", "risk_factor": "age", "condition": "<6months",
             "risk_level": "contraindicated", "description": "Không dùng cho trẻ dưới 6 tháng tuổi"},
            {"drug_name": "Ibuprofen", "risk_factor": "existing_conditions", "condition": "loét dạ dày",
             "risk_level": "contraindicated", "description": "Chống chỉ định với loét dạ dày-tá tràng"},
            {"drug_name": "Ibuprofen", "risk_factor": "existing_conditions", "condition": "suy thận",
             "risk_level": "contraindicated", "description": "Chống chỉ định với suy thận nặng"},
            {"drug_name": "Ibuprofen", "risk_factor": "existing_conditions", "condition": "hen suyễn",
             "risk_level": "caution", "description": "Cẩn thận với bệnh nhân hen suyễn"},
            {"drug_name": "Ibuprofen", "risk_factor": "current_drugs", "condition": "ace inhibitor",
             "risk_level": "caution", "description": "Giảm tác dụng của thuốc ức chế ACE"},
            
            # Aspirin
            {"drug_name": "Aspirin", "risk_factor": "age", "condition": "<16years",
             "risk_level": "contraindicated", "description": "Nguy cơ hội chứng Reye ở trẻ em"},
            {"drug_name": "Aspirin", "risk_factor": "existing_conditions", "condition": "hen suyễn",
             "risk_level": "contraindicated", "description": "Có thể gây co thắt phế quản"},
            {"drug_name": "Aspirin", "risk_factor": "current_drugs", "condition": "warfarin",
             "risk_level": "contraindicated", "description": "Tăng nguy cơ xuất huyết"},
            
            # Loratadine
            {"drug_name": "Loratadine", "risk_factor": "age", "condition": "<2years",
             "risk_level": "contraindicated", "description": "Không dùng cho trẻ dưới 2 tuổi"},
            {"drug_name": "Loratadine", "risk_factor": "existing_conditions", "condition": "suy gan nặng",
             "risk_level": "caution", "description": "Giảm liều với bệnh nhân suy gan"},
            
            # Domperidone
            {"drug_name": "Domperidone", "risk_factor": "age", "condition": "<12years",
             "risk_level": "caution", "description": "Cần tính toán liều cẩn thận ở trẻ em"},
            {"drug_name": "Domperidone", "risk_factor": "existing_conditions", "condition": "bệnh tim",
             "risk_level": "contraindicated", "description": "Tăng nguy cơ rối loạn nhịp tim"},
            {"drug_name": "Domperidone", "risk_factor": "current_drugs", "condition": "ketoconazole",
             "risk_level": "contraindicated", "description": "Tương tác nghiêm trọng với ketoconazole"},
            
            # Dextromethorphan
            {"drug_name": "Dextromethorphan", "risk_factor": "current_drugs", "condition": "maoi",
             "risk_level": "contraindicated", "description": "Chống chỉ định với thuốc ức chế MAO"},
            {"drug_name": "Dextromethorphan", "risk_factor": "existing_conditions", "condition": "suy hô hấp",
             "risk_level": "contraindicated", "description": "Có thể ức chế hô hấp"},
            
            # Xylometazoline
            {"drug_name": "Xylometazoline", "risk_factor": "existing_conditions", "condition": "tăng huyết áp",
             "risk_level": "caution", "description": "Có thể làm tăng huyết áp"},
            {"drug_name": "Xylometazoline", "risk_factor": "existing_conditions", "condition": "tăng nhãn áp",
             "risk_level": "contraindicated", "description": "Chống chỉ định với tăng nhãn áp góc đóng"},
            
            # Chung cho phụ nữ có thai
            {"drug_name": "Ibuprofen", "risk_factor": "pregnancy", "condition": "trimester3",
             "risk_level": "contraindicated", "description": "Chống chỉ định 3 tháng cuối thai kỳ"},
            {"drug_name": "Aspirin", "risk_factor": "pregnancy", "condition": "any",
             "risk_level": "caution", "description": "Cần cân nhắc lợi ích/rủi ro"},
            {"drug_name": "Domperidone", "risk_factor": "pregnancy", "condition": "any",
             "risk_level": "caution", "description": "Chỉ dùng khi thực sự cần thiết"},
        ]
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(sample_data)
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"Đã tạo database rủi ro mẫu: {path}")
        
    def build_model(self):
        """Xây dựng model phân loại rủi ro"""
        # Input layers cho các features
        inputs = []
        processed_inputs = []
        
        # Age input
        age_input = tf.keras.layers.Input(shape=(1,), name='age')
        inputs.append(age_input)
        processed_inputs.append(age_input)
        
        # Gender input (categorical)
        gender_input = tf.keras.layers.Input(shape=(1,), name='gender')
        gender_embedded = tf.keras.layers.Embedding(3, 4)(gender_input)  # male, female, other
        gender_flattened = tf.keras.layers.Flatten()(gender_embedded)
        inputs.append(gender_input)
        processed_inputs.append(gender_flattened)
        
        # Existing conditions (multi-hot encoding)
        conditions_input = tf.keras.layers.Input(shape=(20,), name='existing_conditions')  # Max 20 conditions
        inputs.append(conditions_input)
        processed_inputs.append(conditions_input)
        
        # Current drugs (multi-hot encoding)
        drugs_input = tf.keras.layers.Input(shape=(50,), name='current_drugs')  # Max 50 drugs
        inputs.append(drugs_input)
        processed_inputs.append(drugs_input)
        
        # Drug being checked (categorical)
        target_drug_input = tf.keras.layers.Input(shape=(1,), name='target_drug')
        drug_embedded = tf.keras.layers.Embedding(100, 16)(target_drug_input)  # Max 100 drugs
        drug_flattened = tf.keras.layers.Flatten()(drug_embedded)
        inputs.append(target_drug_input)
        processed_inputs.append(drug_flattened)
        
        # Concatenate all inputs
        concat = tf.keras.layers.Concatenate()(processed_inputs)
        
        # Dense layers
        x = tf.keras.layers.Dense(128, activation='relu')(concat)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        
        # Output layer
        output = tf.keras.layers.Dense(
            len(self.risk_levels), 
            activation='softmax', 
            name='risk_prediction'
        )(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=output)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def check_risk_rule_based(self, drug_name, user_profile):
        """Kiểm tra rủi ro dựa trên quy tắc"""
        # Lọc các quy tắc cho thuốc này
        drug_rules = self.risk_database[
            self.risk_database['drug_name'].str.lower() == drug_name.lower()
        ]
        
        if drug_rules.empty:
            return {
                'risk_level': 'safe',
                'confidence': 0.8,
                'warnings': [],
                'recommendations': [f"Không có dữ liệu rủi ro cụ thể cho {drug_name}. Hãy đọc kỹ hướng dẫn sử dụng."]
            }
        
        warnings = []
        max_risk_level = 'safe'
        risk_priority = {'safe': 0, 'caution': 1, 'contraindicated': 2}
        
        for _, rule in drug_rules.iterrows():
            risk_factor = rule['risk_factor']
            condition = rule['condition']
            risk_level = rule['risk_level']
            description = rule['description']
            
            # Kiểm tra từng yếu tố rủi ro
            is_risk = False
            
            if risk_factor == 'age':
                is_risk = self._check_age_condition(user_profile.get('age'), condition)
            elif risk_factor == 'gender':
                is_risk = self._check_gender_condition(user_profile.get('gender'), condition)
            elif risk_factor == 'existing_conditions':
                is_risk = self._check_existing_conditions(user_profile.get('existing_conditions', []), condition)
            elif risk_factor == 'current_drugs':
                is_risk = self._check_current_drugs(user_profile.get('current_drugs', []), condition)
            elif risk_factor == 'pregnancy':
                is_risk = self._check_pregnancy_condition(user_profile, condition)
            
            if is_risk:
                warnings.append({
                    'risk_factor': risk_factor,
                    'condition': condition,
                    'risk_level': risk_level,
                    'description': description
                })
                
                # Cập nhật mức độ rủi ro cao nhất
                if risk_priority[risk_level] > risk_priority[max_risk_level]:
                    max_risk_level = risk_level
        
        # Tạo recommendations
        recommendations = self._generate_recommendations(drug_name, max_risk_level, warnings)
        
        return {
            'risk_level': max_risk_level,
            'confidence': 0.95 if warnings else 0.8,
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    def _check_age_condition(self, age, condition):
        """Kiểm tra điều kiện tuổi"""
        if age is None:
            return False
            
        if '<' in condition:
            limit_str = condition.replace('<', '')
            if 'months' in limit_str:
                limit_months = int(limit_str.replace('months', ''))
                return age * 12 < limit_months  # Convert age to months
            elif 'years' in limit_str:
                limit_years = int(limit_str.replace('years', ''))
                return age < limit_years
        elif '>' in condition:
            limit_str = condition.replace('>', '')
            if 'years' in limit_str:
                limit_years = int(limit_str.replace('years', ''))
                return age > limit_years
                
        return False
    
    def _check_gender_condition(self, gender, condition):
        """Kiểm tra điều kiện giới tính"""
        if gender is None:
            return False
        return gender.lower() == condition.lower()
    
    def _check_existing_conditions(self, existing_conditions, condition):
        """Kiểm tra bệnh nền"""
        if not existing_conditions:
            return False
        
        condition_lower = condition.lower()
        for existing_condition in existing_conditions:
            if condition_lower in existing_condition.lower():
                return True
        return False
    
    def _check_current_drugs(self, current_drugs, condition):
        """Kiểm tra thuốc đang dùng"""
        if not current_drugs:
            return False
            
        condition_lower = condition.lower()
        for drug in current_drugs:
            if condition_lower in drug.lower():
                return True
        return False
    
    def _check_pregnancy_condition(self, user_profile, condition):
        """Kiểm tra tình trạng thai kỳ"""
        pregnancy_status = user_profile.get('pregnancy')
        if not pregnancy_status:
            return False
            
        if condition == 'any':
            return pregnancy_status.get('is_pregnant', False)
        elif condition == 'trimester3':
            return (pregnancy_status.get('is_pregnant', False) and 
                   pregnancy_status.get('trimester', 0) == 3)
        
        return False
    
    def _generate_recommendations(self, drug_name, risk_level, warnings):
        """Tạo khuyến nghị dựa trên mức độ rủi ro"""
        recommendations = []
        
        if risk_level == 'contraindicated':
            recommendations.append(f"🚫 KHÔNG NÊN sử dụng {drug_name}")
            recommendations.append("💡 Hãy tham khảo ý kiến bác sĩ hoặc dược sĩ để tìm thuốc thay thế")
            
        elif risk_level == 'caution':
            recommendations.append(f"⚠️ Cần thận trọng khi sử dụng {drug_name}")
            recommendations.append("💡 Tham khảo ý kiến dược sĩ trước khi sử dụng")
            recommendations.append("📋 Theo dõi cẩn thận các phản ứng bất lợi")
            
        else:  # safe
            recommendations.append(f"✅ Có thể sử dụng {drug_name} theo hướng dẫn")
            recommendations.append("📖 Đọc kỹ hướng dẫn sử dụng trước khi dùng")
            recommendations.append("🏥 Nếu có triệu chứng bất thường, hãy ngưng thuốc và đến gặp bác sĩ")
        
        # Thêm khuyến nghị cụ thể cho từng warning
        for warning in warnings:
            if warning['risk_level'] == 'contraindicated':
                recommendations.append(f"❌ {warning['description']}")
            elif warning['risk_level'] == 'caution':
                recommendations.append(f"⚠️ {warning['description']}")
        
        return recommendations
    
    def check_drug_interactions(self, target_drug, current_drugs):
        """Kiểm tra tương tác thuốc"""
        interactions = []
        
        # Lọc các tương tác với thuốc đang dùng
        for current_drug in current_drugs:
            interaction_rules = self.risk_database[
                (self.risk_database['drug_name'].str.lower() == target_drug.lower()) &
                (self.risk_database['risk_factor'] == 'current_drugs') &
                (self.risk_database['condition'].str.lower() == current_drug.lower())
            ]
            
            for _, rule in interaction_rules.iterrows():
                interactions.append({
                    'drug1': target_drug,
                    'drug2': current_drug,
                    'risk_level': rule['risk_level'],
                    'description': rule['description']
                })
        
        return interactions
    
    def get_safety_profile(self, drug_name):
        """Lấy hồ sơ an toàn của thuốc"""
        drug_risks = self.risk_database[
            self.risk_database['drug_name'].str.lower() == drug_name.lower()
        ]
        
        if drug_risks.empty:
            return None
        
        safety_profile = {
            'drug_name': drug_name,
            'contraindications': [],
            'cautions': [],
            'age_restrictions': [],
            'pregnancy_category': 'unknown',
            'drug_interactions': []
        }
        
        for _, risk in drug_risks.iterrows():
            risk_info = {
                'factor': risk['risk_factor'],
                'condition': risk['condition'],
                'description': risk['description']
            }
            
            if risk['risk_level'] == 'contraindicated':
                safety_profile['contraindications'].append(risk_info)
            elif risk['risk_level'] == 'caution':
                safety_profile['cautions'].append(risk_info)
            
            if risk['risk_factor'] == 'age':
                safety_profile['age_restrictions'].append(risk_info)
            elif risk['risk_factor'] == 'pregnancy':
                safety_profile['pregnancy_category'] = risk['risk_level']
            elif risk['risk_factor'] == 'current_drugs':
                safety_profile['drug_interactions'].append(risk_info)
        
        return safety_profile
    
    def save_model(self, path=None):
        """Lưu model"""
        if path is None:
            path = self.model_config['path']
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if self.model:
            self.model.save(path)
        
        # Save encoders
        encoders_path = path.replace('.h5', '_encoders.json')
        with open(encoders_path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_encoders, f, ensure_ascii=False, indent=2)
    
    def load_model(self, path=None):
        """Load model đã huấn luyện"""
        if path is None:
            path = self.model_config['path']
        
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path)
            
            # Load encoders
            encoders_path = path.replace('.h5', '_encoders.json')
            if os.path.exists(encoders_path):
                with open(encoders_path, 'r', encoding='utf-8') as f:
                    self.feature_encoders = json.load(f)
        else:
            print(f"Model file không tồn tại: {path}")
            self.build_model()


# Example usage và test
if __name__ == "__main__":
    # Test user profiles
    test_profiles = [
        {
            'age': 25,
            'gender': 'female',
            'existing_conditions': ['hen suyễn'],
            'current_drugs': ['salbutamol'],
            'pregnancy': {'is_pregnant': False}
        },
        {
            'age': 5,
            'gender': 'male',
            'existing_conditions': [],
            'current_drugs': []
        },
        {
            'age': 45,
            'gender': 'male',
            'existing_conditions': ['suy thận', 'tăng huyết áp'],
            'current_drugs': ['ace inhibitor', 'diuretic']
        }
    ]
    
    test_drugs = ['Paracetamol', 'Ibuprofen', 'Aspirin']
    
    # Initialize risk checker
    checker = RiskChecker()
    
    print("Model Risk Checker đã được khởi tạo!")
    print(f"Có {len(checker.risk_database)} quy tắc rủi ro")
    
    # Test risk checking
    for i, profile in enumerate(test_profiles):
        print(f"\n=== Test Profile {i+1} ===")
        print(f"Tuổi: {profile['age']}, Giới: {profile['gender']}")
        print(f"Bệnh nền: {profile.get('existing_conditions', [])}")
        print(f"Thuốc đang dùng: {profile.get('current_drugs', [])}")
        
        for drug in test_drugs[:2]:  # Test 2 thuốc đầu
            print(f"\n--- Kiểm tra {drug} ---")
            result = checker.check_risk_rule_based(drug, profile)
            
            print(f"Mức độ rủi ro: {result['risk_level']} (confidence: {result['confidence']:.2f})")
            
            if result['warnings']:
                print("Cảnh báo:")
                for warning in result['warnings']:
                    print(f"  - {warning['description']} ({warning['risk_level']})")
            
            print("Khuyến nghị:")
            for rec in result['recommendations']:
                print(f"  {rec}")
            print()