#!/usr/bin/env python3
"""
Test comprehensive với data hoàn toàn mới không có trong training set
"""

import sys
import os
import joblib
import json

# Add path để import models
sys.path.append(os.path.join(os.getcwd(), 'medical-chatbot', 'src'))

try:
    from models.medical_intent_classifier import MedicalIntentClassifier
except ImportError:
    print("⚠️ Cannot import MedicalIntentClassifier, will use fallback")
    MedicalIntentClassifier = None

def load_ner_model():
    """Load NER model 5K"""
    try:
        model_path = 'e:/MedCare/medical-chatbot/data/models/ner_model_5k.joblib'
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            return model_data['model']
        else:
            print(" NER model not found")
            return None
    except Exception as e:
        print(f" Error loading NER model: {e}")
        return None

def prepare_features(sentence):
    """Prepare features for CRF NER model"""
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

def predict_ner(model, text):
    """Predict NER entities"""
    if not model:
        return []
        
    tokens = text.split()
    features = prepare_features(tokens)
    predicted_labels = model.predict([features])[0]
    
    # Extract entities
    entities = []
    current_entity = None
    current_text = []
    
    for token, label in zip(tokens, predicted_labels):
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
    
    return entities

def test_comprehensive():
    """Test với nhiều câu hoàn toàn mới"""
    
    # Load models
    print("🔧 Loading models...")
    
    # Intent Classifier
    intent_classifier = None
    if MedicalIntentClassifier:
        try:
            intent_classifier = MedicalIntentClassifier()
            intent_classifier.train()
            print(" Intent Classifier loaded")
        except Exception as e:
            print(f" Intent Classifier error: {e}")
    
    # NER Model
    ner_model = load_ner_model()
    if ner_model:
        print(" NER Model loaded")
    
    # Test cases - hoàn toàn mới, không có trong dataset
    test_cases = [
        # Triệu chứng phức tạp
        "Bệnh nhân nam 38 tuổi đau bụng thượng vị kèm ợ chua và nóng rát dạ dày từ 5 ngày",
        "Cháu gái 14 tuổi bị mệt mỏi thường xuyên và da xanh xao gần đây",
        "Ông cụ 82 tuổi ho đờm có máu và khó thở khi nằm ngửa",
        
        # Thuốc và liều lượng đặc biệt
        "Bác sĩ kê đơn cefixime 200mg uống 12 tiếng một lần trong 7 ngày",
        "Thuốc prednisolone 5mg có gây tăng cân và giữ nước không?",
        "Liều methylprednisolone tiêm tĩnh mạch cho trẻ em như thế nào?",
        
        # Cấp cứu đa dạng
        "Khẩn cấp! Bé trai 2 tuổi nuốt phải pin cúc áo và khóc thét",
        "SOS! Người phụ nữ 45 tuổi bị va chạm xe máy gãy cẳng chân hở",
        "Cần cấp cứu ngay! Ông già bị ngạt khói trong đám cháy",
        
        # Tương tác thuốc
        "Có thể uống cùng lúc lansoprazole và clopidogrel không?",
        "Thuốc rosuvastatin có tương tác với thuốc tim digitalis?",
        "Metoprolol và amlodipine dùng chung có an toàn?",
        
        # Tác dụng phụ hiếm
        "Adalimumab có nguy cơ gây nhiễm trùng cơ hội nào?",
        "Finasteride có thể gây rối loạn chức năng sinh dục nam không?",
        "Isotretinoin có ảnh hưởng đến thai nhi như thế nào?",
        
        # Sức khỏe nâng cao
        "Chế độ ketogenic có phù hợp cho người bị rối loạn lipid máu?",
        "Intermittent fasting có lợi ích gì cho người tiểu đường type 2?",
        "Probiotics nào tốt nhất cho hội chứng ruột kích thích?",
        
        # Chào hỏi đa dạng  
        "Chào bác sĩ ơi, em muốn hỏi về bệnh của mẹ",
        "Xin chào, tôi cần tư vấn về tình trạng sức khỏe",
        "Doctor, I need medical advice about my condition",
        
        # Câu không liên quan
        "Hôm nay giá vàng thế giới tăng hay giảm?",
        "Mùa hè này nên đi du lịch Đà Lạt hay Sapa?",
        "Công thức nấu phở bò ngon như thế nào?",
        
        # Y học chuyên sâu
        "Hội chứng Stevens-Johnson do carbamazepine có nguy hiểm không?",
        "Liệu pháp miễn dịch checkpoint inhibitor cho ung thư phổi",
        "Kỹ thuật ECMO có chỉ định trong suy hô hấp nặng COVID-19?",
        
        # Trường hợp phức tạp
        "Bệnh nhân nữ 29 tuổi mang thai 28 tuần bị tiền sản giật nặng cần làm gì?",
        "Người bệnh suy tim độ 3 có thể phẫu thuật thay khớp háng được không?",
        "Trẻ sơ sinh 3 ngày tuổi vàng da mức bilirubin 18mg/dL cần can thiệp gì?",

        "Tôi bị thiếu ngủ kèo dài mấy tháng nay và chỉ có thể ngủ vào ban ngày",
        "Đau đầu quá đi ,Stress nữa"
    ]
    
    print(f"\n🧪 Testing with {len(test_cases)} completely new test cases")
    print("=" * 80)
    
    # Test từng case
    for i, text in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}:")
        print(f"📝 Text: {text}")
        
        # Test Intent Classification
        if intent_classifier:
            try:
                # Predict trả về string intent, cần get confidence riêng
                intent = intent_classifier.predict(text)
                confidence = intent_classifier.get_confidence(text)
                
                # Hiển thị intent với màu sắc
                if intent == 'emergency':
                    print(f"🚨 Intent: {intent} ({confidence:.1%}) ⚠️ EMERGENCY!")
                elif confidence > 0.8:
                    print(f"✅ Intent: {intent} ({confidence:.1%})")
                elif confidence > 0.6:
                    print(f"🟡 Intent: {intent} ({confidence:.1%})")
                else:
                    print(f"🔴 Intent: {intent} ({confidence:.1%}) - Low confidence")
                    
            except Exception as e:
                print(f"❌ Intent prediction failed: {e}")
        
        # Test NER
        if ner_model:
            try:
                entities = predict_ner(ner_model, text)
                if entities:
                    print("🏷️  Entities:")
                    for entity_type, entity_text in entities:
                        print(f"     • {entity_type}: '{entity_text}'")
                else:
                    print("🏷️  Entities: None detected")
            except Exception as e:
                print(f"❌ NER prediction failed: {e}")
        
        print("-" * 60)
    
    print("\n Test Summary:")
    print(" Tested models with completely unseen data")
    print(" Evaluated generalization capability")
    print(" Checked edge cases and complex scenarios")
    print("\n Use these results to assess real-world performance!")

if __name__ == "__main__":
    test_comprehensive()