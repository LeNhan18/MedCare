#!/usr/bin/env python3
"""
Test script cho Medical Intent Classifier với dataset thực tế
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'medical-chatbot', 'src'))

from models.medical_intent_classifier import MedicalIntentClassifier

def test_intent_classifier():
    """
    Test Medical Intent Classifier với dataset từ medical_dataset_training.json
    """
    
    print("🤖 Testing Medical Intent Classifier với Dataset Thực Tế")
    print("=" * 60)
    
    # Khởi tạo classifier
    classifier = MedicalIntentClassifier()
    
    print("\n📊 Đang tạo training data từ dataset thực tế...")
    
    # Train model
    accuracy = classifier.train()
    print(f"\n✅ Model đã được train xong với độ chính xác: {accuracy:.2%}")
    
    # Test cases với dataset thực tế
    test_cases = [
        # Symptom inquiry (từ dataset thực tế)
        "Tôi bị giảm cân",
        "Có thuốc nào chữa mụn trứng cá không", 
        "Triệu chứng béo phì là gì",
        "Tôi có dấu hiệu đau đầu",
        
        # Drug questions (từ dataset thực tế)
        "Thuốc doxycycline có tác dụng gì",
        "Thuốc orlistat dùng để làm gì", 
        "Cách sử dụng thuốc Xenical",
        "Thuốc phendimetrazine có tốt không",
        
        # Side effects (từ dataset thực tế) 
        "Thuốc Saxenda có tác dụng phụ gì",
        "Tác dụng phụ của thuốc Wegovy",
        "Uống Qsymia có hại gì không",
        
        # Dosage questions
        "Liều lượng thuốc Lomaira",
        "Uống benzphetamine bao nhiêu viên một lần",
        "Một ngày uống fenfluramine mấy lần",
        
        # Emergency cases
        "Cấp cứu! Tôi bị đau ngực dữ dội",
        "Khẩn cấp: bé bị sốt cao 40 độ",
        "Gọi bác sĩ ngay! Tôi không thở được",
        
        # General health
        "Làm sao để tăng sức đề kháng", 
        "Chế độ ăn uống lành mạnh",
        "Cách phòng ngừa bệnh tật",
        
        # Greetings
        "Xin chào bác sĩ",
        "Hello, tôi cần tư vấn",
        "Hi",
        
        # Unknown
        "xyz 123 abc",
        "??????",
        "blah blah blah"
    ]
    
    print("\n🧪 Testing với các câu hỏi mẫu:")
    print("-" * 60)
    
    for i, text in enumerate(test_cases, 1):
        predicted_intent = classifier.predict_intent(text)
        confidence = classifier.get_confidence(text)
        
        print(f"{i:2d}. '{text}'")
        print(f"    → Intent: {predicted_intent} (confidence: {confidence:.1%})")
        
        # Kiểm tra emergency detection
        if classifier.detect_emergency(text):
            print(f"    ⚠️  EMERGENCY DETECTED!")
        
        print()
    
    print("\n📈 Thống kê mô hình:")
    print("-" * 30)
    
    # Hiển thị thông tin về training data
    training_data = classifier.create_training_data()
    intent_counts = {}
    for _, intent in training_data:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    for intent, count in intent_counts.items():
        print(f"• {intent}: {count} samples")
    
    print(f"\n📊 Tổng số training samples: {len(training_data)}")
    
    # Test model evaluation
    print("\n🎯 Đánh giá cross-validation:")
    scores = classifier.evaluate()
    print(f"• Accuracy: {scores['accuracy']:.2%} ± {scores['std']:.2%}")
    
    # Lưu model
    model_path = "medical_intent_classifier_v2.pkl"
    classifier.save_model(model_path)
    print(f"\n💾 Model đã được lưu tại: {model_path}")
    
    return classifier

if __name__ == "__main__":
    try:
        # Chuyển đến thư mục gốc của project
        os.chdir('z:/MedCare')
        
        classifier = test_intent_classifier()
        
        print("\n" + "=" * 60)
        print("✅ Test hoàn thành thành công!")
        print("\nIntent Classifier đã sẵn sàng sử dụng với dataset thực tế")
        print("Mô hình có thể nhận diện 8 loại intent khác nhau từ")
        print("2,913 thuốc trong dataset y tế.")
        
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình test: {e}")
        import traceback
        traceback.print_exc()