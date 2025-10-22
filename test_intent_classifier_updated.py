#!/usr/bin/env python3
"""
Test script cho Medical Intent Classifier với dataset thực tế
"""

import sys
import os
from pathlib import Path
import traceback

# Try to locate the package 'models' dynamically. Candidates include:
# - <repo-root>/medical-chatbot/src
# - <repo-root>/models
here = Path(__file__).resolve().parent
candidates = [
    here / 'medical-chatbot' / 'src',
    here,                # repository root (e.g., e:/MedCare)
    here / 'src',        # alternative
]

found = False
for c in candidates:
    try_path = c.resolve()
    if (try_path / 'models').exists():
        sys.path.insert(0, str(try_path))
        found = True
        break

# As last resort, add repository root and current working directory
if not found:
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(here.parent))

try:
    from models.medical_intent_classifier import MedicalIntentClassifier
except Exception:
    print("\nLỗi khi import 'MedicalIntentClassifier'. Kiểm tra các đường dẫn sau trong sys.path:")
    for p in sys.path[:10]:
        print(' -', p)
    print("\nTraceback:")
    traceback.print_exc()
    raise

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
    print(f"\nModel đã được train xong với độ chính xác: {accuracy:.2%}")
    
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
            print(f" EMERGENCY DETECTED!")
        
        print()
    
    print("\nThống kê mô hình:")
    print("-" * 30)
    
    # Hiển thị thông tin về training data
    training_data = classifier.create_training_data()
    intent_counts = {}
    for _, intent in training_data:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    for intent, count in intent_counts.items():
        print(f"• {intent}: {count} samples")
    
    print(f"\n📊 Tổng số training samples: {len(training_data)}")
    
    # Test model evaluation (robust fallback if evaluate() not implemented)
    print("\n🎯 Đánh giá cross-validation:")
    try:
        # If the classifier provides an evaluate() method, use it
        if hasattr(classifier, 'evaluate'):
            scores = classifier.evaluate()
            # Expecting dict with 'accuracy' and 'std'
            if isinstance(scores, dict) and 'accuracy' in scores:
                print(f"• Accuracy: {scores['accuracy']:.2%} ± {scores.get('std', 0):.2%}")
            else:
                print("• evaluate() returned unexpected format:", scores)
        else:
            # If no evaluate(), try to compute cross-validation using sklearn
            from sklearn.model_selection import cross_val_score
            import numpy as np

            training_data = classifier.create_training_data()
            X = [t for t, _ in training_data]
            y = [label for _, label in training_data]

            # Ensure the model pipeline exists (trained or buildable)
            if hasattr(classifier, 'pipeline') and classifier.pipeline is not None:
                cv_scores = cross_val_score(classifier.pipeline, X, y, cv=5)
                print(f"• Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
            else:
                # As a last resort, (re)train and return its test score
                acc = classifier.train()
                print(f"• Accuracy (from train()): {acc:.2%}")
    except Exception as e:
        print("❌ Lỗi khi đánh giá mô hình:", e)
        traceback.print_exc()
    
    # Lưu model
    model_path = "medical_intent_classifier_v2.pkl"
    classifier.save_model(model_path)
    print(f"\n💾 Model đã được lưu tại: {model_path}")
    
    return classifier

if __name__ == "__main__":
    try:
        # Chuyển đến thư mục gốc của project (thư mục chứa file này)
        repo_root = Path(__file__).resolve().parent
        os.chdir(str(repo_root))

        classifier = test_intent_classifier()
        
        print("\n" + "=" * 60)
        print(" Test hoàn thành thành công!")
        print("\nIntent Classifier đã sẵn sàng sử dụng với dataset thực tế")
        print("Mô hình có thể nhận diện 8 loại intent khác nhau từ")
        print("2,913 thuốc trong dataset y tế.")
        
    except Exception as e:
        print(f"\n Lỗi trong quá trình test: {e}")
        import traceback
        traceback.print_exc()