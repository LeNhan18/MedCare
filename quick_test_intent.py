#!/usr/bin/env python3
"""
Test Intent Classifier với CSV dataset
"""

import sys
import os
sys.path.insert(0, os.path.join('medical-chatbot', 'src'))

try:
    from models.medical_intent_classifier import MedicalIntentClassifier
    print('🤖 Testing Intent Classifier với CSV Dataset')
    print('=' * 50)
    
    classifier = MedicalIntentClassifier()
    print('📊 Đang train model...')
    
    accuracy = classifier.train()
    print(f'✅ Training hoàn tất! Accuracy: {accuracy:.2%}')
    
    # Test cases từ dataset thực tế
    tests = [
        'Tôi bị giảm cân',
        'Thuốc doxycycline có tác dụng gì',
        'Cấp cứu! Đau ngực dữ dội', 
        'Liều lượng thuốc như thế nào',
        'Thuốc này có tác dụng phụ gì',
        'Làm sao để khỏe mạnh',
        'Xin chào bác sĩ',
        'Tôi bị mụn trứng cá',
        'Thuốc orlistat dùng để làm gì'
    ]
    
    print('\n🧪 Testing predictions:')
    for i, text in enumerate(tests, 1):
        intent = classifier.predict_intent(text)
        conf = classifier.get_confidence(text)
        emergency = "⚠️ EMERGENCY!" if classifier.detect_emergency(text) else ""
        print(f'{i:2d}. "{text}" → {intent} ({conf:.1%}) {emergency}')
        
    print('\n📊 Model Statistics:')
    training_data = classifier.create_training_data()
    intent_counts = {}
    for _, intent in training_data:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    for intent, count in sorted(intent_counts.items()):
        print(f'  • {intent}: {count} samples')
        
    print(f'\n✅ Intent Classifier sẵn sàng! Total: {len(training_data)} training samples')
    
except Exception as e:
    print(f'❌ Lỗi: {e}')
    import traceback
    traceback.print_exc()