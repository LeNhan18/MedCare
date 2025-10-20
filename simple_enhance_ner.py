"""
Simple Enhanced NER Data Generator  
Không cần external dependencies
"""

import json
import os

def create_enhanced_disease_examples():
    """Tạo examples cho DISEASE detection"""
    diseases = [
        'viêm họng', 'viêm phổi', 'viêm dạ dày', 'viêm gan', 'viêm thận',
        'tiểu đường', 'cao huyết áp', 'hạ huyết áp', 'gout', 'hen suyễn',
        'trầm cảm', 'lo âu', 'mất ngủ', 'đau nửa đầu', 'cảm lạnh', 'cảm cúm'
    ]
    
    templates = [
        "Tôi bị {disease} có nguy hiểm không?",
        "{disease} có thể điều trị được không?", 
        "Triệu chứng của {disease} là gì?",
        "Người bệnh {disease} cần kiêng gì?",
        "Nguyên nhân gây {disease} là gì?",
    ]
    
    examples = []
    for disease in diseases:
        for template in templates:
            text = template.format(disease=disease)
            tokens = text.split()
            labels = ['O'] * len(tokens)
            
            # Find and label disease
            disease_tokens = disease.split()
            for i in range(len(tokens) - len(disease_tokens) + 1):
                if tokens[i:i+len(disease_tokens)] == disease_tokens:
                    labels[i] = 'B-DISEASE'
                    for j in range(1, len(disease_tokens)):
                        labels[i+j] = 'I-DISEASE'
                    break
            
            examples.append({
                'text': text,
                'tokens': tokens,
                'labels': labels,
                'intent': 'general_health'
            })
    
    return examples

def create_enhanced_body_part_examples():
    """Tạo examples cho BODY_PART detection"""
    body_parts = [
        'đầu', 'mặt', 'mắt', 'tai', 'mũi', 'miệng', 'họng', 'cổ',
        'vai', 'tay', 'ngực', 'lưng', 'bụng', 'chân', 'tim', 'phổi',
        'gan', 'thận', 'dạ dày', 'da'
    ]
    
    templates = [
        "Đau ở {body_part} có nguy hiểm không?",
        "{body_part} tôi bị sưng và đỏ",
        "Làm sao để giảm đau {body_part}?",
        "{body_part} bị ngứa",
        "Chăm sóc {body_part} như thế nào?",
    ]
    
    examples = []
    for body_part in body_parts:
        for template in templates:
            text = template.format(body_part=body_part)
            tokens = text.split()
            labels = ['O'] * len(tokens)
            
            # Find and label body part
            body_part_tokens = body_part.split()
            for i in range(len(tokens) - len(body_part_tokens) + 1):
                if tokens[i:i+len(body_part_tokens)] == body_part_tokens:
                    labels[i] = 'B-BODY_PART'
                    for j in range(1, len(body_part_tokens)):
                        labels[i+j] = 'I-BODY_PART'
                    break
            
            examples.append({
                'text': text,
                'tokens': tokens,
                'labels': labels,
                'intent': 'symptom_inquiry'
            })
    
    return examples

def main():
    print("🚀 Creating Enhanced NER Training Data...")
    
    # Load existing data
    original_path = 'z:/MedCare/medical-chatbot/data/processed/ner_training_data.json'
    
    try:
        with open(original_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        print(f"Loaded {len(original_data)} original samples")
    except Exception as e:
        print(f"Error loading original data: {e}")
        original_data = []
    
    # Generate enhanced examples
    print("Generating DISEASE examples...")
    disease_examples = create_enhanced_disease_examples()
    print(f"Created {len(disease_examples)} DISEASE examples")
    
    print("Generating BODY_PART examples...")
    body_part_examples = create_enhanced_body_part_examples()
    print(f"Created {len(body_part_examples)} BODY_PART examples")
    
    # Combine data
    enhanced_data = original_data + disease_examples + body_part_examples
    
    print(f"\n📊 ENHANCED DATASET SUMMARY:")
    print(f"  Original samples: {len(original_data)}")
    print(f"  DISEASE samples: {len(disease_examples)}")
    print(f"  BODY_PART samples: {len(body_part_examples)}")
    print(f"  Total samples: {len(enhanced_data)}")
    
    # Save enhanced data
    output_path = 'z:/MedCare/medical-chatbot/data/processed/ner_training_data_enhanced.json'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Enhanced data saved to: {output_path}")
    
    # Show some examples
    print(f"\n🧪 SAMPLE ENHANCED EXAMPLES:")
    
    print("\nDISEASE examples:")
    for i, example in enumerate(disease_examples[:3]):
        print(f"  {i+1}. {example['text']}")
        entity_pairs = [(token, label) for token, label in zip(example['tokens'], example['labels']) if label != 'O']
        print(f"     Entities: {entity_pairs}")
    
    print("\nBODY_PART examples:")  
    for i, example in enumerate(body_part_examples[:3]):
        print(f"  {i+1}. {example['text']}")
        entity_pairs = [(token, label) for token, label in zip(example['tokens'], example['labels']) if label != 'O']
        print(f"     Entities: {entity_pairs}")

if __name__ == "__main__":
    main()