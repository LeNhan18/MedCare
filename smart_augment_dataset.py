#!/usr/bin/env python3
"""
Smart Dataset Augmentation for Medical Intent Classification
Tăng cường dataset thông minh - tránh trùng lặp và học vẹt
"""

import pandas as pd
import numpy as np
import json
import re
import random
from typing import List, Dict, Set
from collections import defaultdict, Counter
import itertools

class SmartDatasetAugmenter:
    def __init__(self):
        # Synonyms và variations cho từng loại intent
        self.symptom_variations = {
            'mệt mỏi': ['mệt mỏi', 'kiệt sức', 'mệt lả', 'uể oải', 'mỏi mệt', 'thiếu sức lực', 'không còn sức'],
            'đau đầu': ['đau đầu', 'nhức đầu', 'đau nửa đầu', 'choáng váng', 'hoa mắt', 'chóng mặt'],
            'đau bụng': ['đau bụng', 'đau dạ dày', 'đau tụy', 'buồn nôn', 'khó tiêu'],
            'sốt': ['sốt', 'sốt cao', 'sốt nhẹ', 'ớn lạnh', 'run rẩy', 'nóng người'],
            'ho': ['ho', 'ho khan', 'ho có đờm', 'ho mãn tính', 'ho không ngừng'],
            'đau họng': ['đau họng', 'viêm họng', 'khô họng', 'rát họng', 'nuốt khó'],
            'chán ăn': ['chán ăn', 'mất cảm giác ngon miệng', 'không muốn ăn', 'biếng ăn'],
            'ngứa': ['ngứa', 'ngứa mẩn đỏ', 'ngứa khó chịu', 'da ngứa'],
            'tiêu chảy': ['tiêu chảy', 'đi ngoài lỏng', 'bụng xoắn', 'đau bụng đi ngoài'],
            'táo bón': ['táo bón', 'khó đi ngoài', 'đại tiện khó', 'bí bụng'],
            'khó thở': ['khó thở', 'thở gấp', 'hụt hơi', 'thiếu hơi', 'ngạt thở']
        }
        
        self.drug_variations = {
            'paracetamol': ['paracetamol', 'acetaminophen', 'Panadol', 'Efferalgan', 'Tylenol'],
            'amoxicillin': ['amoxicillin', 'Amoxil', 'Augmentin', 'Clamoxyl'],
            'aspirin': ['aspirin', 'acetylsalicylic acid', 'Cardiprin', 'Bayer'],
            'ibuprofen': ['ibuprofen', 'Brufen', 'Advil', 'Nurofen'],
            'metformin': ['metformin', 'Glucophage', 'Diabetmin'],
            'omeprazole': ['omeprazole', 'Losec', 'Prilosec'],
            'cetirizine': ['cetirizine', 'Zyrtec', 'Reactine']
        }
        
        # Các cách diễn đạt khác nhau
        self.question_patterns = {
            'symptom_inquiry': [
                "Tôi bị {symptom}",
                "Tôi có triệu chứng {symptom}",
                "Có dấu hiệu {symptom}",
                "Xuất hiện triệu chứng {symptom}",
                "Mấy ngày nay {symptom}",
                "Tuần này {symptom}",
                "Hôm nay {symptom}",
                "Từ sáng {symptom}",
                "Gần đây {symptom}",
                "Làm sao để chữa {symptom}?",
                "Cách điều trị {symptom}",
                "Thuốc gì chữa {symptom}?",
                "Triệu chứng {symptom} là gì?",
                "{family_member} bị {symptom}",
                "Con tôi có {symptom}",
                "Vợ/chồng tôi {symptom}"
            ],
            'drug_question': [
                "Thuốc {drug} có tác dụng gì?",
                "{drug} là thuốc gì?",
                "Công dụng của {drug}",
                "Tác dụng thuốc {drug}",
                "{drug} dùng để làm gì?",
                "Thuốc {drug} chữa bệnh gì?",
                "Cách sử dụng thuốc {drug}",
                "Thành phần của {drug}",
                "Thông tin về thuốc {drug}",
                "{drug} có hiệu quả không?"
            ],
            'dosage_question': [
                "Liều lượng thuốc {drug}",
                "Cách uống {drug}",
                "Thuốc {drug} uống lúc nào?",
                "Thuốc {drug} uống trước hay sau ăn?",
                "Một ngày uống {drug} mấy lần?",
                "Liều dùng {drug} cho người lớn",
                "Cách dùng {drug} đúng cách",
                "Thời gian uống {drug}",
                "Khoảng cách giữa các lần uống {drug}",
                "Lịch uống thuốc {drug}"
            ],
            'side_effects': [
                "Thuốc {drug} có tác dụng phụ gì?",
                "Tác dụng phụ của {drug}",
                "Sau khi uống {drug} bị {symptom}",
                "Uống {drug} có hại gì không?",
                "Phản ứng phụ của {drug}",
                "Tác hại của thuốc {drug}",
                "{drug} có an toàn không?",
                "Lưu ý khi dùng {drug}",
                "Chống chỉ định của {drug}",
                "Thuốc {drug} có độc không?"
            ]
        }
        
        self.family_members = [
            "con tôi", "vợ tôi", "chồng tôi", "bố tôi", "mẹ tôi", 
            "bà ngoại", "ông ngoại", "em bé", "anh ấy", "chị ấy",
            "cậu bé", "cô gái", "người bạn", "đồng nghiệp"
        ]
        
        self.time_expressions = [
            "hôm nay", "hôm qua", "mấy ngày nay", "tuần này", "tháng này",
            "từ sáng", "từ tối qua", "từ hôm qua", "gần đây", "thời gian gần đây",
            "một thời gian", "lâu nay", "mới đây"
        ]
        
    def load_existing_data(self, file_path: str) -> pd.DataFrame:
        """Load existing clean dataset"""
        df = pd.read_csv(file_path)
        # Remove quotes if any
        df['text'] = df['text'].str.strip('"')
        return df
    
    def extract_patterns(self, df: pd.DataFrame) -> Dict:
        """Phân tích patterns từ dữ liệu hiện có"""
        patterns = {
            'symptoms': set(),
            'drugs': set(),
            'templates': defaultdict(list)
        }
        
        for _, row in df.iterrows():
            text = row['text'].lower()
            intent = row['intent']
            
            # Extract symptoms (từ symptom_inquiry)
            if intent == 'symptom_inquiry':
                # Tìm patterns phổ biến
                if 'bị' in text:
                    symptom = re.search(r'bị\s+(.+?)(?:\s|$)', text)
                    if symptom:
                        patterns['symptoms'].add(symptom.group(1).strip())
                        
                if 'triệu chứng' in text:
                    symptom = re.search(r'triệu chứng\s+(.+?)(?:\s|$)', text)
                    if symptom:
                        patterns['symptoms'].add(symptom.group(1).strip())
            
            # Extract drug names
            if intent in ['drug_question', 'dosage_question', 'side_effects']:
                # Tìm tên thuốc
                for drug_name in self.drug_variations.keys():
                    if drug_name in text:
                        patterns['drugs'].add(drug_name)
                        
            # Lưu template chung
            patterns['templates'][intent].append(text)
            
        return patterns
    
    def generate_variations(self, original_text: str, intent: str, count: int = 5) -> List[str]:
        """Tạo variations của một câu gốc"""
        variations = []
        original_lower = original_text.lower()
        
        if intent == 'symptom_inquiry':
            # Thay thế symptoms bằng synonyms
            for symptom, synonyms in self.symptom_variations.items():
                if symptom in original_lower:
                    for synonym in synonyms[:3]:  # Chỉ lấy 3 synonym
                        new_text = original_text.replace(symptom, synonym)
                        if new_text != original_text:
                            variations.append(new_text)
                            
            # Thay thế time expressions
            for time_expr in self.time_expressions:
                if any(t in original_lower for t in ['hôm nay', 'mấy ngày', 'tuần này']):
                    new_text = re.sub(r'(hôm nay|mấy ngày nay|tuần này|tháng này)', 
                                    time_expr, original_text, flags=re.IGNORECASE)
                    if new_text != original_text:
                        variations.append(new_text)
                        
        elif intent in ['drug_question', 'dosage_question', 'side_effects']:
            # Thay thế drug names bằng synonyms
            for drug, synonyms in self.drug_variations.items():
                if drug in original_lower:
                    for synonym in synonyms[:2]:  # Chỉ lấy 2 synonym
                        new_text = original_text.replace(drug, synonym)
                        if new_text != original_text:
                            variations.append(new_text)
        
        # Thêm variations về structure
        structure_variations = self._create_structure_variations(original_text, intent)
        variations.extend(structure_variations)
        
        # Loại bỏ trùng lặp và giới hạn số lượng
        unique_variations = list(set(variations))
        return unique_variations[:count]
    
    def _create_structure_variations(self, text: str, intent: str) -> List[str]:
        """Tạo variations về cấu trúc câu"""
        variations = []
        text_lower = text.lower()
        
        if intent == 'symptom_inquiry':
            # "Tôi bị X" -> "Có triệu chứng X", "Xuất hiện X"
            if text_lower.startswith('tôi bị'):
                symptom = text[7:]  # Bỏ "Tôi bị "
                variations.extend([
                    f"Có triệu chứng {symptom.lower()}",
                    f"Xuất hiện triệu chứng {symptom.lower()}",
                    f"Có dấu hiệu {symptom.lower()}"
                ])
            
            # "Có triệu chứng X" -> "Tôi bị X"
            elif 'có triệu chứng' in text_lower:
                symptom = re.search(r'có triệu chứng\s+(.+)', text_lower)
                if symptom:
                    variations.append(f"Tôi bị {symptom.group(1)}")
                    
        elif intent == 'drug_question':
            # "Thuốc X có tác dụng gì?" -> "X là thuốc gì?"
            if 'có tác dụng gì' in text_lower:
                drug = re.search(r'thuốc\s+(\w+)', text_lower)
                if drug:
                    variations.extend([
                        f"{drug.group(1)} là thuốc gì?",
                        f"Công dụng của {drug.group(1)}",
                        f"Tác dụng thuốc {drug.group(1)}"
                    ])
        
        return variations
    
    def generate_smart_combinations(self, df: pd.DataFrame, target_size: int = 5000) -> List[Dict]:
        """Tạo combinations thông minh"""
        augmented_data = []
        existing_texts = set(df['text'].str.lower())
        
        # Đầu tiên, copy tất cả data gốc
        for _, row in df.iterrows():
            augmented_data.append({
                'text': row['text'],
                'intent': row['intent'],
                'category': row['category'],
                'confidence': row['confidence']
            })
        
        # Phân tích intent distribution
        intent_counts = df['intent'].value_counts()
        print(f"📊 Intent distribution hiện tại:")
        for intent, count in intent_counts.items():
            print(f"  {intent}: {count}")
        
        # Calculate target distribution (balanced)
        target_per_intent = target_size // len(intent_counts)
        
        for intent in intent_counts.index:
            current_count = intent_counts[intent]
            needed = max(0, target_per_intent - current_count)
            
            print(f"🎯 {intent}: có {current_count}, cần thêm {needed}")
            
            if needed > 0:
                intent_data = df[df['intent'] == intent]
                generated = self._generate_for_intent(intent_data, needed, existing_texts)
                augmented_data.extend(generated)
                
                # Update existing_texts để tránh trùng lặp
                for item in generated:
                    existing_texts.add(item['text'].lower())
        
        return augmented_data
    
    def _generate_for_intent(self, intent_data: pd.DataFrame, needed: int, existing_texts: Set[str]) -> List[Dict]:
        """Generate data cho một intent cụ thể"""
        generated = []
        intent = intent_data.iloc[0]['intent']
        category = intent_data.iloc[0]['category']
        
        attempts = 0
        max_attempts = needed * 10  # Giới hạn số lần thử
        
        while len(generated) < needed and attempts < max_attempts:
            attempts += 1
            
            # Chọn random một sample gốc
            sample = intent_data.sample(1).iloc[0]
            original_text = sample['text']
            
            # Tạo variations
            variations = self.generate_variations(original_text, intent, count=10)
            
            for variation in variations:
                if variation.lower() not in existing_texts:
                    generated.append({
                        'text': variation,
                        'intent': intent,
                        'category': category,
                        'confidence': 0.8  # Lower confidence for generated data
                    })
                    existing_texts.add(variation.lower())
                    
                    if len(generated) >= needed:
                        break
            
            # Nếu không đủ variations, tạo từ template
            if len(generated) < needed:
                template_variations = self._generate_from_templates(intent, existing_texts)
                for variation in template_variations:
                    if len(generated) >= needed:
                        break
                    generated.append({
                        'text': variation,
                        'intent': intent,
                        'category': category,
                        'confidence': 0.7
                    })
                    existing_texts.add(variation.lower())
        
        print(f"  ✅ Đã tạo {len(generated)} mẫu cho {intent}")
        return generated
    
    def _generate_from_templates(self, intent: str, existing_texts: Set[str], count: int = 50) -> List[str]:
        """Tạo từ templates có sẵn - An toàn với missing variables"""
        generated = []
        
        if intent not in self.question_patterns:
            return generated
            
        templates = self.question_patterns[intent]
        
        # Chuẩn bị các variables để fill template
        variables = {
            'symptom': list(self.symptom_variations.keys()),
            'drug': list(self.drug_variations.keys()),
            'family_member': self.family_members,
            'age': ['trẻ em', 'người lớn', 'người già', 'bé', 'ông bà'],
            'severity': ['nhẹ', 'vừa phải', 'nặng', 'nghiêm trọng'],
            'duration': ['hôm qua', 'mấy ngày nay', 'tuần này', 'từ lâu']
        }
        
        for template in templates:
            # Tìm tất cả variables trong template
            import re
            template_vars = re.findall(r'\{(\w+)\}', template)
            
            if not template_vars:
                # Template không có biến
                if template.lower() not in existing_texts:
                    generated.append(template)
                continue
            
            # Tạo combinations cho các variables
            if len(template_vars) == 1:
                # Template có 1 biến
                var_name = template_vars[0]
                if var_name in variables:
                    for value in variables[var_name][:10]:  # Limit 10 per variable
                        try:
                            text = template.format(**{var_name: value})
                            if text.lower() not in existing_texts:
                                generated.append(text)
                        except (KeyError, ValueError):
                            continue
                            
            elif len(template_vars) == 2:
                # Template có 2 biến
                var1, var2 = template_vars
                if var1 in variables and var2 in variables:
                    for val1 in variables[var1][:5]:
                        for val2 in variables[var2][:5]:
                            try:
                                text = template.format(**{var1: val1, var2: val2})
                                if text.lower() not in existing_texts:
                                    generated.append(text)
                            except (KeyError, ValueError):
                                continue
            
            # Limit số lượng để tránh quá nhiều
            if len(generated) >= count:
                break
                
        return generated[:count]
    
    def remove_duplicates_and_similar(self, data: List[Dict], similarity_threshold: float = 0.9) -> List[Dict]:
        """Loại bỏ trùng lặp và tương tự"""
        from difflib import SequenceMatcher
        
        unique_data = []
        seen_texts = set()
        
        for item in data:
            text = item['text'].lower().strip()
            
            # Check exact duplicate
            if text in seen_texts:
                continue
                
            # Check similarity với existing texts
            is_similar = False
            for existing in seen_texts:
                similarity = SequenceMatcher(None, text, existing).ratio()
                if similarity > similarity_threshold:
                    is_similar = True
                    break
            
            if not is_similar:
                unique_data.append(item)
                seen_texts.add(text)
        
        return unique_data
    
    def save_augmented_dataset(self, data: List[Dict], output_path: str):
        """Lưu dataset đã augment"""
        df = pd.DataFrame(data)
        
        # Shuffle data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save with proper encoding
        df.to_csv(output_path, index=False, encoding='utf-8-sig', quoting=1)
        
        print(f"💾 Đã lưu {len(df)} mẫu vào {output_path}")
        
        # Print statistics
        print(f"\n📊 Thống kê dataset mới:")
        intent_dist = df['intent'].value_counts()
        for intent, count in intent_dist.items():
            print(f"  {intent}: {count}")
        
        print(f"\n📈 Confidence distribution:")
        conf_dist = df['confidence'].value_counts()
        for conf, count in conf_dist.items():
            print(f"  {conf}: {count}")

def main():
    print("🚀 Smart Dataset Augmentation")
    print("=" * 50)
    
    # Initialize augmenter
    augmenter = SmartDatasetAugmenter()
    
    # Load existing clean data
    input_file = 'e:/MedCare/data/medical_intent_dataset_v2_clean.csv'
    df = augmenter.load_existing_data(input_file)
    print(f"📂 Loaded {len(df)} clean samples")
    
    # Generate smart augmentation
    target_size = 5000
    augmented_data = augmenter.generate_smart_combinations(df, target_size)
    print(f"🎯 Generated {len(augmented_data)} total samples")
    
    # Remove duplicates and very similar texts
    print("🧹 Removing duplicates and similar texts...")
    clean_data = augmenter.remove_duplicates_and_similar(augmented_data, similarity_threshold=0.85)
    print(f"✅ After cleaning: {len(clean_data)} unique samples")
    
    # Save augmented dataset
    output_file = 'e:/MedCare/data/medical_intent_training_dataset_5k_smart.csv'
    augmenter.save_augmented_dataset(clean_data, output_file)
    
    print(f"\n🎉 Hoàn thành! Dataset thông minh đã được tạo:")
    print(f"   📁 Input: {input_file} ({len(df)} mẫu)")
    print(f"   📁 Output: {output_file} ({len(clean_data)} mẫu)")
    print(f"   📈 Tăng trưởng: {len(clean_data) - len(df)} mẫu mới")

if __name__ == "__main__":
    main()